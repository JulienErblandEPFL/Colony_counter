from pathlib import Path
import cv2
import numpy as np
import os
import sys

# -------------------- Global parameters --------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

SRC_ROOT = Path(os.path.join(ROOT, "data", "raw"))
DST_ROOT = Path(os.path.join(ROOT, "data", "cropped_data"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Ring-template matching
RING_SCALES    = [0.035, 0.045, 0.055, 0.065]
RING_THICK_FR  = 0.18
RING_THRESH    = 0.38
MIN_RING_COUNT = 6

# Color (HSV) thresholds for purple/magenta wells
SAT_MIN = 50
VAL_MIN = 40
HUE_RANGES = [(120, 170), (0, 10)]

# HoughCircles
WELL_MIN_R_FR = 0.02
WELL_MAX_R_FR = 0.18
HOUGH_DP       = 1.2
HOUGH_MIN_DIST = 0.08
HOUGH_PARAM1   = 120
HOUGH_PARAM2   = 25

# --- BBox / Padding Rules (ADJUSTED FOR LOOSER CROP) ---
# PAD_FRAC is the extra empty space added relative to image size
PAD_FRAC = 0.01  
# ASPECT ratios for a 12-well plate
ASPECT_MIN, ASPECT_MAX = 0.9, 1.9
MAX_AREA_FRAC = 0.85

# --- NEW: acceptance + fallback ---
MIN_WELLS_FOR_CONFIDENT = 10
# Fallback rect (x_frac, y_frac, w_frac, h_frac) (if there is an error on the first crop)
FALLBACK_RECT_DEFAULT = (0.14, 0.56, 0.66, 0.36)
# Increased fallback padding to prevent tight cuts on failed detections
FALLBACK_PAD_FRAC = 0.05 
# -----------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clip(a, lo, hi):
    return max(lo, min(hi, a))

def expand_box(x, y, w, h, pad, W, H):
    """
    Expands the bounding box by 'pad' pixels on all sides,
    clipping to image boundaries.
    """
    x1 = clip(int(x - pad), 0, W-1)
    y1 = clip(int(y - pad), 0, H-1)
    
    # x2 and y2 calculation based on original w, h
    x2 = clip(int(x + w + pad), 0, W-1)
    y2 = clip(int(y + h + pad), 0, H-1)
    
    return x1, y1, x2 - x1, y2 - y1

def valid_bbox(w, h, W, H):
    ar = w / (h + 1e-6)
    area = w * h
    return (ASPECT_MIN <= ar <= ASPECT_MAX) and (area < MAX_AREA_FRAC * W * H)

# ---------- Detectors ----------

def make_ring_template(radius, thickness):
    R = int(round(radius))
    T = max(1, int(round(thickness)))
    yy, xx = np.mgrid[-R:R+1, -R:R+1]
    rr = np.sqrt(xx*xx + yy*yy)
    ring = ((rr >= (R - T)) & (rr <= (R + T))).astype(np.float32)
    ring = cv2.GaussianBlur(ring, (0,0), sigmaX=R*0.15 + 0.5)
    ring = ring - ring.mean()
    ring /= (ring.std() + 1e-6)
    return ring

def nms_peaks(res, thr, radius):
    mask = (res >= thr).astype(np.uint8)
    if mask.sum() == 0:
        return []
    k = max(3, int(radius*0.75)//2*2+1)
    dil = cv2.dilate(res, np.ones((k,k), np.uint8))
    peaks = (res == dil) & (res >= thr) & (mask > 0)
    ys, xs = np.where(peaks)
    return list(zip(xs, ys))

def detect_ring(gray):
    """Return (bbox, count) or (None, 0)."""
    H, W = gray.shape
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    centers, radii = [], []
    for fr in RING_SCALES:
        rad = max(8, int(min(H, W) * fr))
        tmpl = make_ring_template(rad, rad * RING_THICK_FR)
        res = cv2.matchTemplate(mag, tmpl, cv2.TM_CCOEFF_NORMED)
        for x, y in nms_peaks(res, RING_THRESH, rad):
            centers.append((x+rad, y+rad))
            radii.append(rad)

    count = len(centers)
    if count < MIN_RING_COUNT:
        return None, 0

    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])
    r  = np.mean(radii)

    # BBox around CENTERS
    x, y = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    
    # --- FIX FOR SHARPNESS ---
    # Previous logic used max(r, global_pad).
    # New logic: Pad = Radius (to reach edge) + 20% Radius (plastic rim) + Global Buffer
    pad = int(r * 1.2) + int(PAD_FRAC * min(H, W))
    
    x, y, w, h = expand_box(x, y, x2-x, y2-y, pad, W, H)
    if not valid_bbox(w,h,W,H):
        return None, 0
    return (x,y,w,h), count

def detect_purple(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_total = np.zeros((H,W), dtype=np.uint8)
    for lo_h, hi_h in HUE_RANGES:
        lower = np.array([lo_h, SAT_MIN, VAL_MIN], dtype=np.uint8)
        upper = np.array([hi_h, 255, 255], dtype=np.uint8)
        mask_total = cv2.bitwise_or(mask_total, cv2.inRange(hsv, lower, upper))
    
    mask = cv2.medianBlur(mask_total, 5)
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0

    minR = max(8, int(min(H, W) * WELL_MIN_R_FR))
    good = []
    radii = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi*minR*minR*0.4:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:  
            continue
        circularity = 4*np.pi*area/(perim*perim)
        if circularity < 0.6:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(cnt)
        good.append((cx,cy,rad))
        radii.append(rad)

    if not good:
        return None, 0

    xs = np.array([g[0] for g in good])
    ys = np.array([g[1] for g in good])
    r  = np.mean(radii) if radii else minR

    x, y = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    # --- FIX FOR SHARPNESS ---
    # Ensure we cover the radius + extra buffer
    pad = int(r * 1.2) + int(PAD_FRAC * min(H, W))

    x, y, w, h = expand_box(x, y, x2-x, y2-y, pad, W, H)
    if not valid_bbox(w,h,W,H):
        return None, 0
    return (x,y,w,h), len(good)

def detect_hough(gray):
    H, W = gray.shape
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 5)

    minR = max(8, int(min(H, W) * WELL_MIN_R_FR))
    maxR = int(min(H, W) * WELL_MAX_R_FR)

    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=int(min(H, W)*HOUGH_MIN_DIST),
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2, minRadius=minR, maxRadius=maxR
    )
    if circles is None:
        return None, 0
    circles = np.uint16(np.around(circles[0]))
    cnt = len(circles)
    if cnt < 4:
        return None, 0

    xs = circles[:,0]
    ys = circles[:,1]
    r = circles[:,2].mean()

    x, y = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    # --- FIX FOR SHARPNESS ---
    pad = int(r * 1.2) + int(PAD_FRAC * min(H, W))

    x, y, w, h = expand_box(x, y, x2-x, y2-y, pad, W, H)
    if not valid_bbox(w,h,W,H):
        return None, 0
    return (x,y,w,h), cnt

def detect_edges(gray):
    # This detector doesn't have a concept of "wells" or "radius", 
    # so we just increase the global padding
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(g, 50, 150)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = None
    best_score = -1
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if not valid_bbox(w,h,W,H):
            continue
        score = w*h
        if score > best_score:
            best_score = score
            best = (x,y,w,h)
    if best is None:
        return None
        
    # Increase padding for edge fallback as well
    pad = int(0.08 * min(W,H)) 
    x,y,w,h = expand_box(*best, pad, W, H)
    return (x,y,w,h)

# ---------- Main Decision ----------

def find_plate_bbox_and_count(img):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Ring-template
    bbox, count = detect_ring(gray)
    if bbox is not None and count >= MIN_WELLS_FOR_CONFIDENT:
        return bbox, count

    # 2) Color
    bbox, count = detect_purple(img)
    if bbox is not None and count >= MIN_WELLS_FOR_CONFIDENT:
        return bbox, count

    # 3) Hough
    bbox, count = detect_hough(gray)
    if bbox is not None and count >= MIN_WELLS_FOR_CONFIDENT:
        return bbox, count

    # 4) Edges
    bbox = detect_edges(gray)
    if bbox is not None:
        return bbox, 0

    return (0,0,W,H), 0

def crop_with_fallback(img, calibrated_rect_norm=None):
    H, W = img.shape[:2]
    bbox, count = find_plate_bbox_and_count(img)

    if count >= MIN_WELLS_FOR_CONFIDENT:
        # Confident detection
        x,y,w,h = bbox
        rect_norm = (x/W, y/H, w/W, h/H)
        return img[y:y+h, x:x+w], rect_norm, True

    # Fallback
    if calibrated_rect_norm is None:
        fx, fy, fw, fh = FALLBACK_RECT_DEFAULT
    else:
        fx, fy, fw, fh = calibrated_rect_norm

    x = int(fx * W)
    y = int(fy * H)
    w = int(fw * W)
    h = int(fh * H)

    # Make fallback padding bigger too
    pad = int(FALLBACK_PAD_FRAC * min(W, H))
    x, y, w, h = expand_box(x, y, w, h, pad, W, H)

    return img[y:y+h, x:x+w], calibrated_rect_norm, False

def process_folder(src_root: Path, dst_root: Path):
    files = [p for p in src_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        print(f"No images found under {src_root}")
        return

    ensure_dir(dst_root)

    calibrated_rect_norm = None

    #Create a list to store filenames that failed detection
    fallback_list = []

    for src in files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        ensure_dir(dst.parent)

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable: {src}")
            continue

        cropped, learned_rect, confident = crop_with_fallback(img, calibrated_rect_norm)

        if confident:
            calibrated_rect_norm = learned_rect
        else:
            #Add the file to the list if not confident
            fallback_list.append(rel)
            
        if cv2.imwrite(str(dst), cropped):
            tag = "detected" if confident else "fallback"
            print(f"Saved ({tag}): {dst}")
        else:
            print(f"Failed to save: {dst}")

    # Print Summary
    print("\n" + "="*50)
    if fallback_list:
        print(f"SUMMARY: {len(files) - len(fallback_list)}/{len(files)} images SUCCESSFULLY cropped !")
        print(f"  => {len(fallback_list)} image(s crop are not confident and need to be checked mannually:")
        for f in fallback_list:
            print(f"  [X] {f}")
    else:
        print("SUMMARY: Perfect run! All images were confidently detected.")
    print("="*50 + "\n")


def main():
    ensure_dir(DST_ROOT)
    process_folder(SRC_ROOT, DST_ROOT)

if __name__ == "__main__":
    main()