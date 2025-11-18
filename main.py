from crop_wells import crop_wells
import cv2

def main():
    file_path = "test_data/PR8_Plate_1.jpg"
    plate_type = "PR8"

    # Run the cropping function and receive the dict of images
    cropped = crop_wells(
        file_path=file_path,
        plate_type=plate_type,
        debug=True,
        save_files=True
    )

if __name__ == "__main__":
    main()
