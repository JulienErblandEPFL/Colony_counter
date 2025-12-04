from typing import Sequence, Optional
from torchvision import transforms

# Default image size
DEFAULT_IMG_SIZE = 320 # 224 or 256 or 320

# Common normalization profile for ImageNet-pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Recommended simpler profile for colony wells
COLONY_MEAN = [0.5, 0.5, 0.5]
COLONY_STD  = [0.25, 0.25, 0.25]



def _build_normalization(
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    use_imagenet_stats: bool,
):
    """
    Priority:
    1) If mean & std are provided -> use those.
    2) Else if use_imagenet_stats=True -> use ImageNet stats.
    3) Else -> use COLONY_MEAN/STD as default.
    """

    # 1) Explicit custom stats
    if mean is not None and std is not None:
        return transforms.Normalize(mean=mean, std=std)

    # 2) Manual override: use ImageNet normalization
    if use_imagenet_stats:
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # 3) Default colony-preferred normalization
    return transforms.Normalize(mean=COLONY_MEAN, std=COLONY_STD)


def get_classifier_train_transforms(
    img_size: int = 224,
    use_imagenet_stats: bool = False,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """
    Transforms for training the countability classifier.
    They preserve:
      - global plate structure
      - colony density cues
      - staining patterns
    Only light augmentations are used.
    """
    tfms = [
        transforms.Resize((img_size, img_size)),

        # Mild geometric variations
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),

        transforms.ToTensor(),
    ]

    # Normalization logic
    if mean is not None and std is not None:
        tfms.append(transforms.Normalize(mean=mean, std=std))

    elif use_imagenet_stats:
        tfms.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )

    else:
        # Default sensible stats for purple-staining wells
        tfms.append(
            transforms.Normalize(mean=COLONY_MEAN, std=COLONY_STD)
        )

    return transforms.Compose(tfms)

def get_classifier_test_transforms(
    img_size: int = 224,
    use_imagenet_stats: bool = False,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """
    Validation/Test transforms for the classifier:
    no distortion, deterministic, clean images.
    """
    tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    # Normalization confidence
    if mean is not None and std is not None:
        tfms.append(transforms.Normalize(mean=mean, std=std))

    elif use_imagenet_stats:
        tfms.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )

    else:
        tfms.append(
            transforms.Normalize(mean=COLONY_MEAN, std=COLONY_STD)
        )

    return transforms.Compose(tfms)


def get_counter_train_transforms(
    img_size: int = DEFAULT_IMG_SIZE,
    use_imagenet_stats: bool = False,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """
    Train-time transforms optimized for colony structure.
    These improve:
      - contrast
      - color bias robustness
      - texture discrimination
      - spatial reasoning
    """

    tfms = [
        transforms.Resize((img_size, img_size)),

        # Reveal colony structure in dense plates
        transforms.RandomAutocontrast(p=0.35),

        # Make model invariant to staining and illumination variation
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.08,
        ),

        # Reduce pixel-noise overfitting,
        # enforce colony-scale structure learning
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.20
        ),

        # Very useful for colony density modeling
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.9, 1.0),
            ratio=(0.98, 1.02),
        ),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(10),

        transforms.ToTensor(),
    ]

    # Default normalization: colony-specific stats
    if mean is None and std is None and not use_imagenet_stats:
        tfms.append(
            transforms.Normalize(mean=COLONY_MEAN, std=COLONY_STD)
        )
    else:
        norm = _build_normalization(
            mean=mean, std=std, use_imagenet_stats=use_imagenet_stats
        )
        if norm is not None:
            tfms.append(norm)

    return transforms.Compose(tfms)


def get_counter_test_transforms(
    img_size: int = DEFAULT_IMG_SIZE,
    use_imagenet_stats: bool = False,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """
    Test/validation transforms (no augmentation).
    Clean transform pipeline: no distortions, no randomness.
    """
    tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    # Same logic for normalization
    if mean is None and std is None and not use_imagenet_stats:
        tfms.append(
            transforms.Normalize(mean=COLONY_MEAN, std=COLONY_STD)
        )
    else:
        norm = _build_normalization(
            mean=mean, std=std, use_imagenet_stats=use_imagenet_stats
        )
        if norm is not None:
            tfms.append(norm)

    return transforms.Compose(tfms)
