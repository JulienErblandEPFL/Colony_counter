from typing import Sequence, Optional
from torchvision import transforms

# Default image size
DEFAULT_IMG_SIZE = 224

# Common normalization profile for ImageNet-pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_normalization(
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    use_imagenet_stats: bool,
):
    """
    Priority:
    1) If mean & std are provided -> use those.
    2) Else if use_imagenet_stats -> use ImageNet stats.
    3) Else -> no normalization.
    """
    if mean is not None and std is not None:
        return transforms.Normalize(mean=mean, std=std)

    if use_imagenet_stats:
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return None


def get_train_transforms(
    img_size: int = DEFAULT_IMG_SIZE,
    use_imagenet_stats: bool = True,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """
    Train-time transforms for 3-channel (RGB) models.
    """
    tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]

    norm = _build_normalization(mean=mean, std=std, use_imagenet_stats=use_imagenet_stats)
    if norm is not None:
        tfms.append(norm)

    return transforms.Compose(tfms)


def get_test_transforms(
    img_size: int = DEFAULT_IMG_SIZE,
    use_imagenet_stats: bool = True,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """
    Test/validation transforms (no augmentation), for 3-channel (RGB) models.
    """
    tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    norm = _build_normalization(mean=mean, std=std, use_imagenet_stats=use_imagenet_stats)
    if norm is not None:
        tfms.append(norm)

    return transforms.Compose(tfms)
