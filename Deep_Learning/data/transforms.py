"""Image augmentation / preprocessing."""
from torchvision import transforms as T

from ..config import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(img_size: int = 384, train: bool = True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25),
        ])
    return T.Compose([
        T.Resize(img_size),            # short side -> img_size, keeps 4:3 aspect
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
