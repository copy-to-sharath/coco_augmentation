import albumentations as A
from albumentations.pytorch import ToTensorV2

# Augmentation pipeline
TRANSFORMATIONS = [
    # A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.3),
    A.ToGray(p=0.1),
    ToTensorV2()
]
