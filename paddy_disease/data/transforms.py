from torchvision import transforms

from paddy_disease.config import TransformsConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(cfg: TransformsConfig) -> transforms.Compose:
    aug = []
    if cfg.use_augmentations:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice(
                [
                    transforms.Pad(padding=10),
                    transforms.CenterCrop(480),
                    transforms.RandomRotation(20),
                    transforms.CenterCrop((576, 432)),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.1,
                    ),
                ]
            ),
        ]

    return transforms.Compose(
        [
            *aug,
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_test_transform(cfg: TransformsConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
