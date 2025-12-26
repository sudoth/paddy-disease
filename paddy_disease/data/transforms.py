from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Train data transformation

    :param image_size: Description
    :type image_size: int
    :return: Description
    :rtype: Compose
    """
    return transforms.Compose(
        [
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
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_test_transform(image_size: int = 224) -> transforms.Compose:
    """
    Test data transform

    :param image_size: Description
    :type image_size: int
    :return: Description
    :rtype: Compose
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
