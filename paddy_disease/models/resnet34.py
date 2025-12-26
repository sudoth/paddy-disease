from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


def build_resnet34(
    num_classes: int = 10, dropout: float = 0.1, pretrained: bool = True
) -> nn.Module:
    """
    Function that returns ResNet34 model

    :param num_classes: Description
    :type num_classes: int
    :param dropout: Description
    :type dropout: float
    :param pretrained: Description
    :type pretrained: bool
    :return: Description
    :rtype: Module
    """
    weights = ResNet34_Weights.DEFAULT if pretrained else None
    model = resnet34(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
