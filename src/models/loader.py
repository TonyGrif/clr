import torch.nn as nn
import torchvision.models as tv_models


SUPPORTED_MODELS = {"resnet18", "densenet121"}


def load_model(name: str) -> nn.Module:
    """Load a torchvision model adapted for CIFAR-10 (32x32 inputs).

    Replaces the initial 7x7 stride-2 conv and max pool (designed for
    ImageNet 224x224) with a 3x3 stride-1 conv and identity, preserving
    spatial resolution through the early layers.

    Args:
        name: Model name, one of 'resnet18' or 'densenet121'.

    Returns:
        Untrained model with output dimension 10.

    Raises:
        ValueError: If the model name is not supported.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{name}'. Choose from: {SUPPORTED_MODELS}")

    if name == "resnet18":
        model = tv_models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)

    elif name == "densenet121":
        model = tv_models.densenet121(weights=None)
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        model.classifier = nn.Linear(model.classifier.in_features, 10)

    return model
