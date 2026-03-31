from typing import Dict

import torch.nn as nn
import torchvision.models as tv_models


SUPPORTED_MODELS = {"resnet18", "densenet121"}


def load_model(name: str) -> nn.Module:
    """Load a torchvision model by name with a 10-class output head.

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
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif name == "densenet121":
        model = tv_models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 10)

    return model
