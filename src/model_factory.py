"""
    Factory function to create and return different model architectures.
"""

import torch
from torchvision.models import resnet18

from src.models import BaseNet

def create_model(name: str, num_classes=10, pretrained=False):
    """
        Factory function to create model instances based on a given name.

        Args:
            name (str): Model identifier string (e.g., 'basenet', 'resnet18').
            num_classes (int): Number of output classes. Default is 10.
            pretrained (bool): Whether to load pretrained weights for resnet18. Default is False.

        Returns:
            torch.nn.Module: Instantiated model.

        Raises:
            ValueError: If an unsupported model name is given.
    """
    name = name.lower()
    if name == 'basenet':
        return BaseNet()
    elif name == 'resnet18':
        model = resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Model '{name}' is not supported.")