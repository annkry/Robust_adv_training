"""
    Model definitions for BaseNet and MixedNUTS ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    """A simple convolutional neural network for CIFAR-10 classification."""

    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
            Forward pass through the network.

            Args:
                x (Tensor): Input tensor.

            Returns:
                Tensor: Output logits.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MixedNUTSNet(nn.Module):
    """A model combining an accurate and a robust network with nonlinear mixing for robustness."""
     
    def __init__(self, device):
        super(MixedNUTSNet, self).__init__()
        self.accurate = BaseNet().to(device)
        self.robust = BaseNet().to(device)
        self.device = device
        self.s, self.p, self.c, self.alpha = 1.0, 1.0, 0.0, 0.5

    def forward(self, x):
        """
            Forward pass mixing accurate and transformed robust logits.

            Args:
                x (Tensor): Input tensor.

            Returns:
                Tensor: Log probabilities from the mixed model.
        """
        acc_logits = self.accurate(x)
        rob_logits = self.robust(x)
        rob_logits = self.transform_logits(rob_logits)
        return self.mix_logits(acc_logits, rob_logits)

    def transform_logits(self, logits):
        """
            Apply a nonlinear transformation to the logits.

            Args:
                logits (Tensor): Robust logits.

            Returns:
                Tensor: Transformed logits.
        """
        logits = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + 1e-5)
        logits = F.gelu(logits + self.c)
        return self.s * torch.pow(torch.abs(logits), self.p) * torch.sign(logits)

    def mix_logits(self, acc_logits, rob_logits):
        """
            Mix accurate and robust probabilities using a weighted average.

            Args:
                acc_logits (Tensor): Accurate logits.
                rob_logits (Tensor): Transformed robust logits.

            Returns:
                Tensor: Logarithm of mixed probabilities.
        """
        acc_probs = F.softmax(acc_logits, dim=1)
        rob_probs = F.softmax(rob_logits, dim=1)
        mixed_probs = (1 - self.alpha) * acc_probs + self.alpha * rob_probs
        return torch.log(mixed_probs)

    def load(self, path):
        """
            Load model parameters and mixing parameters from a file.

            Args:
                path (str): Path to the model file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.accurate.load_state_dict(checkpoint['accurate'])
        self.robust.load_state_dict(checkpoint['robust'])
        self.s = checkpoint['s']
        self.p = checkpoint['p']
        self.c = checkpoint['c']
        self.alpha = checkpoint['alpha']

    def save(self, path):
        """
            Save model and mixing parameters to a file.

            Args:
                path (str): Destination path for the model file.
        """
        torch.save({
            'accurate': self.accurate.state_dict(),
            'robust': self.robust.state_dict(),
            's': self.s,
            'p': self.p,
            'c': self.c,
            'alpha': self.alpha
        }, path)

    def load_models(self, acc_path, rob_path):
        """
            Load only the accurate and robust model weights.

            Args:
                acc_path (str): Path to accurate model.
                rob_path (str): Path to robust model.
        """
        self.accurate.load_state_dict(torch.load(acc_path, map_location=self.device, weights_only=True))
        self.robust.load_state_dict(torch.load(rob_path, map_location=self.device, weights_only=True))