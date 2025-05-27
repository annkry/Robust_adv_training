"""
    Implements adversarial attack methods: FGSM and PGD.
"""

import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon):
    """
        Fast Gradient Sign Method (FGSM) adversarial attack.

        Args:
            model (nn.Module): Model to attack.
            images (Tensor): Input images.
            labels (Tensor): True labels for inputs.
            epsilon (float): Perturbation magnitude.

        Returns:
            Tensor: Adversarial examples.
    """
    images = images.clone().detach().requires_grad_(True)
    model.zero_grad()
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    grad_sign = images.grad.data.sign()
    return torch.clamp(images + epsilon * grad_sign, 0, 1)

def pgd_attack(model, images, labels, epsilon, alpha, steps):
    """
        Projected Gradient Descent (PGD) adversarial attack.

        Args:
            model (nn.Module): Model to attack.
            images (Tensor): Input images.
            labels (Tensor): True labels.
            epsilon (float): Maximum perturbation.
            alpha (float): Step size for each PGD iteration.
            steps (int): Number of PGD steps.

        Returns:
            Tensor: Adversarial examples.
    """
    images = images.clone().detach()
    delta = torch.zeros_like(images, requires_grad=True)

    for _ in range(steps):
        outputs = model(images + delta)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        delta.data = (delta + alpha * delta.grad.sign()).clamp(-epsilon, epsilon)
        delta.data = (images + delta.data).clamp(0, 1) - images
        delta.grad.zero_()

    return torch.clamp(images + delta.detach(), 0, 1)