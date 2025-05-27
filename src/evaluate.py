"""
    Evaluation utilities for clean and adversarial accuracy including AutoAttack.
"""

import torch
from autoattack import AutoAttack

from src.logger import setup_logging
from src.attacks import fgsm_attack, pgd_attack

logger = setup_logging()

def evaluate_natural(model, loader, device):
    """
        Evaluate model on clean (non-adversarial) data.

        Args:
            model (nn.Module): The model to evaluate.
            loader (DataLoader): Data loader.
            device (torch.device): Device to evaluate on.

        Returns:
            float: Accuracy percentage on clean data.
    """
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100. * correct / total

def evaluate_adversarial(model, loader, device, attack_fn):
    """
        Evaluate model using a specified adversarial attack.

        Args:
            model (nn.Module): The model to evaluate.
            loader (DataLoader): Data loader.
            device (torch.device): Device to evaluate on.
            attack_fn (callable): Function that returns adversarial examples.

        Returns:
            float: Accuracy percentage on adversarial examples.
    """
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(model, x, y)
        preds = model(x_adv).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100. * correct / total

def evaluate_with_autoattack(model, loader, device, epsilon=0.03):
    """
        Evaluate model using AutoAttack (Linf norm).

        Args:
            model (nn.Module): The model to evaluate.
            loader (DataLoader): Data loader.
            device (torch.device): Device to evaluate on.
            epsilon (float): Perturbation bound for attack.

        Returns:
            float: Accuracy percentage under AutoAttack.
    """
    model.eval()
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)

    x_all = torch.cat(all_x).to(device)
    y_all = torch.cat(all_y).to(device)
    adv_x = adversary.run_standard_evaluation(x_all, y_all, bs=128)

    with torch.no_grad():
        preds = model(adv_x).argmax(1)
        acc = (preds == y_all).float().mean().item() * 100
    return acc

def evaluate_all(model, loader, device, args):
    """
        Evaluate model on clean data and multiple adversarial attacks.

        Args:
            model (nn.Module): The model to evaluate.
            loader (DataLoader): Data loader.
            device (torch.device): Device to evaluate on.
            args (argparse.Namespace): Contains evaluation parameters like epsilon and steps.
    """
    logger.info("========== Evaluation results ==========")
    acc_clean = evaluate_natural(model, loader, device)
    logger.info(f"Clean accuracy: {acc_clean:.2f}%")

    model.eval()

    acc_fgsm = evaluate_adversarial(
        model, loader, device,
        lambda m, x, y: fgsm_attack(m, x, y, args.epsilon)
    )
    logger.info(f"FGSM accuracy (\u03b5={args.epsilon}): {acc_fgsm:.2f}%")

    acc_pgd = evaluate_adversarial(
        model, loader, device,
        lambda m, x, y: pgd_attack(m, x, y, args.epsilon, args.pgd_alpha, args.pgd_steps)
    )
    logger.info(f"PGD accuracy (\u03b5={args.epsilon}, steps={args.pgd_steps}): {acc_pgd:.2f}%")

    acc_auto = evaluate_with_autoattack(model, loader, device, epsilon=args.epsilon)
    logger.info(f"AutoAttack accuracy (\u03b5={args.epsilon}): {acc_auto:.2f}%")