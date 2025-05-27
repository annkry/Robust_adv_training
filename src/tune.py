"""
    Grid search for MixedNUTS model parameters: s, p, c, alpha.
"""

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from src.logger import setup_logging

logger = setup_logging()

def layer_normalization(logits):
    """
        Apply layer normalization to logits.

        Args:
            logits (Tensor): Logits tensor of shape (N, C)

        Returns:
            Tensor: Normalized logits.
    """
    mean = logits.mean(dim=1, keepdim=True)
    std = logits.std(dim=1, keepdim=True)
    return (logits - mean) / (std + 1e-5)

def nonlinear_transform(logits, s, p, c):
    """
        Apply nonlinear transformation to logits using parameters s, p, and c.

        Args:
            logits (Tensor): Input logits.
            s (float): Scaling parameter.
            p (float): Power parameter.
            c (float): Bias term.

        Returns:
            Tensor: Transformed logits.
    """
    logits = layer_normalization(logits)
    logits = F.gelu(logits + c)
    return s * torch.pow(torch.abs(logits), p) * torch.sign(logits)

def mix_logits(acc_logits, rob_logits, alpha):
    """
        Linearly blend accurate and robust logits.

        Args:
            acc_logits (Tensor): Clean model logits.
            rob_logits (Tensor): Robust model logits.
            alpha (float): Mixing coefficient.

        Returns:
            Tensor: Mixed log-probabilities.
    """
    acc_probs = F.softmax(acc_logits, dim=1)
    rob_probs = F.softmax(rob_logits, dim=1)
    mixed_probs = (1 - alpha) * acc_probs + alpha * rob_probs
    return torch.log(mixed_probs)

def compute_quantile(tensor, beta):
    """
        Compute the (1 - beta) quantile from a tensor.

        Args:
            tensor (Tensor): Input 1D tensor.
            beta (float): Quantile threshold.

        Returns:
            float: Quantile value.
    """
    k = int(np.ceil((1 - beta) * len(tensor)))
    return torch.topk(tensor, k=k, largest=False).values.max().item()

def tune_mixednuts_parameters(model, val_loader, device, beta=0.5):
    """
        Tune parameters (s, p, c, alpha) for MixedNUTS model to optimize adversarial robustness.

        Args:
            model (MixedNUTSNet): The MixedNUTS model.
            val_loader (DataLoader): Validation data loader.
            device (torch.device): Device to run evaluation on.
            beta (float): Robustness constraint threshold.

        Returns:
            tuple: Best parameters (s, p, c, alpha).
    """
    s_vals = np.linspace(0.5, 2.0, 50)
    p_vals = np.linspace(0.5, 2.0, 50)
    c_vals = np.linspace(-1.0, 1.0, 50)

    best_acc = 0.0
    best_params = (1.0, 1.0, 0.0, 0.5)

    acc_logits_all, rob_logits_all, labels_all = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            acc_logits_all.append(model.accurate(x))
            rob_logits_all.append(model.robust(x))
            labels_all.append(y)

    acc_logits = torch.cat(acc_logits_all)
    rob_logits = torch.cat(rob_logits_all)
    labels = torch.cat(labels_all)

    acc_preds = acc_logits.argmax(dim=1)
    rob_preds = rob_logits.argmax(dim=1)

    # identify the cases where robust model is correct and accurate model is wrong
    mask_wrong_acc = acc_preds != labels
    mask_correct_rob = rob_preds == labels

    std_logits_wrong = acc_logits[mask_wrong_acc]
    rob_logits_correct = rob_logits[mask_correct_rob]
    true_labels_wrong = labels[mask_wrong_acc]
    true_labels_correct = labels[mask_correct_rob]

    for s in tqdm(s_vals, desc="Tuning"):
        for p in p_vals:
            for c in c_vals:
                transformed_robust_logits = nonlinear_transform(rob_logits, s, p, c)
                transformed_robust_correct = transformed_robust_logits[mask_correct_rob]
                transformed_robust_wrong = transformed_robust_logits[mask_wrong_acc]

                # margins for correct robust predictions
                top2 = torch.topk(transformed_robust_correct, 2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                q = compute_quantile(margin, beta)
                alpha = 1 / (1 + q)

                # mix logits and compute accuracy on robust-correct, acc-wrong samples
                mixed_logits = mix_logits(std_logits_wrong, transformed_robust_wrong, alpha)
                preds = mixed_logits.argmax(dim=1)
                acc = (preds == true_labels_wrong).float().mean().item()

                if acc > best_acc:
                    best_acc = acc
                    best_params = (s, p, c, alpha)

    model.s, model.p, model.c, model.alpha = best_params
    logger.info(f"Best MixedNUTS params: s={model.s}, p={model.p}, c={model.c}, alpha={model.alpha}, acc={best_acc:.2%}")
    return best_params