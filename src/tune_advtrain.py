"""
    Hyperparameter tuning for adversarial training using validation accuracy.
"""

import os
import csv
import json
import torch
import itertools

from src.models import BaseNet
from src.train import train_adversarial
from src.evaluate import evaluate_natural
from src.logger import setup_logging

logger = setup_logging()

def tune_adversarial_training(param_grid, train_loader, val_loader, lr=0.001, save_dir="tuned_models", max_epochs=10, save_config=False, config_path=None):
    """
        Grid search over adversarial training hyperparameters to maximize validation accuracy.

        Args:
            param_grid (dict): Dictionary of parameter lists to try.
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            lr (float): Learning rate.
            save_dir (str): Directory to save results.
            max_epochs (int): Training epochs.
            save_config (bool): Whether to save best config as JSON.
            config_path (str): Path to save the best config.

        Returns:
            tuple: Best parameter configuration and its accuracy.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys, values = zip(*param_grid.items())
    best_acc = 0
    best_config = None

    with open(os.path.join(save_dir, "tuning_log.csv"), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(keys) + ["val_accuracy"])

        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            logger.info(f"Testing configuration: {config}")

            model = BaseNet().to(device)
            train_adversarial(
                model,
                train_loader,
                device,
                # construct a dummy args object with needed attributes
                args=type('Args', (object,), {
                    'lr': lr,
                    'num_epochs': max_epochs,
                    'epsilon_fgsm': config['epsilon_fgsm'],
                    'epsilon_pgd': config['epsilon_pgd'],
                    'pgd_steps': config['pgd_steps'],
                    'pgd_alpha': config['pgd_alpha'],
                    'clean_weight': config['clean_weight'],
                    'fgsm_weight': config['fgsm_weight'],
                    'pgd_weight': config['pgd_weight']
                })
            )

            acc = evaluate_natural(model, val_loader, device)
            logger.info(f"Validation accuracy: {acc}%")

            writer.writerow(list(combo) + [acc])

            if acc > best_acc:
                best_acc = acc
                best_config = config

    if save_config and best_config is not None:
        with open(config_path, 'w') as json_file:
            json.dump(best_config, json_file, indent=4)
        logger.info(f"Best configuration saved to {config_path}")

    logger.info(f"Best config: {best_config} with accuracy {best_acc}%")
    return best_config, best_acc