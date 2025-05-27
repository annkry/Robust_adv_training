"""
    Defines training routines for standard and adversarial training using FGSM and PGD.
"""

import torch.nn as nn
import torch.optim as optim

from src.attacks import fgsm_attack, pgd_attack
from src.logger import setup_logging

logger = setup_logging()

def train_standard(model, train_loader, device, args):
    """
        Train a model using standard (non-adversarial) training.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): Training data loader.
            device (torch.device): Device to run training on.
            args (argparse.Namespace): Contains training hyperparameters.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"[standard] Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss / len(train_loader):.4f}")


def train_adversarial(model, train_loader, device, args):
    """
        Train a model using adversarial training combining clean, FGSM, and PGD examples.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): Training data loader.
            device (torch.device): Device to run training on.
            args (argparse.Namespace): Contains training hyperparameters including epsilon and loss weights.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # generate adversarial samples
            fgsm_inputs = fgsm_attack(model, inputs, labels, args.epsilon_fgsm)
            pgd_inputs = pgd_attack(model, inputs, labels, args.epsilon_pgd, args.pgd_alpha, args.pgd_steps)

            optimizer.zero_grad()

            # compute loss for clean and adversarial examples
            loss_clean = criterion(model(inputs), labels)
            loss_fgsm = criterion(model(fgsm_inputs), labels)
            loss_pgd = criterion(model(pgd_inputs), labels)

            # weighted combination of losses
            total_loss = (
                args.clean_weight * loss_clean +
                args.fgsm_weight * loss_fgsm +
                args.pgd_weight * loss_pgd
            )

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        logger.info(f"[adversarial] Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss / len(train_loader):.4f}")