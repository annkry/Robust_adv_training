"""
    Collects logits from clean and adversarial examples for later analysis.
"""

import torch
from pathlib import Path
from tqdm import tqdm

from src.logger import setup_logging

logger = setup_logging()

def collect_logits(clean_model, robust_model, dataloader, device, attack_fns, save_path="logits_results.pt"):
    """
        Collect logits for clean and adversarial examples from both clean and robust models.

        Args:
            clean_model (nn.Module): Clean model.
            robust_model (nn.Module): Robust model.
            dataloader (DataLoader): DataLoader for input data.
            device (torch.device): Torch device to use.
            attack_fns (dict): Dictionary of adversarial attack functions.
            save_path (str): Path to save the results file.
    """
    clean_model.eval()
    robust_model.eval()
    results = []

    for x, y in tqdm(dataloader, desc="Collecting logits"):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            clean_logits_clean_model = clean_model(x)
            clean_logits_robust_model = robust_model(x)

        entry = {
            'True label': y.cpu(),
            'Clean logits clean model': clean_logits_clean_model.cpu(),
            'Clean Logits robust model': clean_logits_robust_model.cpu()
        }

        # apply all adversarial attacks and collect logits from both models
        for name, attack_fn in attack_fns.items():
            x_adv = attack_fn(clean_model, x, y)
            entry[f'Adv {name.upper()} logits clean model'] = clean_model(x_adv).detach().cpu()

            x_adv_rob = attack_fn(robust_model, x, y)
            entry[f'Adv {name.upper()} logits robust model'] = robust_model(x_adv_rob).detach().cpu()

        # decompose batch entries to individual sample records
        results.extend([{k: v[i] for k, v in entry.items()} for i in range(x.size(0))])

    save_path = Path(save_path)
    torch.save(results, save_path)
    logger.info(f"Saved logits to {save_path.resolve()}")