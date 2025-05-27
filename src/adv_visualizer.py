"""
    Generates and logs adversarial examples to disk with predictions and paths.
"""

import pandas as pd
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm

from src.logger import setup_logging

logger = setup_logging()

def generate_and_log_adv_examples(model, dataloader, device, attack_fns, save_dir="adv_images", results_file="adv_log.csv", max_images=1000):
    """
        Generate and log adversarial examples using specified attacks.

        Saves adversarial images and a CSV log with paths and prediction outcomes.

        Args:
            model (nn.Module): PyTorch model to attack.
            dataloader (DataLoader): DataLoader containing data to attack.
            device (torch.device): Device to run the model on.
            attack_fns (dict): Dictionary mapping attack names to functions.
            save_dir (str): Directory to save adversarial images.
            results_file (str): Name of the CSV file to log results.
            max_images (int): Maximum number of images to process.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    image_count = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Generating adversarial images")):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        if image_count >= max_images:
            break

        remaining = max_images - image_count
        process_count = min(batch_size, remaining)

        inputs = inputs[:process_count]
        targets = targets[:process_count]

        # get original predictions
        orig_preds = model(inputs).argmax(dim=1)

        # apply each attack
        adv_outputs = {}
        for attack_name, attack_fn in attack_fns.items():
            adv_inputs = attack_fn(model, inputs, targets)
            adv_outputs[attack_name] = {
                "images": adv_inputs,
                "preds": model(adv_inputs).argmax(dim=1)
            }

        # save individual results
        for i in range(process_count):
            idx = image_count + i
            label = targets[i].item()

            row = {
                "Index": idx,
                "Label": label,
            }

            # save original image and prediction result
            orig_path = save_dir / f"img_{idx}_original.png"
            save_image(inputs[i].cpu(), orig_path)
            row["Original image"] = str(orig_path)
            row["Original accuracy"] = int(orig_preds[i] == label)

            # save each adversarial image and its prediction
            for attack_name, output in adv_outputs.items():
                adv_img = output["images"][i].cpu()
                adv_pred = output["preds"][i]
                adv_path = save_dir / f"img_{idx}_atk_{attack_name}.png"
                save_image(adv_img, adv_path)
                row[f"Adv_{attack_name}"] = str(adv_path)
                row[f"{attack_name} accuracy"] = int(adv_pred == label)

            results.append(row)

        image_count += process_count

    # save all results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(save_dir / results_file, index=False)
    logger.info(f"Adversarial example log saved to {save_dir / results_file}")