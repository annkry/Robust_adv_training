"""
    Main entry point for robust training pipeline. Supports training, tuning, evaluation, visualization, and logit collection.
"""

import os
import json
import torch
import argparse
from autoattack import AutoAttack
from torchattacks import PGDL2

from src.logger import setup_logging
from src.dataloaders import get_data_loaders
from src.model_factory import create_model
from src.models import MixedNUTSNet
from src.train import train_standard, train_adversarial
from src.evaluate import evaluate_all
from src.tune import tune_mixednuts_parameters
from src.tune_advtrain import tune_adversarial_training
from src.adv_visualizer import generate_and_log_adv_examples
from src.logit_collector import collect_logits

# set up a global logger
logger = setup_logging()

def load_mixednuts_model(device, args):
    """
        Load a MixedNUTSNet model from specified paths and apply tuned parameters if available.

        Args:
            device (torch.device): The device to load the model on.
            args (argparse.Namespace): Arguments including paths and beta value.

        Returns:
            MixedNUTSNet: Configured model instance.
    """
    model = MixedNUTSNet(device)
    model.load_models(args.clean_model_path, args.robust_model_path)

    mixednuts_config_path = os.path.join(args.save_dir, "best_mixednuts_config.json")
    beta_key = f"beta_{args.beta:.2f}"

    # load tuned parameters if available
    if os.path.exists(mixednuts_config_path):
        with open(mixednuts_config_path, 'r') as f:
            params = json.load(f)

        if beta_key in params:
            beta_params = params[beta_key]
            model.s = beta_params.get("s", model.s)
            model.p = beta_params.get("p", model.p)
            model.c = beta_params.get("c", model.c)
            model.alpha = beta_params.get("alpha", model.alpha)
            logger.info(f"Loaded MixedNUTS config for beta={beta_key} from {mixednuts_config_path}")
        else:
            logger.warning(f"No config for beta={beta_key}, using default model parameters.")
    else:
        logger.warning(f"MixedNUTS config file not found at {mixednuts_config_path}, using defaults.")

    return model


def main(args):
    """
        Entry point for handling different pipeline modes like training, tuning, evaluation, etc.

        Args:
            args (argparse.Namespace): Parsed command line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # load data
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size, val_size=args.valid_size)

    # handle each mode of operation
    if args.mode == 'train':
        logger.info("Starting standard training.")
        model = create_model(args.model_name).to(device)
        train_standard(model, train_loader, device, args)
        os.makedirs(os.path.dirname(args.clean_model_path), exist_ok=True)
        torch.save(model.state_dict(), args.clean_model_path)
        logger.info(f"Saved standard model to '{args.clean_model_path}'")

    elif args.mode == 'adv-train':
        logger.info("Starting adversarial training.")
        model = create_model(args.model_name).to(device)

        # load tuned adversarial training config if present
        if args.config_path and os.path.exists(args.config_path):
            with open(args.config_path, 'r') as f:
                params = json.load(f)
            args.epsilon_fgsm = params.get("epsilon_fgsm", args.epsilon_fgsm)
            args.epsilon_pgd = params.get("epsilon_pgd", args.epsilon_pgd)
            args.pgd_alpha = params.get("pgd_alpha", args.pgd_alpha)
            args.pgd_steps = params.get("pgd_steps", args.pgd_steps)
            args.clean_weight = params.get("clean_weight", args.clean_weight)
            args.fgsm_weight = params.get("fgsm_weight", args.fgsm_weight)
            args.pgd_weight = params.get("pgd_weight", args.pgd_weight)
            logger.info(f"Loaded adversarial training parameters from a file '{args.config_path}'")
        train_adversarial(model, train_loader, device, args)
        os.makedirs(os.path.dirname(args.robust_model_path), exist_ok=True)
        torch.save(model.state_dict(), args.robust_model_path)
        logger.info(f"Saved robust model to '{args.robust_model_path}'")

    elif args.mode == 'tune':
        logger.info("Tuning MixedNUTS parameters.")
        model = MixedNUTSNet(device)
        model.load_models(args.clean_model_path, args.robust_model_path)
        save_path = os.path.join(args.save_dir, "best_mixednuts_config.json")

        best_params = tune_mixednuts_parameters(model, val_loader, device, beta=args.beta)

        beta_key = f"beta_{args.beta:.2f}"
        new_config = {
            "s": best_params[0],
            "p": best_params[1],
            "c": best_params[2],
            "alpha": best_params[3]
        }

        # load or initialize the JSON file
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                all_configs = json.load(f)
        else:
            all_configs = {}

        all_configs[beta_key] = new_config

        with open(save_path, "w") as f:
            json.dump(all_configs, f, indent=4)

        logger.info(f"Saved best MixedNUTS parameters for beta={args.beta:.2f} to {save_path}")

    elif args.mode == 'evaluate':
        logger.info("Evaluating model.")
        if args.model_name == 'mixednuts':
            model = load_mixednuts_model(device, args)
        else:
            model = create_model(args.model_name).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))

        evaluate_all(model, test_loader, device, args)

    elif args.mode == 'tune-advtrain': 
        logger.info("Tuning adversarial training parameters...")
        param_grid = {
            'epsilon_fgsm': [0.01, 0.03],
            'epsilon_pgd':  [0.01, 0.03],
            'pgd_alpha':    [0.004, 0.01],
            'pgd_steps':    [10, 40],
            'clean_weight': [0.2, 0.4],
            'fgsm_weight':  [0.3, 0.4],
            'pgd_weight':   [0.3, 0.4]
        }
        tune_adversarial_training(param_grid, train_loader, val_loader, lr=args.lr, save_dir=args.save_dir, max_epochs=args.num_epochs, save_config=True, config_path=args.config_path)
    
    elif args.mode == 'visualize':
        logger.info("Generating adversarial images...")
        if args.model_name == 'mixednuts':
            model = load_mixednuts_model(device, args)
        else:
            model = create_model(args.model_name).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model.eval()
        attack_fns = {
            "AutoLinf": lambda model, x, y: AutoAttack(model, norm='Linf', eps=0.03).run_standard_evaluation(x, y, bs=x.size(0)),
            "PGD_L2": lambda model, x, y: PGDL2(model, eps=0.03, alpha=0.01, steps=40)(x, y)
        }
        generate_and_log_adv_examples(model=model, dataloader=test_loader, device=device, attack_fns=attack_fns, save_dir=args.output_dir, results_file="summary.csv", max_images=1000)
    
    elif args.mode == 'collect-logits':
        logger.info("Collecting logits from clean and adversarial inputs...")
        model = load_mixednuts_model(device, args)
        attack_fns = {
            "AutoLinf": lambda model, x, y: AutoAttack(model, norm='Linf', eps=0.03).run_standard_evaluation(x, y, bs=x.size(0)),
            "PGD_L2": lambda model, x, y: PGDL2(model, eps=0.03, alpha=0.01, steps=40)(x, y)
        }
        collect_logits(model.accurate, model.robust, test_loader, device, attack_fns, save_path=args.output_path)
    else:
        raise ValueError("Invalid mode. Choose from: train, adv-train, tune, evaluate, tune-advtrain, visualize, collect-logits.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust training pipeline")

    parser.add_argument('--mode', type=str, required=True, choices=['train', 'adv-train', 'tune', 'evaluate', 'tune-advtrain', 'visualize', 'collect-logits'], help="Execution mode: training, tuning, evaluation, etc.")
    parser.add_argument('--num-epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size for data loaders.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--valid-size', type=int, default=1024, help="Validation set size.")
    parser.add_argument('--model-path', type=str, default='models/robust.pth', help="Path to load model for evaluation or visualization.")
    parser.add_argument('--clean-model-path', type=str, default='models/clean.pth', help="Path to save/load clean model.")
    parser.add_argument('--robust-model-path', type=str, default='models/robust.pth', help="Path to save/load robust model.")
    parser.add_argument('--epsilon', type=float, default=0.03, help="Attack perturbation budget.")
    parser.add_argument('--pgd-alpha', type=float, default=0.01, help="PGD attack step size.")
    parser.add_argument('--pgd-steps', type=int, default=40, help="Number of PGD attack steps.")
    parser.add_argument('--epsilon-fgsm', type=float, default=0.03, help="FGSM attack epsilon.")
    parser.add_argument('--epsilon-pgd', type=float, default=0.03, help="PGD attack epsilon.")
    parser.add_argument('--clean-weight', type=float, default=0.37, help="Loss weight for clean data.")
    parser.add_argument('--fgsm-weight', type=float, default=0.26, help="Loss weight for FGSM examples.")
    parser.add_argument('--pgd-weight', type=float, default=0.37, help="Loss weight for PGD examples.")
    parser.add_argument('--save-dir', type=str, default='tuned_models', help="Directory to save tuning results.")
    parser.add_argument('--model-name', type=str, choices=['basenet', 'mixednuts', 'resnet18'], default='basenet', help="Model architecture to use.")
    parser.add_argument('--config-path', type=str, default=None, help="Path to adversarial training config JSON.")
    parser.add_argument('--output-dir', type=str, default='adv_images', help="Directory for saving adversarial images.")
    parser.add_argument('--output-path', type=str, default='logits_results.pt', help="Path for saving logits.")
    parser.add_argument('--beta', type=float, default=0.5, help="Robustness constraint for MixedNUTS tuning.")

    args = parser.parse_args()
    main(args)