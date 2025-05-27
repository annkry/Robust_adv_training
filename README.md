# Robust adversarial training and evaluation framework

This project addresses adversarial robustness in deep learning using CIFAR-10. It integrates adversarial training (FGSM, PGD), AutoAttack evaluation, and MixedNUTS ensembles to explore the trade-off between accuracy and robustness. The framework supports hyperparameter tuning, logit analysis, and adversarial visualization through a modular PyTorch pipeline.

---

## Features

| module              | description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `main.py`           | CLI interface to run training, tuning, evaluation                           |
| `models.py`         | Definitions of the BaseNet and MixedNUTS models                             |
| `train.py`          | Clean and adversarial training loops                                        |
| `attacks.py`        | FGSM and PGD implementations                                                |
| `evaluate.py`       | Reports natural, FGSM, PGD, and AutoAttack accuracy                         |
| `tune.py`           | Grid search for MixedNUTS parameters (`s`, `p`, `c`, `alpha`)               |
| `tune_advtrain.py`  | Grid search for adversarial training params with JSON config saving         |
| `model_factory.py`  | Choose models like BaseNet, ResNet18                                        |
| `dataloaders.py`    | CIFAR-10 train/val/test loader setup                                        |
| `logger.py`         | Timestamped console logging                                                 |
| `logit_collector.py`| Collect logits from clean and adversarial examples                          |
| `adv_visualizer.py` | Generate adversarial images and log to CSV                                  |


---

## Supported evaluations

- clean accuracy
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- AutoAttack (L∞, untargeted)

---

## MixedNUTS defense

MixedNUTS combines predictions from an accurate model and a robust model via a nonlinear transformation and logit blending. Parameters `(s, p, c, alpha)` are automatically optimized using validation data.

---

# Configuration setups

The `tuned_models/best_mixednuts_config.json` file stores the configuration for the MixedNUTS model parameters and can be adjusted as needed.

The `tuned_models/best_advtrain_config.json` file stores the configuration for the adversarial training hyperparameters and can be adjusted as needed.

---
## How to run

Install dependencies
```bash
pip install -r requirements.txt
```

Standard training
```bash
python main.py --mode train --model-name basenet
```

Adversarial training hyperparameter tuning
```bash
python main.py --mode tune-advtrain --num-epochs 10 --config-path tuned_models/best_advtrain_config.json
```

Adversarial training (FGSM + PGD)
```bash
python main.py --mode adv-train --model-name basenet --epsilon-fgsm 0.03 --epsilon-pgd 0.03 --pgd-steps 40 --pgd-alpha 0.01
```

Load tuned config automatically
```bash
python main.py --mode adv-train --config-path tuned_models/best_advtrain_config.json
```

MixedNUTS parameter tuning
```bash
python main.py --mode tune --clean-model-path models/clean.pth --robust-model-path models/robust.pth --beta 0.7
```

Evaluation (clean + attacks) of MixedNUTS model
```bash
python main.py --mode evaluate --model-name mixednuts --beta 0.7
```

Evaluation (clean + attacks) of standard model
```bash
python main.py --mode evaluate --model-path models/clean.pth
```
Evaluation (clean + attacks) of robust model
```bash
python main.py --mode evaluate --model-path models/robust.pth
```

Switch architecture (e.g., ResNet18)
```bash
python main.py --mode train --model-name resnet18
```

Collect logits for analysis (MixedNUTS model)
```bash
python main.py --mode collect-logits --output-path logits_results.pt
```

Visualize adversarial examples for robust model
```bash
python main.py --mode visualize --model-path models/robust.pth --output-dir adv_images_robust
```
Visualize adversarial examples for clean model
```bash
python main.py --mode visualize --model-path models/clean.pth --output-dir adv_images_clean
```
Visualize adversarial examples for MixedNUTS model
```bash
python main.py --mode visualize --model-name mixednuts --output-dir adv_images_mixednuts
```

---

## Output directories

- `models/` - saved model checkpoints
- `tuned_models/` - best parameter values and tuning logs

---

# Performance example

Clean and robust models were trained on 100 epochs.

| Model                          | Clean accuracy | FGSM robust accuracy | PGD robust accuracy | Autoattack accuracy |
| ------------------------------ | -------------- | -------------------- | ------------------- |---------------------|
| Standard (clean)               | 71%            | 22%                  | 2%                  | 1%                  |
| Robust (adversarially trained) | 65%            | 38%                  | 14%                 | 12%                 |
| Mixed (β = 0.0)                | 71%            | 30%                  | 12%                 | 10%                 |
| Mixed (β = 0.7)                | 72%            | 34%                  | 14%                 | 11%                 |
| Mixed (β = 0.8)                | 73%            | 38%                  | 16%                 | 12%                 |

---

## References

- Bai, Yatong et al. *MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers*. 2024. arXiv:2402.02263 [cs.LG]. [arXiv link](https://arxiv.org/abs/2402.02263)

- PyTorch CIFAR Models: [https://github.com/chenyaofo/pytorch-cifar-models/tree/master](https://github.com/chenyaofo/pytorch-cifar-models/tree/master)

- PyTorch Adversarial Training CIFAR: [https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR/tree/master.](https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR/tree/master)

---