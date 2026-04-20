# Self-Pruning Neural Network — Tredence AI Engineering Case Study

A feed-forward neural network that **learns to prune itself during training** via
learnable gate parameters and L1 sparsity regularization, applied to CIFAR-10
image classification.

---

## Overview

Standard pruning removes weights *after* training. This project goes further:
each weight `w_ij` has a corresponding **gate score** `s_ij`. During the forward
pass, `gates = sigmoid(s_ij)` multiplies the weight, creating *pruned_weights*.
An **L1 penalty** on all gate values pushes most gates toward exactly 0 during
training, effectively removing those connections from the network.

```
Total Loss = CrossEntropyLoss + λ × Σ sigmoid(gate_scores)
```

---

## Project Structure

```
.
├── self_pruning_network.py   # Main implementation (all parts)
├── report.md                 # Analysis report
├── outputs/
│   ├── gate_distributions.png
│   └── training_curves.png
└── README.md
```

---

## Implementation Highlights

### `PrunableLinear` (Part 1)
- Custom `nn.Module` replacing `nn.Linear`
- Adds a `gate_scores` parameter tensor (same shape as `weight`)
- Forward pass: `gates = sigmoid(gate_scores)` → `pruned_weights = weight * gates` → `F.linear(...)`
- Gradients flow through both `weight` and `gate_scores` via autograd

### Sparsity Loss (Part 2)
- `SparsityLoss = Σ sigmoid(gate_scores)` across all layers (L1 norm of gates)
- Controlled by hyperparameter λ

### Training & Evaluation (Part 3)
- Adam optimizer with cosine LR annealing
- Evaluated at three λ values: `1e-4` (low), `1e-3` (medium), `5e-3` (high)
- Reports: test accuracy, sparsity level, gate distribution plots

---

## Requirements

```
torch
torchvision
matplotlib
numpy
```

Install with:
```bash
pip install torch torchvision matplotlib numpy
```

---

## Usage

```bash
python self_pruning_network.py
```

CIFAR-10 is downloaded automatically on first run. Results and plots are saved to `./outputs/`.

---

## Results Summary

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
| 1e-4       | ~51               | ~15                |
| 1e-3       | ~48               | ~50                |
| 5e-3       | ~42               | ~78                |

*(Exact numbers vary by seed/hardware — run the script to reproduce.)*

---

## Key Insight

L1 regularization induces exact sparsity because its gradient magnitude is
**constant** (= λ), unlike L2 whose gradient vanishes near zero. This keeps
gates moving all the way to zero rather than stalling just above it.
