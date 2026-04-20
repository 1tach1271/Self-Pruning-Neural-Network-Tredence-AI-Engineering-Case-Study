# Self-Pruning Neural Network — Results Report

## 1. Why does an L1 penalty on sigmoid gates encourage sparsity?

The total training loss is:

```
Total Loss = CrossEntropyLoss(logits, labels) + λ × Σ sigmoid(gate_scores_ij)
```

**Sigmoid** maps every gate score to the open interval `(0, 1)`, so all gate values
are strictly non-negative. The **L1 norm** of non-negative values is simply their
sum. Minimising this sum puts constant gradient pressure on the optimizer to push as
many gates as possible toward **exactly 0**.

### Why L1 and not L2?

| Penalty | Gradient magnitude | Effect near zero |
|:-------:|:------------------:|:-----------------|
| L2 (sum of squares) | `2 × gate` — shrinks to zero as gate → 0 | Gradient vanishes; gate stalls above zero |
| **L1 (sum of values)** | **constant `λ`** regardless of gate magnitude | **Keeps pushing all the way to zero** |

This constant-gradient property is why L1 regularisation (also seen in Lasso
regression) produces **exact sparsity**, while L2 merely produces small-but-nonzero
values.

In practice, a gate score that diverges to −∞ gives `sigmoid → 0`, which:
1. Zeroes out that weight's contribution in the forward pass.
2. Zeroes out gradients flowing back to the corresponding `weight` parameter.

The network therefore learns a soft binary mask **end-to-end without any
post-training step**.

---

## 2. Results Table

*(Fill in your actual numbers after running the script.)*

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
| 1e-4       | ~50–52            | ~10–20             |
| 1e-3       | ~47–50            | ~40–60             |
| 5e-3       | ~40–45            | ~70–85             |

> **Note:** Exact numbers depend on hardware, random seed, and number of epochs.
> Run `self_pruning_network.py` to reproduce results; the script prints and saves the
> exact table automatically.

---

## 3. Gate Value Distribution (`gate_distributions.png`)

A **successful** pruning run produces a bimodal histogram:

- A **tall spike near 0** — weights whose gates have been driven to zero (pruned).
- A **secondary cluster toward 1** — weights that survived as important connections.

As λ increases:
- The spike at 0 becomes taller (more pruning).
- The active cluster shrinks (fewer surviving weights).
- Classification accuracy decreases (over-pruning hurts capacity).

---

## 4. Training Curves (`training_curves.png`)

| λ value | Accuracy trend | Sparsity trend |
|:-------:|:--------------|:--------------|
| **Low (1e-4)** | Converges high (~50 %+) | Grows slowly; modest pruning |
| **Medium (1e-3)** | Slight accuracy drop | Meaningful sparsity without collapse |
| **High (5e-3)** | Notable accuracy drop | Aggressive pruning; many gates zeroed early |

---

## 5. Analysis: The Sparsity–Accuracy Trade-off

Increasing λ shifts the loss surface so that *compactness* is rewarded more than
*correctness*. The sweet spot (medium λ) achieves a sparser network with only a
minor accuracy penalty — this is the practical goal of learned pruning.

For production use cases, one would:
1. Train with a medium λ to identify unimportant weights.
2. Hard-prune (remove) those weights and fine-tune without the sparsity term.
3. Obtain a smaller, faster model with minimal accuracy loss.

---

## 6. How to Reproduce

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Run (downloads CIFAR-10 automatically on first run)
python self_pruning_network.py

# Outputs saved to ./outputs/
#   gate_distributions.png
#   training_curves.png
#   report.md
```
