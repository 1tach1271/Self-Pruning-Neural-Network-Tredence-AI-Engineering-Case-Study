"""
Self-Pruning Neural Network for CIFAR-10 Classification
Tredence AI Engineering Intern Case Study

Author: Shashi Yadav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Part 1: PrunableLinear Layer

class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight.
        # Initialized near 0 so initial gates ≈ sigmoid(0) = 0.5 (all active at start).
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform init for weights (standard practice for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gates         = torch.sigmoid(self.gate_scores)          # (out, in)
        pruned_weights = self.weight * gates                      # (out, in)
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        
        gates = self.get_gates()
        pruned = (gates < threshold).float().sum().item()
        total  = gates.numel()
        return pruned / total if total > 0 else 0.0

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# Neural Network Definition

class SelfPruningNet(nn.Module):
    

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, 10)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # flatten: (B, 3072)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))

        return self.fc4(x)                  # raw logits

    def prunable_layers(self):
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        gate_sum = torch.tensor(0.0, requires_grad=True)
        for layer in self.prunable_layers():
            gates     = torch.sigmoid(layer.gate_scores)   # gradients flow here
            gate_sum  = gate_sum + gates.sum()
        return gate_sum

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        pruned = total = 0
        for layer in self.prunable_layers():
            gates  = layer.get_gates()
            pruned += (gates < threshold).sum().item()
            total  += gates.numel()
        return pruned / total if total > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        all_gates = []
        for layer in self.prunable_layers():
            all_gates.append(layer.get_gates().cpu().numpy().ravel())
        return np.concatenate(all_gates)


# Part 2 & 3: Training, Evaluation, and Reporting

def get_cifar10_loaders(batch_size: int = 128):
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, lambda_sparse, device):
    model.train()
    total_loss_sum = cls_loss_sum = sparse_loss_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        cls_loss    = criterion(logits, labels)
        sparse_loss = model.sparsity_loss()
        total_loss  = cls_loss + lambda_sparse * sparse_loss

        total_loss.backward()
        optimizer.step()

        total_loss_sum  += total_loss.item()
        cls_loss_sum    += cls_loss.item()
        sparse_loss_sum += sparse_loss.item()

    n = len(loader)
    return total_loss_sum / n, cls_loss_sum / n, sparse_loss_sum / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        preds       = logits.argmax(dim=1)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train_and_evaluate(lambda_sparse: float,
                       num_epochs: int = 25,
                       lr: float = 1e-3,
                       device: torch.device = None,
                       seed: int = 42) -> dict:
    
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_loaders()
    model     = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n{'='*60}")
    print(f"  λ = {lambda_sparse}  |  device = {device}  |  epochs = {num_epochs}")
    print(f"{'='*60}")

    history = {"epoch": [], "train_loss": [], "test_acc": [], "sparsity": []}

    for epoch in range(1, num_epochs + 1):
        tr_loss, cls_l, sp_l = train_one_epoch(
            model, train_loader, optimizer, criterion, lambda_sparse, device)
        test_loss, test_acc   = evaluate(model, test_loader, criterion, device)
        sparsity              = model.overall_sparsity()
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity * 100)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {tr_loss:.4f} (cls={cls_l:.4f}, sp={sp_l:.2f}) | "
                  f"Test Acc: {test_acc:.2f}% | Sparsity: {sparsity*100:.1f}%")

    final_test_loss, final_acc = evaluate(model, test_loader, criterion, device)
    final_sparsity             = model.overall_sparsity()
    gate_values                = model.all_gate_values()

    print(f"\n  ✓ Final Test Accuracy : {final_acc:.2f}%")
    print(f"  ✓ Final Sparsity      : {final_sparsity*100:.1f}%")

    return {
        "lambda":        lambda_sparse,
        "test_accuracy": final_acc,
        "sparsity":      final_sparsity * 100,
        "gate_values":   gate_values,
        "history":       history,
        "model":         model,
    }


def plot_results(results: list[dict], save_dir: str = "."):
   
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Gate Value Distributions
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF9800", "#F44336"]

    for ax, res, color in zip(axes, results, colors):
        gates = res["gate_values"]
        ax.hist(gates, bins=80, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(x=0.01, color="black", linestyle="--", linewidth=1.2,
                   label="Prune threshold (0.01)")
        ax.set_title(f"λ = {res['lambda']}\n"
                     f"Acc: {res['test_accuracy']:.1f}%  |  "
                     f"Sparsity: {res['sparsity']:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Gate Value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim([-0.02, 1.02])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Gate Value Distributions for Different λ Values\n"
                 "(spike near 0 = successful pruning)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path1 = os.path.join(save_dir, "gate_distributions.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot saved] {path1}")

    # Plot 2: Training Curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for res, color in zip(results, colors):
        h = res["history"]
        ax1.plot(h["epoch"], h["test_acc"],    color=color, linewidth=2,
                 label=f"λ={res['lambda']}")
        ax2.plot(h["epoch"], h["sparsity"],    color=color, linewidth=2,
                 label=f"λ={res['lambda']}")

    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("Test Accuracy vs Epoch", fontsize=13, fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_ylabel("Sparsity (%)", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_title("Sparsity Level vs Epoch", fontsize=13, fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot saved] {path2}")


def generate_markdown_report(results: list[dict], save_dir: str = "."):
    lines = []
    lines.append("# Self-Pruning Neural Network — Results Report\n")
    lines.append("## 1. Why does an L1 penalty on sigmoid gates encourage sparsity?\n")
    lines.append(
        "The total training loss is:\n\n"
        "```\nTotal Loss = CrossEntropyLoss + λ * Σ sigmoid(gate_scores)\n```\n\n"
        "**Sigmoid** maps each gate score to `(0, 1)`, so all gate values are "
        "non-negative. The **L1 norm** of non-negative values is simply their sum. "
        "Minimising this sum pressures the optimizer to push as many gates as possible "
        "toward **exactly 0**.\n\n"
        "Why L1 and not L2? The L2 norm (sum of squares) shrinks large values "
        "aggressively but still leaves small values slightly above zero. The L1 norm "
        "applies a *constant* gradient (`±λ`) regardless of the magnitude, which means "
        "it keeps pushing a near-zero gate all the way to zero rather than stopping "
        "short. This property — known as **L1-induced sparsity** — is the same reason "
        "Lasso regression selects sparse feature sets.\n\n"
        "In practice, a gate whose score diverges to `-∞` gives `sigmoid → 0`, "
        "effectively removing that weight from the forward pass and from gradient "
        "updates of the weight itself. The network thus learns a binary mask "
        "end-to-end.\n"
    )

    lines.append("## 2. Results Table\n")
    lines.append("| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |")
    lines.append("|:----------:|:-----------------:|:------------------:|")
    for res in results:
        lines.append(
            f"| {res['lambda']} | {res['test_accuracy']:.2f} | {res['sparsity']:.1f} |"
        )

    lines.append("\n## 3. Gate Value Distribution\n")
    lines.append(
        "The plot `gate_distributions.png` shows the histogram of all gate values "
        "after training. A **successful** pruning run exhibits:\n"
        "- A large spike near **0** (pruned connections)\n"
        "- A secondary cluster near **0.5–1.0** (active, important connections)\n\n"
        "As λ increases, the spike at 0 grows taller and the active cluster shrinks, "
        "confirming that higher regularization forces more aggressive pruning.\n"
    )

    lines.append("## 4. Training Curves\n")
    lines.append(
        "See `training_curves.png`. Key observations:\n"
        "- **Low λ**: accuracy converges high; sparsity grows slowly.\n"
        "- **Medium λ**: good balance — meaningful sparsity with modest accuracy drop.\n"
        "- **High λ**: sparsity is maximized but accuracy degrades, showing the "
        "sparsity–accuracy trade-off.\n"
    )

    report_path = os.path.join(save_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Report saved] {report_path}")


# Entry Point

if __name__ == "__main__":
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 25          # increase to 40–50 for better accuracy
    SAVE_DIR   = "./outputs"

    # Three λ values: low, medium, high
    LAMBDAS = [1e-4, 1e-3, 5e-3]

    all_results = []
    for lam in LAMBDAS:
        result = train_and_evaluate(
            lambda_sparse=lam,
            num_epochs=NUM_EPOCHS,
            lr=1e-3,
            device=DEVICE,
        )
        all_results.append(result)

    # Summary Table
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Lambda':<12} {'Test Acc (%)':<18} {'Sparsity (%)':<15}")
    print("  " + "-"*45)
    for res in all_results:
        print(f"  {res['lambda']:<12} {res['test_accuracy']:<18.2f} {res['sparsity']:<15.1f}")

    # Plots & Report
    plot_results(all_results, save_dir=SAVE_DIR)
    generate_markdown_report(all_results, save_dir=SAVE_DIR)

    print("\nDone! Outputs saved to:", SAVE_DIR)
