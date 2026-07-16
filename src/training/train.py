"""Training loop.

## The test set is locked

`evaluate_on_test` refuses to run unless explicitly unlocked. This is a guard rail against the
exact failure in the group's notebooks, where test accuracy appears six times — 66.9, 70.0, 71.3,
71.5, 72.3, 74.1 — climbing monotonically. That pattern is what selecting-on-test looks like:
try a change, check test, keep it if test improved. After that, test measures how well you
searched, not how well the model generalizes.

Model selection here uses validation only. Test is touched once, at the end, by a script that has
to say so out loud.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.utils.data import DataLoader

from src.models.cnn import decode_ordinal, ordinal_targets


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_mae_calories: float
    incoherent_fraction: float
    seconds: float


@dataclass
class History:
    epochs: list[EpochResult] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_epoch: int = -1

    def record(self, result: EpochResult) -> bool:
        """Append, and report whether this is a new best (by validation accuracy)."""
        self.epochs.append(result)
        if result.val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = result.val_accuracy
            self.best_epoch = result.epoch
            return True
        return False


def compute_loss(
    ordinal_logits: torch.Tensor,
    regression: torch.Tensor,
    labels: torch.Tensor,
    log_calories: torch.Tensor,
    *,
    regression_weight: float,
) -> torch.Tensor:
    """Ordinal BCE + auxiliary regression on log1p(calories).

    Regression is on the log scale because calories span 1–3,900 kcal with a long right tail.
    On the raw scale, a single 3,900 kcal dish contributes as much squared error as hundreds of
    ordinary ones, so the model spends its capacity on outliers. Log compresses that.

    Smooth L1 rather than MSE for the same reason: it stops one bad label from dominating a batch.
    """
    ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets(labels))
    regression_loss = F.smooth_l1_loss(regression, log_calories)
    return ordinal_loss + regression_weight * regression_loss


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, *, regression_weight: float
) -> tuple[float, float, float, float]:
    """Returns (loss, accuracy, mae_calories, incoherent_fraction)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    abs_error = 0.0
    incoherent = 0

    for images, labels, log_calories in loader:
        images = images.to(device)
        labels = labels.to(device)
        log_calories = log_calories.to(device)

        ordinal_logits, regression = model(images)
        loss = compute_loss(
            ordinal_logits, regression, labels, log_calories, regression_weight=regression_weight
        )

        predicted, coherent = decode_ordinal(ordinal_logits)
        correct += int((predicted == labels).sum().item())
        incoherent += int((~coherent).sum().item())
        total_loss += float(loss.item()) * labels.size(0)

        # Report MAE in kcal, not log-space: "off by 0.31 log-kcal" means nothing to a reader.
        abs_error += float((regression.expm1() - log_calories.expm1()).abs().sum().item())
        n += labels.size(0)

    return total_loss / n, correct / n, abs_error / n, incoherent / n


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    regression_weight: float = 0.3,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[History, dict[str, torch.Tensor]]:
    """Train, selecting the best epoch on VALIDATION accuracy.

    Returns the history and the best checkpoint's weights. The returned weights are the best
    validation epoch — not the last — so a model that starts overfitting doesn't get shipped
    just because training ended there.
    """
    device = device or pick_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = History()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        started = time.time()
        model.train()
        running = 0.0
        seen = 0

        for images, labels, log_calories in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            log_calories = log_calories.to(device)

            optimizer.zero_grad(set_to_none=True)
            ordinal_logits, regression = model(images)
            loss = compute_loss(
                ordinal_logits, regression, labels, log_calories, regression_weight=regression_weight
            )
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * labels.size(0)
            seen += labels.size(0)

        scheduler.step()
        val_loss, val_accuracy, val_mae, incoherent = evaluate(
            model, val_loader, device, regression_weight=regression_weight
        )
        result = EpochResult(
            epoch=epoch,
            train_loss=running / seen,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_mae_calories=val_mae,
            incoherent_fraction=incoherent,
            seconds=time.time() - started,
        )
        if history.record(result):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            star = " *" if history.best_epoch == epoch else ""
            print(
                f"  epoch {epoch:2d}  train {result.train_loss:.4f}  "
                f"val {val_loss:.4f}  acc {val_accuracy:.4f}  "
                f"mae {val_mae:6.1f}kcal  {result.seconds:.1f}s{star}"
            )

    return history, best_state
