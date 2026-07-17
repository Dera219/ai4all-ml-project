"""My CNN — designed from the problem, not copied from the group's notebook.

## Three deliberate departures, each with a reason

**1. Global average pooling instead of a giant flatten→FC.**

The group's model flattens `128 x 14 x 14` into a `Linear(25088, 128)`. That single layer is
**3.2 million parameters** — trained on ~2,300 images. It is by far the largest part of the model
and it exists only to collapse spatial dimensions. Global average pooling does the same job with
**zero** parameters, taking the trunk from ~3.3M params to ~100K. On a dataset this small, that
gap is most of the overfitting.

**2. BatchNorm after every conv.**

Their model has none. Without it, a 4-layer stack trains slowly and is sensitive to
initialization and learning rate. This is cheap and standard.

**3. Ordinal head, kept.**

This one the group got right and it's worth keeping: Low < Medium < High are *ordered*, so a
3-way softmax is the wrong loss — it treats Low-vs-High as no worse than Low-vs-Medium. Two
sigmoid thresholds ("above Low?", "above Medium?") put the ordering into the objective.

Where I differ is the decode. Summing passed thresholds silently accepts `[0, 1]` — "not above
the first threshold but above the second" — which is incoherent. I decode by walking the
thresholds in order and stopping at the first failure, which cannot produce an incoherent class,
and I report how often the raw predictions disagreed with themselves.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class CalorieCNN(nn.Module):
    """Compact ordinal CNN with an auxiliary calorie-regression head.

    `in_channels=4` feeds RGB + depth. Depth is the instructor's suggested direction and it's
    physically motivated: calories track mass, mass tracks volume, and volume is exactly what a
    depth map sees and a photograph cannot.
    """

    def __init__(
        self, *, in_channels: int = 3, width: int = 32, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.features = nn.Sequential(
            ConvBlock(in_channels, width),
            ConvBlock(width, width),
            nn.MaxPool2d(2),  # 224 -> 112
            ConvBlock(width, width * 2),
            nn.MaxPool2d(2),  # 112 -> 56
            ConvBlock(width * 2, width * 4),
            nn.MaxPool2d(2),  # 56 -> 28
            ConvBlock(width * 4, width * 8),
            nn.MaxPool2d(2),  # 28 -> 14
            ConvBlock(width * 8, width * 8),
        )
        # Zero parameters, and it makes the model input-size agnostic.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        trunk = width * 8
        self.ordinal_head = nn.Linear(trunk, 2)
        self.regression_head = nn.Linear(trunk, 1)  # log1p(calories)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        h = self.dropout(h)
        return self.ordinal_head(h), self.regression_head(h).squeeze(-1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GroupStyleCNN(nn.Module):
    """A faithful reproduction of the group's architecture, for the leakage comparison.

    Their `ExtraLayerOrdinalMultiTaskCNN224`: four conv blocks (no BatchNorm) into a **giant
    flatten -> Linear**, then ordinal + regression heads. The defining feature is that flatten:
    at 224px it is `Linear(128*14*14=25088, 128)` = 3.2M parameters, which is what a
    high-capacity model needs to memorize training-specific detail — exactly the mechanism that
    would let it exploit session leakage.

    The only change from their code is `nn.LazyLinear` instead of a hardcoded `Linear(25088, ...)`,
    so the same architecture runs at any input size (the flatten dimension is inferred on first
    forward). At 224px it is bit-for-bit the same parameter count as theirs; at 160px it adapts.
    This exists ONLY as the high-capacity arm of the experiment — it is not the model I ship.
    """

    def __init__(self, *, in_channels: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.shared_fc = nn.LazyLinear(128)  # the giant flatten->FC, size-adaptive
        self.dropout = nn.Dropout(dropout)
        self.ordinal_head = nn.Linear(128, 2)
        self.regression_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.flatten(1)
        features = self.dropout(F.relu(self.shared_fc(x)))
        return self.ordinal_head(features), self.regression_head(features).squeeze(-1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def decode_ordinal(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinal logits -> (class index, was_coherent).

    Walks thresholds in order and stops at the first one not passed, so the output is always a
    valid ordered class. `[0, 1]` decodes to 0 (the first threshold failed) rather than being
    summed into 1 as if nothing were wrong.

    Also returns whether each prediction's thresholds were self-consistent, so incoherence can be
    reported instead of hidden.
    """
    passed = (torch.sigmoid(logits) >= 0.5).int()  # (N, 2)
    first = passed[:, 0]
    second = passed[:, 1]

    # Stop at the first failure: class = 0 if !t1, else 1 if !t2, else 2.
    predicted = torch.where(first == 0, 0, torch.where(second == 0, 1, 2))
    coherent = ~((first == 0) & (second == 1))
    return predicted, coherent


def ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    """Class index -> cumulative threshold targets. 0 -> [0,0], 1 -> [1,0], 2 -> [1,1]."""
    return torch.stack([(labels >= 1).float(), (labels >= 2).float()], dim=1)


class MajorityBaseline:
    """Always predicts the most common training class.

    Not a formality. It is the only thing that says whether the CNN earned its complexity — on
    balanced tertiles it scores ~33%, so any model that can't clear that is worse than a constant.
    """

    def __init__(self) -> None:
        self.majority: int = 0

    def fit(self, labels: torch.Tensor) -> MajorityBaseline:
        self.majority = int(torch.bincount(labels, minlength=3).argmax().item())
        return self

    def predict(self, n: int) -> torch.Tensor:
        return torch.full((n,), self.majority, dtype=torch.long)
