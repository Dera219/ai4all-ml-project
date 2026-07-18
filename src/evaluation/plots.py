"""Presentation figures: confusion matrix and ROC curves.

Matplotlib only, no seaborn — one fewer dependency, and the styling here is deliberate rather
than a theme default. Figures are saved at 150 dpi, which is sharp on a projector without being
huge.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write straight to file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation.metrics import CLASS_NAMES  # noqa: E402


def plot_confusion_matrix(cm: np.ndarray, out_path: Path | str, *, title: str = "Confusion matrix") -> Path:
    """Row-normalized confusion matrix, ordered Low→High.

    Normalized by row (true class) so each cell reads as "of the actual Lows, what fraction did we
    call Medium?" — the recall view, which is what you want when the question is where a given
    true class leaks to. Raw counts are annotated underneath so the support is visible too.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            frac = normalized[i, j]
            # Contrast: white text on dark cells, dark on light.
            color = "white" if frac > 0.5 else "#16150f"
            ax.text(
                j, i, f"{frac:.0%}\n({cm[i, j]})",
                ha="center", va="center", color=color, fontsize=10,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curves(roc: dict, out_path: Path | str, *, title: str = "ROC — one vs rest") -> Path:
    """Overlay the three one-vs-rest ROC curves with their AUCs in the legend."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#a3a09a", label="chance (AUC 0.50)")

    for name in CLASS_NAMES:
        if name not in roc:
            continue
        curve = roc[name]
        ax.plot(curve["fpr"], curve["tpr"], linewidth=2, label=f"{name} (AUC {curve['auc']:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
