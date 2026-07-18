"""Evaluation metrics for the 3-class calorie classifier.

Implements the metrics Ru's checklist asks for — confusion matrix, per-class precision / recall /
specificity / F1, and ROC/AUC — computed directly rather than pulled from a library black box, so
the presentation can explain each number.

## The one thing to get right for ordered classes

Accuracy and a plain confusion matrix treat all errors alike. For Low < Medium < High that's
wrong: confusing Medium with High is a near miss; confusing Low with High means the model learned
nothing about that image. Every function here keeps the ordering visible — the confusion matrix
is laid out Low→High so off-diagonal *distance* is readable, and `off_by_two_rate` reports how
often the model made the maximal error, which is the number that says whether the ordinal head is
doing its job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

CLASS_NAMES = ("Low", "Medium", "High")


def ordinal_class_probabilities(ordinal_logits: torch.Tensor) -> np.ndarray:
    """Turn the two cumulative threshold logits into a proper 3-class probability distribution.

    The head predicts cumulative probabilities: t1 = P(class ≥ Medium), t2 = P(class ≥ High).
    The per-class probabilities follow from the cumulative ones:

        P(Low)    = 1 - t1
        P(Medium) = t1 - t2
        P(High)   = t2

    `t2 > t1` (incoherent — "more likely High than not-Low") would make P(Medium) negative; we
    clip at zero and renormalize so the result is always a valid distribution. This is what ROC
    needs (a score per class), and clipping rather than ignoring keeps the incoherent cases in the
    denominator instead of silently dropping them.
    """
    probs = torch.sigmoid(ordinal_logits).numpy()
    t1, t2 = probs[:, 0], probs[:, 1]
    p_low = 1.0 - t1
    p_med = np.clip(t1 - t2, 0.0, None)
    p_high = t2
    stacked = np.stack([p_low, p_med, p_high], axis=1)
    return stacked / stacked.sum(axis=1, keepdims=True)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Rows = true class, columns = predicted class, ordered Low→High."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred, strict=True):
        matrix[int(true), int(pred)] += 1
    return matrix


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    name: str
    precision: float
    recall: float  # sensitivity
    specificity: float
    f1: float
    support: int


def per_class_metrics(cm: np.ndarray) -> list[ClassMetrics]:
    """Derive precision, recall, specificity, and F1 for each class from the confusion matrix.

    Reported per class, not just averaged, because the classes are balanced by construction but
    the *kinds* of mistake are not symmetric — and a single averaged F1 hides which class the
    model is worst at.
    """
    total = cm.sum()
    out: list[ClassMetrics] = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        out.append(
            ClassMetrics(
                name=name,
                precision=float(precision),
                recall=float(recall),
                specificity=float(specificity),
                f1=float(f1),
                support=int(cm[i, :].sum()),
            )
        )
    return out


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0


def macro_f1(metrics: list[ClassMetrics]) -> float:
    return float(np.mean([m.f1 for m in metrics])) if metrics else 0.0


def off_by_two_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that confused Low with High (distance 2) — the maximal error.

    This is the ordinal-specific number. A model whose mistakes are mostly adjacent (Low↔Medium,
    Medium↔High) has learned the ordering; one that regularly jumps Low↔High has not. Off-by-two
    should be rare if the ordinal head is working, and it's a strong slide either way.
    """
    distance = np.abs(y_true.astype(int) - y_pred.astype(int))
    return float((distance == 2).mean()) if len(distance) else 0.0


def roc_one_vs_rest(y_true: np.ndarray, probs: np.ndarray) -> dict[str, dict[str, np.ndarray | float]]:
    """One-vs-rest ROC curve and AUC per class.

    Three classes have no single ROC, so each class is scored against the other two using its
    predicted probability. Returns fpr/tpr arrays and the AUC for each class.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    out: dict[str, dict[str, np.ndarray | float]] = {}
    for i, name in enumerate(CLASS_NAMES):
        binary_true = (y_true == i).astype(int)
        scores = probs[:, i]
        # A class absent from the split has an undefined ROC — skip rather than emit a misleading
        # 0.5. (Shouldn't happen on the balanced tertile split, but the test set is small.)
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            continue
        fpr, tpr, _ = roc_curve(binary_true, scores)
        out[name] = {"fpr": fpr, "tpr": tpr, "auc": float(roc_auc_score(binary_true, scores))}
    return out


def metrics_table_markdown(cm: np.ndarray, metrics: list[ClassMetrics], roc: dict) -> str:
    """Render the per-class metrics as a Markdown table for the README / presentation."""
    lines = [
        "| Class | Precision | Recall (sens.) | Specificity | F1 | AUC | Support |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        auc = roc.get(m.name, {}).get("auc")
        auc_str = f"{auc:.3f}" if auc is not None else "—"
        lines.append(
            f"| {m.name} | {m.precision:.3f} | {m.recall:.3f} | {m.specificity:.3f} | "
            f"{m.f1:.3f} | {auc_str} | {m.support} |"
        )
    lines.append(f"\n**Accuracy:** {accuracy(cm):.3f} · **Macro-F1:** {macro_f1(metrics):.3f}")
    return "\n".join(lines)
