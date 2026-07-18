"""Produce the Week 12 evaluation artifacts for my model.

Trains `CalorieCNN` on a clean, session-grouped split, then evaluates on a held-out clean test set
and writes everything Ru's checklist asks for:

- confusion matrix (row-normalized, Low→High) → reports/figures/confusion_matrix.png
- ROC / AUC, one-vs-rest per class          → reports/figures/roc_curves.png
- per-class precision/recall/specificity/F1 → reports/evaluation.md + reports/evaluation.json
- the ordinal-specific off-by-two rate

This runs on MY pipeline — it does not need the group's Kaggle weights. Everything is honest about
that: the numbers here are from a small (990K-param) model on a clean split, and the report says so.

Run:  python experiments/evaluate.py --data-root ~/Downloads/archive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cache import build_cache  # noqa: E402
from src.data.nutrition5k import (  # noqa: E402
    add_calorie_classes,
    holdout_clean_test,
    load_manifest,
    sessions_for,
    train_val_split,
)
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curves  # noqa: E402
from src.models.cnn import CalorieCNN, MajorityBaseline, decode_ordinal  # noqa: E402
from src.training.dataset import DishDataset  # noqa: E402
from src.training.train import pick_device, train  # noqa: E402


def loader(frame, rgb, index_of, *, batch_size: int, shuffle: bool, augment: bool) -> DataLoader:
    ds = DishDataset(frame, rgb, None, index_of, augment=augment, use_depth=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def collect_predictions(model, data_loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, class_probabilities) over the whole loader."""
    model.eval()
    trues, preds, probs = [], [], []
    for images, labels, _ in data_loader:
        ordinal_logits, _ = model(images.to(device))
        ordinal_logits = ordinal_logits.cpu()
        predicted, _ = decode_ordinal(ordinal_logits)
        trues.append(labels.numpy())
        preds.append(predicted.numpy())
        probs.append(M.ordinal_class_probabilities(ordinal_logits))
    return np.concatenate(trues), np.concatenate(preds), np.concatenate(probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Downloads/archive")
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"device: {device} | size: {args.size} | seed: {args.seed}")

    manifest = add_calorie_classes(sessions_for(load_manifest(args.data_root)))[0]
    print(f"dishes: {len(manifest)} | sessions: {manifest.session.nunique()}")

    paths = build_cache(manifest, args.cache, size=args.size, with_depth=True)
    rgb = np.load(paths["rgb"], mmap_mode="r")
    index_of = {d: i for i, d in enumerate(np.load(paths["ids"], allow_pickle=True))}

    # Clean, session-grouped split — the honest evaluation. Held-out test shares no capture
    # session with the training pool.
    pool, clean_test = holdout_clean_test(manifest, test_fraction=0.2, seed=args.seed)
    train_frame, val_frame = train_val_split(pool, strategy="clean", seed=args.seed)

    train_frame, edges = add_calorie_classes(train_frame)
    val_frame, _ = add_calorie_classes(val_frame, edges=edges)
    test_frame, _ = add_calorie_classes(clean_test, edges=edges)
    print(f"train {len(train_frame)} | val {len(val_frame)} | clean test {len(test_frame)}")

    train_loader = loader(train_frame, rgb, index_of, batch_size=args.batch_size, shuffle=True, augment=True)
    val_loader = loader(val_frame, rgb, index_of, batch_size=args.batch_size, shuffle=False, augment=False)
    test_loader = loader(test_frame, rgb, index_of, batch_size=args.batch_size, shuffle=False, augment=False)

    model = CalorieCNN(in_channels=3)
    print(f"model params: {model.n_parameters:,}\ntraining...")
    _, best_state = train(model, train_loader, val_loader, epochs=args.epochs, device=device, verbose=True)
    model.load_state_dict(best_state)
    model.to(device)

    # --- evaluate on the held-out clean test, once ---
    y_true, y_pred, probs = collect_predictions(model, test_loader, device)

    cm = M.confusion_matrix(y_true, y_pred)
    class_metrics = M.per_class_metrics(cm)
    roc = M.roc_one_vs_rest(y_true, probs)

    baseline = MajorityBaseline().fit(torch.tensor(train_frame.label.to_numpy()))
    baseline_acc = float((baseline.predict(len(y_true)).numpy() == y_true).mean())

    # --- write artifacts ---
    args.figures.mkdir(parents=True, exist_ok=True)
    cm_png = plot_confusion_matrix(cm, args.figures / "confusion_matrix.png", title="Calorie class — confusion matrix (clean test)")
    roc_png = plot_roc_curves(roc, args.figures / "roc_curves.png", title="Calorie class — ROC (one vs rest)")

    table = M.metrics_table_markdown(cm, class_metrics, roc)
    report_md = (
        "# Evaluation — my CalorieCNN on a clean session-grouped split\n\n"
        f"Test set: **{len(y_true)} dishes**, session-disjoint from training. "
        f"Majority-class baseline: **{baseline_acc:.1%}**.\n\n"
        f"{table}\n\n"
        f"**Off-by-two rate** (Low↔High, the maximal error): "
        f"**{M.off_by_two_rate(y_true, y_pred):.1%}** — low means the ordinal head keeps mistakes "
        f"adjacent.\n\n"
        f"![confusion matrix](figures/confusion_matrix.png)\n\n"
        f"![ROC curves](figures/roc_curves.png)\n\n"
        "*These numbers are from the ~990K-param CalorieCNN at "
        f"{args.size}px on the clean split — reproducible via this script, independent of the "
        "group's Kaggle weights. They are not the group's reported 74.1%.*\n"
    )
    Path("reports/evaluation.md").write_text(report_md)

    payload = {
        "n_test": int(len(y_true)),
        "accuracy": M.accuracy(cm),
        "macro_f1": M.macro_f1(class_metrics),
        "baseline_accuracy": baseline_acc,
        "off_by_two_rate": M.off_by_two_rate(y_true, y_pred),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {"name": m.name, "precision": m.precision, "recall": m.recall,
             "specificity": m.specificity, "f1": m.f1, "support": m.support,
             "auc": roc.get(m.name, {}).get("auc")}
            for m in class_metrics
        ],
    }
    Path("reports/evaluation.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{'=' * 60}\nEVALUATION COMPLETE\n{'=' * 60}")
    print(table)
    print(f"\nfigures: {cm_png}, {roc_png}")
    print("report:  reports/evaluation.md, reports/evaluation.json")


if __name__ == "__main__":
    main()
