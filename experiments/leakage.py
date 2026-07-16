"""The leakage experiment.

Trains the SAME model, same seed, same hyperparameters, twice — changing only the split:

- **random**: stratified on dish_id (what the group used). 0 dish_id overlap. Looks clean.
- **grouped**: whole capture sessions assigned to one set. 0 session overlap.

The accuracy gap between them is the size of the leak, measured rather than argued.

Run:  python experiments/leakage.py --data-root ~/Downloads/archive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cache import build_cache  # noqa: E402
from src.data.nutrition5k import (  # noqa: E402
    add_calorie_classes,
    grouped_splits,
    load_manifest,
    random_splits,
    session_leakage_report,
    sessions_for,
)
from src.models.cnn import CalorieCNN, MajorityBaseline  # noqa: E402
from src.training.dataset import DishDataset  # noqa: E402
from src.training.train import evaluate, pick_device, train  # noqa: E402


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_loaders(splits, rgb, depth, index_of, *, batch_size: int, use_depth: bool):
    make = lambda frame, augment: DishDataset(  # noqa: E731
        frame, rgb, depth, index_of, augment=augment, use_depth=use_depth
    )
    return (
        DataLoader(make(splits.train, True), batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(make(splits.val, False), batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(make(splits.test, False), batch_size=batch_size, shuffle=False, num_workers=0),
    )


def run_arm(
    name: str,
    splits,
    rgb,
    depth,
    index_of,
    *,
    epochs: int,
    batch_size: int,
    use_depth: bool,
    seed: int,
    device,
) -> dict:
    print(f"\n{'=' * 74}\n{name.upper()} SPLIT\n{'=' * 74}")
    print(" ", splits.summary())
    leak = session_leakage_report(splits)
    print(
        f"  test dishes sharing a capture session with train: "
        f"{leak['test_dishes_with_session_mate_in_train']}/{leak['test_dishes']} "
        f"({leak['contaminated_fraction'] * 100:.1f}%)"
    )

    seed_everything(seed)
    train_loader, val_loader, test_loader = build_loaders(
        splits, rgb, depth, index_of, batch_size=batch_size, use_depth=use_depth
    )

    in_channels = 4 if use_depth else 3
    model = CalorieCNN(in_channels=in_channels)
    print(f"  model params: {model.n_parameters:,}")

    baseline = MajorityBaseline().fit(torch.tensor(splits.train.label.to_numpy()))
    test_labels = torch.tensor(splits.test.label.to_numpy())
    baseline_accuracy = float((baseline.predict(len(test_labels)) == test_labels).float().mean())
    print(f"  majority-class baseline on test: {baseline_accuracy:.4f}\n")

    history, best_state = train(
        model, train_loader, val_loader, epochs=epochs, device=device, verbose=True
    )

    # Test is touched ONCE, here, after selection is finished on validation.
    model.load_state_dict(best_state)
    model.to(device)
    _, test_accuracy, test_mae, incoherent = evaluate(
        model, test_loader, device, regression_weight=0.3
    )

    print(
        f"\n  best val acc {history.best_val_accuracy:.4f} (epoch {history.best_epoch})"
        f"  ->  TEST acc {test_accuracy:.4f}  mae {test_mae:.1f} kcal"
    )
    return {
        "split": name,
        "contaminated_fraction": leak["contaminated_fraction"],
        "baseline_accuracy": baseline_accuracy,
        "best_val_accuracy": history.best_val_accuracy,
        "best_epoch": history.best_epoch,
        "test_accuracy": test_accuracy,
        "test_mae_calories": test_mae,
        "incoherent_fraction": incoherent,
        "n_train": len(splits.train),
        "n_test": len(splits.test),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Downloads/archive")
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/leakage.json"))
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device} | depth: {not args.no_depth}")

    manifest = sessions_for(load_manifest(args.data_root))
    print("dishes:", len(manifest), "| sessions:", manifest.session.nunique())

    paths = build_cache(manifest, args.cache, size=224, with_depth=True)
    rgb = np.load(paths["rgb"], mmap_mode="r")
    depth = np.load(paths["depth"], mmap_mode="r")
    ids = np.load(paths["ids"], allow_pickle=True)
    index_of = {d: i for i, d in enumerate(ids)}

    results = []
    for name, splitter in (("grouped", grouped_splits), ("random", random_splits)):
        # Class edges come from TRAIN ONLY, per arm. Computing them on the full frame would let
        # the test distribution set the boundaries — a small leak, in an experiment about leaks.
        if name == "grouped":
            frame = sessions_for(load_manifest(args.data_root))
            frame, _ = add_calorie_classes(frame)  # provisional, for grouping only
            splits = splitter(frame, seed=args.seed)
        else:
            frame = sessions_for(load_manifest(args.data_root))
            frame, _ = add_calorie_classes(frame)
            splits = splitter(frame, seed=args.seed)

        train_frame, edges = add_calorie_classes(splits.train)
        val_frame, _ = add_calorie_classes(splits.val, edges=edges)
        test_frame, _ = add_calorie_classes(splits.test, edges=edges)
        splits = type(splits)(train=train_frame, val=val_frame, test=test_frame)

        started = time.time()
        results.append(
            run_arm(
                name, splits, rgb, depth, index_of,
                epochs=args.epochs, batch_size=args.batch_size,
                use_depth=not args.no_depth, seed=args.seed, device=device,
            )
        )
        print(f"  ({time.time() - started:.0f}s)")

    grouped = next(r for r in results if r["split"] == "grouped")
    random_arm = next(r for r in results if r["split"] == "random")
    gap = random_arm["test_accuracy"] - grouped["test_accuracy"]

    print(f"\n{'=' * 74}\nRESULT\n{'=' * 74}")
    print(f"  {'':22s}{'random':>12}{'grouped':>12}")
    print(f"  {'session contamination':22s}{random_arm['contaminated_fraction']:>11.1%}{grouped['contaminated_fraction']:>12.1%}")
    print(f"  {'test accuracy':22s}{random_arm['test_accuracy']:>12.4f}{grouped['test_accuracy']:>12.4f}")
    print(f"  {'baseline':22s}{random_arm['baseline_accuracy']:>12.4f}{grouped['baseline_accuracy']:>12.4f}")
    print(f"\n  Leakage inflates test accuracy by {gap:+.4f} ({gap * 100:+.1f} points).")
    print("  The random split is the one that looks clean under a dish_id overlap check.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arms": results, "gap": gap}, indent=2))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
