"""The airtight leakage experiment.

Answers a question the first experiment couldn't, and fixes a confound it had.

## The confound it fixes

`leakage.py` compared a leaky split's test set to a clean split's test set. Those are different
dishes, so any accuracy difference mixes "leakage" with "one test set is harder." Here, every
model — leaky or clean, big or small, every seed — is scored on ONE common held-out test set of
session-disjoint dishes. That test set is clean for everyone, so differences are attributable to
the training/selection split, not to which dishes landed in test.

## The question it answers

Two architectures:
- **small** — my `CalorieCNN` (~100K-param trunk, global average pooling)
- **big** — `GroupStyleCNN`, a faithful reproduction of the group's flatten→giant-FC model

Three split strategies for train/val (the pool is identical; only the partition differs):
- **leaky** — random dish-level split; val shares capture sessions with train
- **clean** — session-grouped split; val is session-disjoint from train
- **official** — the dataset's own `dish_ids/splits` files; val is the official-test side.
  This is the practitioner's realistic middle case: it groups a plate's incremental scans, but
  not whole capture sessions, so ~35% of it still shares a session with train. Its inflation
  should land between the other two — that gap is the cost of the leakage the shipped split
  does not cover.

For each (seed, arch, strategy) we record:
- `val_acc`  — accuracy on that split's own validation (the number you'd select on / report)
- `test_acc` — accuracy on the common clean held-out test (the truth)
- inflation  = val_acc − test_acc

If leakage inflates a model's reported number, `val_acc` will exceed `test_acc` under the leaky
split — and the hypothesis is that the big model inflates far more than the small one, because it
has the capacity to memorize session cues. No augmentation, so this measures the raw effect;
augmentation is a mitigation, tested separately.

Run:  python experiments/leakage_rigorous.py --data-root ~/Downloads/archive
"""

from __future__ import annotations

import argparse
import json
import statistics as st
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
    holdout_clean_test,
    load_manifest,
    sessions_for,
    train_val_split,
)
from src.models.cnn import CalorieCNN, GroupStyleCNN  # noqa: E402
from src.training.dataset import DishDataset  # noqa: E402
from src.training.train import evaluate, pick_device, train  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
ARCHITECTURES = ("small", "big")
STRATEGIES = ("leaky", "clean", "official")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_model(arch: str) -> torch.nn.Module:
    # RGB only — holding channels constant so the comparison is purely architecture, and matching
    # what the group actually trained on (they did not use depth).
    return (
        CalorieCNN(in_channels=3) if arch == "small" else GroupStyleCNN(in_channels=3)
    )


def loader(frame, rgb, index_of, *, batch_size: int, shuffle: bool) -> DataLoader:
    ds = DishDataset(frame, rgb, None, index_of, augment=False, use_depth=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_cell(
    seed: int,
    arch: str,
    strategy: str,
    pool,
    clean_test,
    rgb,
    index_of,
    *,
    epochs: int,
    batch_size: int,
    device,
    val_fraction: float = 0.2,
) -> dict:
    seed_everything(seed)
    train_frame, val_frame = train_val_split(
        pool, strategy=strategy, seed=seed, val_fraction=val_fraction
    )

    # Class edges from THIS train fold only, applied to val and the common test — so no split's
    # own distribution sets the boundaries.
    train_frame, edges = add_calorie_classes(train_frame)
    val_frame, _ = add_calorie_classes(val_frame, edges=edges)
    test_frame, _ = add_calorie_classes(clean_test, edges=edges)

    train_loader = loader(
        train_frame, rgb, index_of, batch_size=batch_size, shuffle=True
    )
    val_loader = loader(val_frame, rgb, index_of, batch_size=batch_size, shuffle=False)
    test_loader = loader(
        test_frame, rgb, index_of, batch_size=batch_size, shuffle=False
    )

    model = make_model(arch)
    # LazyLinear needs one forward to materialize before the optimizer sees its params.
    with torch.no_grad():
        model(torch.zeros(1, 3, rgb.shape[1], rgb.shape[2]))

    history, best_state = train(
        model, train_loader, val_loader, epochs=epochs, device=device, verbose=False
    )

    model.load_state_dict(best_state)
    model.to(device)
    _, val_acc, _, _ = evaluate(model, val_loader, device, regression_weight=0.3)
    _, test_acc, test_mae, incoherent = evaluate(
        model, test_loader, device, regression_weight=0.3
    )

    cell = {
        "seed": seed,
        "arch": arch,
        "strategy": strategy,
        "params": int(model.n_parameters),
        "val_acc": round(val_acc, 4),
        "test_acc": round(test_acc, 4),
        "inflation": round(val_acc - test_acc, 4),
        "test_mae": round(test_mae, 1),
        "incoherent": round(incoherent, 4),
        "best_epoch": history.best_epoch,
        "n_train": len(train_frame),
        "n_val": len(val_frame),
        "n_test": len(test_frame),
    }
    print(
        f"  seed {seed} {arch:5s} {strategy:5s}  "
        f"val {val_acc:.3f}  test {test_acc:.3f}  inflation {cell['inflation']:+.3f}  "
        f"(params {model.n_parameters:,})"
    )
    return cell


def aggregate(cells: list[dict]) -> dict:
    out = {}
    for arch in ARCHITECTURES:
        for strategy in STRATEGIES:
            group = [
                c for c in cells if c["arch"] == arch and c["strategy"] == strategy
            ]
            if not group:
                continue
            key = f"{arch}/{strategy}"

            def ms(values: list[float]) -> dict:
                return {
                    "mean": round(st.mean(values), 4),
                    "sd": round(st.pstdev(values), 4) if len(values) > 1 else 0.0,
                }

            out[key] = {
                "n_seeds": len(group),
                "val_acc": ms([c["val_acc"] for c in group]),
                "test_acc": ms([c["test_acc"] for c in group]),
                "inflation": ms([c["inflation"] for c in group]),
                "params": group[0]["params"],
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path.home() / "Downloads/archive"
    )
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help=(
            "Validation share for the leaky and clean arms. The official arm ignores it — its "
            "val size is whatever the shipped assignment yields (~16%% of the pool). Set this "
            "to 0.16 to equalise val sizes across all three arms and remove selection-noise "
            "as an explanation for the official arm's inflation."
        ),
    )
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument(
        "--out", type=Path, default=Path("reports/leakage_rigorous.json")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES),
        choices=list(STRATEGIES),
        help="Subset of arms to run — lets a follow-up control rerun only what it needs.",
    )
    args = parser.parse_args()

    seeds = args.seeds
    device = pick_device()
    print(
        f"device: {device} | size: {args.size} | seeds: {seeds} | no augmentation, RGB only"
    )

    manifest = sessions_for(load_manifest(args.data_root))
    manifest, _ = add_calorie_classes(manifest)
    print(f"dishes: {len(manifest)} | sessions: {manifest.session.nunique()}")

    paths = build_cache(manifest, args.cache, size=args.size, with_depth=True)
    rgb = np.load(paths["rgb"], mmap_mode="r")
    ids = np.load(paths["ids"], allow_pickle=True)
    index_of = {d: i for i, d in enumerate(ids)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    started = time.time()

    for seed in seeds:
        # Held-out clean test depends on the seed, so across seeds we sample different clean test
        # sets — but within a seed it is common to all six (arch × strategy) cells.
        pool, clean_test = holdout_clean_test(manifest, test_fraction=0.2, seed=seed)
        print(
            f"\nseed {seed}: pool {len(pool)} ({pool.session.nunique()} sessions) | "
            f"common clean test {len(clean_test)} ({clean_test.session.nunique()} sessions)"
        )
        for arch in ARCHITECTURES:
            for strategy in args.strategies:
                cells.append(
                    run_cell(
                        seed,
                        arch,
                        strategy,
                        pool,
                        clean_test,
                        rgb,
                        index_of,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=device,
                        val_fraction=args.val_fraction,
                    )
                )
                # Write after every cell so a crash loses at most one training run.
                args.out.write_text(
                    json.dumps({"cells": cells, "summary": aggregate(cells)}, indent=2)
                )

    summary = aggregate(cells)
    print(
        f"\n{'=' * 78}\nAIRTIGHT RESULT  ({len(seeds)} seeds, {(time.time() - started) / 60:.0f} min)\n{'=' * 78}"
    )
    print(
        f"  {'arch/split':16s}{'params':>10}{'val_acc':>14}{'test_acc':>14}{'inflation':>14}"
    )
    for key, agg in summary.items():
        print(
            f"  {key:16s}{agg['params']:>10,}"
            f"{agg['val_acc']['mean']:>9.3f}±{agg['val_acc']['sd']:.3f}"
            f"{agg['test_acc']['mean']:>9.3f}±{agg['test_acc']['sd']:.3f}"
            f"{agg['inflation']['mean']:>+9.3f}±{agg['inflation']['sd']:.3f}"
        )

    big_leaky = summary["big/leaky"]["inflation"]["mean"]
    big_clean = summary["big/clean"]["inflation"]["mean"]
    small_leaky = summary["small/leaky"]["inflation"]["mean"]
    print(
        "\n  Inflation = how much a split's own validation overstates true (common-test) accuracy."
    )
    print(f"  big model, leaky split : {big_leaky:+.3f}   (clean: {big_clean:+.3f})")
    print(f"  small model, leaky split: {small_leaky:+.3f}")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
