"""Nutrition5k data pipeline — built from the raw dataset, independent of the group's notebooks.

## The finding this module exists for

Nutrition5k dishes were photographed in rapid capture sessions: the median gap between
consecutive `dish_id` timestamps is **41 seconds**, and **96.9% of dishes sit in a session with
other dishes**. Same table, same lighting, same camera pose, same prep batch.

`dish_id` is unique per dish, so a stratified random split shows **zero dish_id overlap** and
looks clean. It isn't. Under that split, **94.7% of test dishes share a capture session with a
training dish**, and session identity alone explains ~25% of label variance. The model can score
well by recognizing "this is the lighting from session 214, and session 214 dishes are Medium"
without learning anything about food.

That is leakage. Not identical images — correlated ones, which is enough.

The fix is a **grouped split on session**, so every dish from a capture session lands entirely in
train, or entirely in val, or entirely in test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Consecutive dishes closer than this (seconds) are treated as one capture session.
#:
#: Chosen from the observed gap distribution: the median gap is 41s and the 75th percentile is
#: 105s, so a threshold here separates "next dish on the same table" from "came back later".
#: It is a judgement call — `sessions_for` takes it as a parameter so the sensitivity can be
#: measured rather than assumed.
SESSION_GAP_SECONDS = 120

CLASS_NAMES = ("Low", "Medium", "High")

#: Dishes above this are almost certainly mis-measured: the dataset's max is 7,975 g, which is
#: 8 kg of food on one plate. Kept generous so it only removes the physically absurd.
MAX_PLAUSIBLE_MASS_G = 3000.0


@dataclass(frozen=True, slots=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        return (
            f"train {len(self.train)} ({self.train.session.nunique()} sessions) | "
            f"val {len(self.val)} ({self.val.session.nunique()}) | "
            f"test {len(self.test)} ({self.test.session.nunique()})"
        )


def load_manifest(root: Path | str, *, require_depth: bool = True) -> pd.DataFrame:
    """Build the dish manifest: labels joined to image paths, cleaned.

    Every filter here is applied for a stated reason. Silently dropping rows is how a dataset
    quietly becomes something other than what the paper describes.
    """
    root = Path(root)
    nutrition = root / "dish_nutrition_values.csv"
    if not nutrition.exists():
        raise FileNotFoundError(
            f"{nutrition} not found. Point --data-root at the extracted Nutrition5k archive "
            f"(the directory containing dish_nutrition_values.csv and imagery/)."
        )

    df = pd.read_csv(nutrition)
    image_root = root / "imagery" / "realsense_overhead"

    df["rgb_path"] = df.dish_id.map(lambda d: image_root / d / "rgb.png")
    df["depth_path"] = df.dish_id.map(lambda d: image_root / d / "depth_raw.png")

    # `.exists()` is not enough: dish_1564159636 has a depth_raw.png that exists and is 0 bytes.
    # An existence check passes it and the loader then dies mid-epoch. Size is a cheap proxy for
    # decodability and catches the real failure in this dataset (verified: 0 corrupt RGB,
    # 1 zero-byte depth across all 3,240 dishes).
    df["has_rgb"] = df.rgb_path.map(_is_readable)
    df["has_depth"] = df.depth_path.map(_is_readable)

    total = len(df)
    missing_rgb = int((~df.has_rgb).sum())
    df = df[df.has_rgb]
    missing_depth = int((~df.has_depth).sum())
    if require_depth:
        df = df[df.has_depth]

    # A zero-calorie "dish" is an empty plate or a measurement failure. There is nothing to
    # learn from it and it anchors the bottom quantile.
    zero = (df.calories <= 0).sum()
    df = df[df.calories > 0]

    # 8 kg on a plate is not a meal. These distort the quantile boundaries for everyone else.
    absurd = (df.mass > MAX_PLAUSIBLE_MASS_G).sum()
    df = df[df.mass <= MAX_PLAUSIBLE_MASS_G]

    df = df.reset_index(drop=True)
    df.attrs["drops"] = {
        "total_in_csv": total,
        "missing_or_empty_rgb": missing_rgb,
        "missing_or_empty_depth": missing_depth,
        "zero_calorie": int(zero),
        "implausible_mass": int(absurd),
        "kept": len(df),
    }
    return df


def _is_readable(path: Path) -> bool:
    """Exists AND has bytes in it."""
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def sessions_for(
    df: pd.DataFrame, *, gap_seconds: int = SESSION_GAP_SECONDS
) -> pd.DataFrame:
    """Derive capture sessions from the timestamp embedded in `dish_id`.

    `dish_id` is literally `dish_<unix_timestamp>`. Sorting by it recovers capture order, and a
    gap larger than `gap_seconds` marks a new session.
    """
    out = df.copy()
    out["ts"] = out.dish_id.str.removeprefix("dish_").astype(np.int64)
    out = out.sort_values("ts").reset_index(drop=True)
    gap = out.ts.diff()
    out["session"] = (gap.isna() | (gap >= gap_seconds)).cumsum()
    return out


def add_calorie_classes(
    df: pd.DataFrame, *, edges: np.ndarray | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Bin calories into Low/Medium/High tertiles.

    Returns the frame and the bin edges.

    **Edges must be computed on TRAIN ONLY and passed to val/test.** Computing tertiles over the
    whole dataset lets the test set's calorie distribution influence the class boundaries — a
    small but real leak, and exactly the kind that survives review because `pd.qcut(df.calories)`
    on the full frame reads so naturally.
    """
    out = df.copy()
    if edges is None:
        _, edges = pd.qcut(out.calories, 3, retbins=True, labels=False)
        # Make the outer edges permissive so unseen extremes in val/test still bin.
        edges = np.asarray(edges, dtype=float)
        edges[0], edges[-1] = -np.inf, np.inf
    out["label"] = pd.cut(
        out.calories, bins=edges, labels=[0, 1, 2], include_lowest=True
    )
    out["label"] = out.label.astype(int)
    return out, edges


def grouped_splits(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Splits:
    """Split so that a capture session never spans two sets.

    Sessions are assigned whole, greedily, to hit the target fractions. `GroupShuffleSplit` would
    also work, but it optimizes nothing about balance and produced a 543-dish test set when 486
    was asked for; assigning sessions in shuffled order and tracking the running count lands much
    closer to the requested sizes while keeping the grouping guarantee.
    """
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    sizes = df.groupby("session").size()
    # copy(): pandas hands back a read-only view of the index, and shuffle mutates in place.
    sessions = sizes.index.to_numpy().copy()
    rng.shuffle(sessions)

    total = len(df)
    want_test = int(round(total * test_fraction))
    want_val = int(round(total * val_fraction))

    test_sessions: list[int] = []
    val_sessions: list[int] = []
    count = 0
    for session in sessions:
        if count < want_test:
            test_sessions.append(session)
        elif count < want_test + want_val:
            val_sessions.append(session)
        else:
            break
        count += int(sizes[session])

    test_set, val_set = set(test_sessions), set(val_sessions)
    test = df[df.session.isin(test_set)]
    val = df[df.session.isin(val_set)]
    train = df[~df.session.isin(test_set | val_set)]

    # The guarantee, asserted rather than trusted.
    assert not (set(train.session) & set(val.session)), (
        "session leaked between train and val"
    )
    assert not (set(train.session) & set(test.session)), (
        "session leaked between train and test"
    )
    assert not (set(val.session) & set(test.session)), (
        "session leaked between val and test"
    )

    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def random_splits(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Splits:
    """Stratified random split on dish_id — **the leaky one**.

    This reproduces the split the group used. It is here ONLY as the control arm of the leakage
    experiment: train the same model under both splits and the accuracy gap is the size of the
    leak. Do not use it to report a result.
    """
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df, test_size=val_fraction + test_fraction, random_state=seed, stratify=df.label
    )
    relative = test_fraction / (val_fraction + test_fraction)
    val, test = train_test_split(
        temp, test_size=relative, random_state=seed, stratify=temp.label
    )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def holdout_clean_test(
    df: pd.DataFrame, *, test_fraction: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve off a session-disjoint test set, returning (pool, clean_test).

    This is the yardstick for the rigorous experiment. `clean_test` shares no capture session
    with `pool`, so it is a fair, leakage-free measure of generalization — and, crucially, it is
    the SAME set of dishes whether `pool` is later split leakily or cleanly. Comparing a leaky
    split to a clean one on their own separate test sets confounds leakage with test-set
    difficulty; scoring both on this common held-out test removes that confound.
    """
    rng = np.random.default_rng(seed)
    sizes = df.groupby("session").size()
    sessions = sizes.index.to_numpy().copy()
    rng.shuffle(sessions)

    want = int(round(len(df) * test_fraction))
    test_sessions: list[int] = []
    count = 0
    for s in sessions:
        if count >= want:
            break
        test_sessions.append(s)
        count += int(sizes[s])

    test_set = set(test_sessions)
    clean_test = df[df.session.isin(test_set)].reset_index(drop=True)
    pool = df[~df.session.isin(test_set)].reset_index(drop=True)
    assert not (set(pool.session) & set(clean_test.session)), (
        "held-out test leaked into pool"
    )
    return pool, clean_test


def train_val_split(
    pool: pd.DataFrame,
    *,
    strategy: str,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `pool` into (train, val) either leakily or cleanly.

    - `strategy="leaky"`: random, dish-level, stratified — val shares sessions with train. This
      is what the group did, and it is what makes the *validation* number (used for model
      selection) optimistic.
    - `strategy="clean"`: session-grouped — val is session-disjoint from train.

    Both draw from the same pool at the same val_fraction, so the only difference is whether the
    split respects sessions. That is the single variable under test.
    """
    if strategy == "leaky":
        from sklearn.model_selection import train_test_split

        train, val = train_test_split(
            pool, test_size=val_fraction, random_state=seed, stratify=pool.label
        )
        return train.reset_index(drop=True), val.reset_index(drop=True)

    if strategy == "clean":
        rng = np.random.default_rng(seed)
        sizes = pool.groupby("session").size()
        sessions = sizes.index.to_numpy().copy()
        rng.shuffle(sessions)
        want = int(round(len(pool) * val_fraction))
        val_sessions: list[int] = []
        count = 0
        for s in sessions:
            if count >= want:
                break
            val_sessions.append(s)
            count += int(sizes[s])
        val_set = set(val_sessions)
        val = pool[pool.session.isin(val_set)].reset_index(drop=True)
        train = pool[~pool.session.isin(val_set)].reset_index(drop=True)
        assert not (set(train.session) & set(val.session)), (
            "clean val leaked into train"
        )
        return train, val

    if strategy == "official":
        # The dataset's own split, used the way a practitioner would: dishes on the official
        # test side become validation, the rest is train. Note what this does NOT do — it never
        # moves an official-test dish into train to hit a size target, because that would
        # violate the very boundary being tested. Val size is therefore whatever the official
        # assignment yields within this pool, not `val_fraction`.
        official_test = official_split_ids()["test"]
        is_val = pool.dish_id.isin(official_test)
        val = pool[is_val].reset_index(drop=True)
        train = pool[~is_val].reset_index(drop=True)
        if val.empty or train.empty:
            raise ValueError(
                "the official split left one side empty for this pool — check that dish_ids "
                "match the official lists (they cover the full dataset, not just RGB)"
            )
        return train, val

    raise ValueError(
        f"strategy must be 'leaky', 'clean', or 'official', got {strategy!r}"
    )


#: Where the dataset publishes its own split files. Documented in the Nutrition5k README, which
#: states that all incremental scans of a unique plate are held within the same split.
OFFICIAL_SPLIT_BASE = (
    "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits"
)


def official_split_ids(*, cache_dir: Path | str = ".cache") -> dict[str, set[str]]:
    """The official RGB train/test dish IDs, fetched once and cached on disk.

    Returned as sets so membership tests are cheap. Cached because the experiment calls this
    once per cell and a network round-trip per cell would be both slow and a silent dependency
    on connectivity partway through a training run.
    """
    cache = Path(cache_dir) / "official_splits"
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, set[str]] = {}
    for side in ("train", "test"):
        local = cache / f"rgb_{side}_ids.txt"
        if not local.exists():
            import urllib.request

            url = f"{OFFICIAL_SPLIT_BASE}/rgb_{side}_ids.txt"
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 — https literal
                local.write_bytes(r.read())
        out[side] = {
            line.strip() for line in local.read_text().splitlines() if line.strip()
        }
    return out


def session_leakage_report(splits: Splits) -> dict[str, float | int]:
    """How much of `test` is contaminated by session-mates in `train`?

    Zero under `grouped_splits`. ~93% under `random_splits`. This function is what turns "we
    thought about leakage" into a number.
    """
    train_sessions = set(splits.train.session)
    contaminated = splits.test.session.isin(train_sessions).sum()
    return {
        "test_dishes": len(splits.test),
        "test_dishes_with_session_mate_in_train": int(contaminated),
        "contaminated_fraction": float(contaminated / len(splits.test))
        if len(splits.test)
        else 0.0,
        "shared_sessions": len(train_sessions & set(splits.test.session)),
    }
