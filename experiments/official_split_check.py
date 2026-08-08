"""Does Nutrition5k's OFFICIAL split already fix session-level leakage?

The dataset's README states that "all incremental scans that compose a unique plate are held
within the same split, to avoid overlap between the train and test splits" — so the authors
grouped deliberately, and shipped split files under `dish_ids/splits/`. That raises the obvious
question the main experiment doesn't answer: if you use the official split instead of
`train_test_split`, is the problem gone?

Answer: mostly, not entirely. A capture **session** is a strictly coarser unit than a **plate**.
The official grouping keeps one plate's incremental scans together; it does not keep two
*different* plates photographed minutes apart on the same table together. Those pairs still
straddle the split.

This script needs no images — dish IDs carry the timestamp, so contamination is arithmetic on
the ID lists. It fetches the official lists over HTTPS (~100 KB) and reports contamination at
several session thresholds, restricted to whichever dishes you actually have a manifest for so
the number is comparable to the write-up's 94.7%.

    python experiments/official_split_check.py

Writes reports/official_split_check.json.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.nutrition5k import load_manifest, sessions_for  # noqa: E402

SPLIT_BASE = (
    "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits"
)
#: Thresholds to report. The write-up's headline uses 120s; the others are here because the
#: absolute contamination rate is genuinely sensitive to this choice and hiding that would be
#: the same sin the write-up is about.
GAPS = (60, 120, 300, 600)
OUT = Path("reports/official_split_check.json")


def _fetch_ids(name: str) -> set[str]:
    url = f"{SPLIT_BASE}/{name}"
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 — fixed https literal above
        return {line.strip() for line in r.read().decode().splitlines() if line.strip()}


def _manifest_ids() -> set[str]:
    """The dishes this project actually trained on, so the number is comparable to the write-up.

    Two sources, cheapest first: the cached ID array written by the training pipeline, or a full
    manifest build if the extracted dataset happens to be on this machine. Neither is required —
    without one, the script still reports contamination over all official IDs.
    """
    for cache in (Path(".cache/dish_ids_224.npy"), Path(".cache/dish_ids_160.npy")):
        if cache.exists():
            import numpy as np

            return {str(x) for x in np.load(cache, allow_pickle=True).tolist()}

    root = Path(os.environ.get("NUTRITION5K_ROOT", "data/raw/nutrition5k"))
    if (root / "dish_nutrition_values.csv").exists():
        return set(load_manifest(root, require_depth=False).dish_id)
    return set()


def contamination(dish_ids: list[str], split: dict[str, str], gap: int) -> dict[str, float]:
    """Share of test dishes sitting in a session that also contains a train dish."""
    frame = sessions_for(pd.DataFrame({"dish_id": dish_ids}), gap_seconds=gap)
    members: dict[int, list[str]] = {}
    for dish, session in zip(frame.dish_id, frame.session, strict=True):
        members.setdefault(int(session), []).append(dish)
    session_of = {d: int(s) for d, s in zip(frame.dish_id, frame.session, strict=True)}

    test = [d for d in dish_ids if split[d] == "test"]
    hit = sum(
        1
        for d in test
        if any(split[o] == "train" for o in members[session_of[d]] if o != d)
    )
    return {
        "gap_seconds": gap,
        "sessions": len(members),
        "test_dishes": len(test),
        "contaminated": hit,
        "contaminated_pct": round(100 * hit / len(test), 1) if test else 0.0,
    }


def main() -> None:
    train = _fetch_ids("rgb_train_ids.txt")
    test = _fetch_ids("rgb_test_ids.txt")
    official = {d: "train" for d in train} | {d: "test" for d in test}

    scopes: dict[str, list[str]] = {"official_all": sorted(official)}
    manifest = _manifest_ids()
    if manifest:
        scopes["rgb_manifest"] = sorted(manifest & set(official))
    else:
        print("(no manifest found — reporting official_all only; see _manifest_ids)")

    report: dict[str, object] = {"source": SPLIT_BASE, "scopes": {}}
    for scope, ids in scopes.items():
        rows = [contamination(ids, official, g) for g in GAPS]
        n_test = sum(1 for d in ids if official[d] == "test")
        report["scopes"][scope] = {"dishes": len(ids), "test_dishes": n_test, "by_gap": rows}  # type: ignore[index]
        print(f"\n{scope}: {len(ids)} dishes ({n_test} official test)")
        for r in rows:
            print(
                f"  gap>{r['gap_seconds']:>3}s  {r['sessions']:>4} sessions  "
                f"test sharing a session with train: "
                f"{r['contaminated']}/{r['test_dishes']} = {r['contaminated_pct']:>5.1f}%"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
