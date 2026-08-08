"""Kaggle runner for the 30-cell three-arm leakage experiment.

Answers the question the write-up currently can't: the official Nutrition5k split leaves 34.7%
of test dishes sharing a capture session with training data — what does that cost?

Three train/val strategies over an identical pool, every model scored on one common
session-disjoint test set:

  leaky     random dish-level split            (published: +3.3pt inflation, 10/10 positive)
  official  the dataset's own dish_ids/splits  (unknown — this is the missing number)
  clean     session-grouped split              (published: -0.8pt, 2/10 positive)

2 architectures x 3 strategies x 5 seeds = 30 cells.

Why this runs here and not locally: an 8 GB MacBook swaps itself to a standstill on cell 2. A
Kaggle T4 has 16 GB of dedicated GPU memory and finishes.

**This kernel needs no internet.** The official split IDs are vendored in the code dataset at
data/official_splits/, and `official_split_ids()` prefers them over the network — so leave
`enable_internet` off, exactly as the prior training kernel had it.

Setup, mirroring chideraonyebu/nutrition5k-clean-training:
  - dataset sources: gillesokhin/nutrition5k-dataset  +  chideraonyebu/nutrition5k-clean-code
  - accelerator: GPU
  - internet: OFF

Writes /kaggle/working/leakage_rigorous.json — download it, drop it into reports/, and run
`python experiments/summarize_arms.py` for the verdict.
"""

import shutil
import sys
from pathlib import Path

_INPUT = Path("/kaggle/input")
print("input mounts:", sorted(str(p) for p in _INPUT.iterdir()))


def _find(marker: str) -> Path:
    """Locate a marker file at any depth Kaggle has mounted things at.

    Kaggle strips the top folder of uploaded zips and the layout has changed between versions,
    so discovering by marker beats assuming a path — and a miss prints the directory listing
    rather than a bare FileNotFoundError.
    """
    for depth in range(1, 6):
        hits = sorted(_INPUT.glob("*/" * depth + marker))
        if hits:
            return hits[0]
    listing = [str(p) for p in _INPUT.glob("*/" * 3) if p.is_dir()]
    raise FileNotFoundError(f"{marker} not found under {_INPUT}; dirs at depth 3: {listing}")


# --- stage the repo code somewhere importable -------------------------------------------
# /tmp, not /kaggle/working — working/ becomes the published kernel output.
_CODE_MARKER = _find("data/nutrition5k.py")
_CODE_ROOT = _CODE_MARKER.parent.parent          # .../src
_REPO_ROOT = _CODE_ROOT.parent                   # repo root as uploaded
_PKG = Path("/tmp/_code")
if not (_PKG / "src").exists():
    shutil.copytree(_CODE_ROOT, _PKG / "src")
    # Vendored official split lists travel with the code; copy them into the same relative
    # position so `VENDORED_SPLITS` (repo_root/data/official_splits) resolves.
    vendored = _REPO_ROOT / "data" / "official_splits"
    if vendored.exists():
        shutil.copytree(vendored, _PKG / "data" / "official_splits")
    for extra in ("experiments",):
        src = _REPO_ROOT / extra
        if src.exists():
            shutil.copytree(src, _PKG / extra)
sys.path.insert(0, str(_PKG))
print("code root:", _CODE_ROOT, "| repo root:", _REPO_ROOT)

from src.data.nutrition5k import official_split_ids  # noqa: E402

# Fail fast and loudly if the split lists didn't travel — the alternative is discovering it
# ten cells into a GPU run, or worse, silently falling through to a network call that can't
# succeed with internet off.
# Discover the split lists by marker rather than by assumed path — Kaggle has changed mount
# layouts before, and the relative-path guess is the one thing here that would fail silently.
_search = [_REPO_ROOT / "data" / "official_splits", _PKG / "data" / "official_splits"]
try:
    _search.insert(0, _find("official_splits/rgb_train_ids.txt").parent)
except FileNotFoundError:
    pass
_ids = official_split_ids(search=_search)
assert len(_ids["train"]) == 4059 and len(_ids["test"]) == 709, (
    f"official split lists look wrong: train={len(_ids['train'])} test={len(_ids['test'])}"
)
print(f"official splits OK: train {len(_ids['train'])}, test {len(_ids['test'])}")

DATA_ROOT = _find("dish_nutrition_values.csv").parent
print("data root:", DATA_ROOT)

# --- run the experiment ------------------------------------------------------------------
sys.argv = [
    "leakage_rigorous.py",
    "--data-root", str(DATA_ROOT),
    "--cache", "/tmp/_cache",
    "--out", "/kaggle/working/leakage_rigorous.json",
]

import runpy  # noqa: E402

runpy.run_path(str(_PKG / "experiments" / "leakage_rigorous.py"), run_name="__main__")

print("\ndone — download /kaggle/working/leakage_rigorous.json into reports/, then run")
print("  python experiments/summarize_arms.py")
