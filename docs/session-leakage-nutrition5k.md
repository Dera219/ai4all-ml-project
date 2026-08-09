# The split looks clean and isn't: session-level leakage in Nutrition5k

**Chidera Onyebu** · University of Maryland · July 2026
*Code and full results: this repository. Every number below is reproducible from the raw
dataset with the commands in [Reproducing](#reproducing).*

## TL;DR

Nutrition5k dishes were photographed in rapid **capture sessions** — same table, same lighting,
same camera pose, minutes apart. A standard stratified random split at the dish level passes the
obvious integrity check (zero dish-ID overlap between train and test) while placing **94.7% of
test dishes in a capture session that also has dishes in the training set**. In a controlled
experiment (5 seeds × 2 architectures, one common session-disjoint test set), model selection on
the session-contaminated split overstated true accuracy by **+3.3 points on average, positive in
10 out of 10 cells**, while selection on a session-grouped split tracked truth (−0.8, positive
in 2 of 10). The effect is **architecture-independent**: it comes from the contaminated
*validation* set rewarding the wrong checkpoint, not from any particular model memorizing images.
The fix costs one line: group the split by capture session.

The dataset's own split files help but do not close it — they group *incremental scans of one
plate*, and under them **34.7%** of the same test dishes still share a session with training
data. That contamination has a measured cost: in a 30-cell rerun, model selection on the official
split inflated by **+3.5pt, positive in 9 of 10 cells** — no better than the random split it was
supposed to improve on, while the session-grouped arm sat at −0.0pt. So the advice is two-part:
use the shipped splits instead of `train_test_split`, and group by session on top of them,
because the shipped splits alone do not protect selection.

Inflation magnitudes are hardware-dependent — identical code and seeds give +3.3pt on Apple MPS
and +2.2pt on a Kaggle P100 — so only comparisons *within* a single run are meaningful.

## Background

[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (Thames et al., CVPR
2021) is a Google Research dataset of ~5,000 cafeteria dishes with per-dish mass, calories, and
macronutrients, captured with overhead RGB-D cameras. It is a common choice for student and
research projects on visual calorie estimation. The natural task setup — the one our AI4ALL
project group used, and the one most public notebooks use — is: bin calories into tertiles
(Low / Medium / High), split dishes randomly with stratification, train a CNN, report test
accuracy.

The natural integrity check is: *does any dish appear in both train and test?* Each `dish_id`
is unique and each dish has one overhead image, so this check passes with zero overlap, and the
split gets called clean. Our group ran exactly this check. It passed.

**The dataset authors anticipated a version of this.** Nutrition5k ships train/test files under
`dish_ids/splits/`, and the README states that "all incremental scans that compose a unique
plate are held within the same split, to avoid overlap between the train and test splits."
Dishes are built up incrementally — one plate photographed again after each ingredient is added
— so those scans are near-duplicates, and the official split keeps them together. That is a real
safeguard and it is not what this write-up rediscovers. The question this write-up asks is what
happens one level coarser: a *plate* is not a *session*, and rolling your own random split
discards even the plate-level protection. Both parts are measured below —
[what the official split does and doesn't fix](#does-the-official-split-fix-it).

## The flaw

`dish_id` is literally `dish_<unix_timestamp>`. Sort the dishes by ID and the capture process
reappears: the median gap between consecutive dishes is **41 seconds**. Dishes were photographed
in batches — plate after plate on the same table, under the same lights, from the same rig, in
one sitting. Treating a >120-second gap as a session boundary (the results are not sensitive to
the exact threshold; see [Limitations](#limitations)):

| Statistic (computed from raw data, RGB manifest, n = 3,241) | Value |
|---|---|
| Dishes captured in a session containing other dishes | **96.9%** |
| Test dishes sharing a session with a *training* dish, under a stratified random split (seed 42) | **94.7%** |
| Calorie-tertile label variance explained by session identity alone | **25.4%** |

The third row is the mechanism. Dishes in one session are not independent samples: they came
from the same meal service, so their calorie labels correlate — and their *pixels* correlate
through lighting, background, and camera pose. A model can therefore improve its score by
learning "this is session 214's lighting, and session 214 dishes ran heavy" without learning
anything about food. Not identical images — **correlated** ones. That is leakage, and a dish-ID
overlap check is structurally unable to see it.

## An honest detour: our first experiment was confounded

Our first attempt at measuring the damage compared models evaluated on a leaky test set versus a
*different* clean test set. The two test sets contained different dishes, so the comparison
conflated "how much does leakage inflate scores" with "which set of dishes is harder" — and the
noise from the second question swamped the first. That experiment concluded, wrongly, that the
leak did not inflate accuracy. It remains in this repository
([`experiments/leakage.py`](../experiments/leakage.py)) with its wrong conclusion on the record,
because deleting failed experiments is its own form of selection bias.

## The experiment that isolates it

Design ([`experiments/leakage_rigorous.py`](../experiments/leakage_rigorous.py)):

1. **One common test set per seed**, held out *session-disjoint* from everything else. Every
   arm within a seed is evaluated on the identical dishes, removing test-difficulty as a
   variable.
2. The remaining pool is split into train/val three ways: **leaky** (dish-level random — the
   standard practice), **clean** (session-grouped), and **official** (the dataset's shipped
   `dish_ids/splits` files). The official arm was added after the first two were published; its
   results are in [What that contamination costs](#what-that-contamination-costs).
3. Train, select the best checkpoint on validation accuracy — the universal procedure — then
   evaluate that checkpoint once on the common test set.
4. The measured quantity is **inflation = validation accuracy − common-test accuracy**: how
   much the split's own validation score overstates the truth.
5. Run the full grid: **5 seeds × 2 architectures × 3 split strategies = 30 training runs**
   (the original published run was 20, before the official arm existed).
   The two architectures bracket the design space: a small CNN with global average pooling
   (989k parameters) and a faithful reproduction of the common flatten-into-big-FC design
   (1.7M parameters). No augmentation, so the raw effect is not partially masked (augmentation
   destroys some session cues — see [Limitations](#limitations)).

### Results

From [`reports/leakage_rigorous_20cell_2seed-arch.json`](../reports/leakage_rigorous_20cell_2seed-arch.json),
the original 20-cell run (Apple MPS):

| Arm | Mean inflation (val − common test) | Cells positive |
|---|---|---|
| **Leaky split, small CNN** | **+4.3 pts** | 5 / 5 |
| **Leaky split, big CNN** | **+2.4 pts** | 5 / 5 |
| **Leaky split, combined** | **+3.3 pts** | **10 / 10** |
| Clean split, combined | −0.8 pts | 2 / 10 |

> **The magnitude is not portable across hardware; the ordering is.** Rerunning this identical
> code, with identical seeds, on a Kaggle P100 gave leaky **+2.2pt / 9 of 10** and clean
> **−0.0pt / 3 of 10** — same conclusion, different numbers, because floating-point and cuDNN
> behaviour differ between backends. Any single inflation figure here should be read as "this
> magnitude, on this hardware," and comparisons should only be made *within* one run. The
> [three-arm table](#what-that-contamination-costs) below is internally consistent for that
> reason: all 30 of its cells come from one machine.

Three things worth stating precisely:

- **The direction is unanimous.** Every leaky cell — both architectures, all five seeds —
  overstated its true accuracy. Every clean arm's validation score was, on average, a slight
  *under*-estimate, which is what an honest validation set looks like.
- **The effect is architecture-independent.** We hypothesized the higher-capacity flatten-FC
  model would inflate *more* (more capacity to memorize session cues). It inflated slightly
  less. That refuted our hypothesis and revealed the actual mechanism: the inflation lives in
  the **contaminated validation set** — it rewards whichever checkpoint best exploits session
  cues, regardless of architecture — not in any specific model's memorization. This makes the
  finding *stronger*: it applies to any model trained under this split, including ones that
  don't exist yet.
- **The size matters at leaderboard scale.** Three points is the difference between "beats the
  published baseline" and "doesn't" in most reports on this dataset. Our group's own headline
  number (74.1% under a dish-random split) should be read as ≈71% true generalization.

### The clean number

Trained end-to-end on the session-grouped split (edges and scaler fit on train only, test
evaluated exactly once with best-validation weights): **74.2% test accuracy** against a 33.3%
random baseline — roughly the same headline as the leaky pipeline reports, but this one means
what it says.

## Does the official split fix it?

Partly. Since the dataset ships grouped splits, the useful question is not "is leakage possible"
but "how much survives if you use the files provided." Contamination is arithmetic on the ID
timestamps — no images needed — so this is cheap to answer directly
([`experiments/official_split_check.py`](../experiments/official_split_check.py), which fetches
the official lists over HTTPS and needs nothing downloaded):

| Split, evaluated on the same 3,239-dish RGB manifest | Test dishes sharing a capture session with a training dish |
|---|---|
| Stratified random dish-level split (seed 42) — what our group used | **94.7%** |
| **Nutrition5k official `rgb_train/test_ids`** | **34.7%** |

Two conclusions, and they point in opposite directions:

**The official split helps a lot.** Rolling your own random split nearly triples session
contamination. If you take one thing from this document, take that: *use the shipped split
files.* Most public Nutrition5k notebooks, ours included, call `train_test_split` on the
manifest and never look at `dish_ids/splits/` at all.

**It does not close the gap.** A third of the official test set still shares a capture session
with training data, because the official unit is the plate and the leak lives at the session.
Two different plates photographed four minutes apart on the same table, under the same lights,
can legitimately land on opposite sides of the official split — they are different plates, which
is all the documented guarantee promises.

The absolute figure moves with the session threshold, so it should never be quoted without one:

| Session gap threshold | Official split, same 3,239-dish manifest |
|---|---|
| 60s | 9.4% |
| **120s** (used throughout) | **34.7%** |
| 300s | 66.3% |
| 600s | 84.7% |

The specific percentage is a function of where you draw the session boundary, so the honest
presentation is the sweep rather than the single most striking row. The random-split figure is
reported at 120s only — reproducing it at other thresholds needs the calorie labels the
stratification depends on, and it is not quoted here at any threshold it was not measured at.

### What that contamination costs

Contamination is not damage. To find out whether 34.7% actually inflates anything, the
controlled experiment was rerun with the official split as a third arm — 2 architectures × 3
strategies × 5 seeds = 30 cells, every model scored on the same session-disjoint held-out test
set ([`experiments/leakage_rigorous.py --strategies leaky clean official`](../experiments/leakage_rigorous.py)):

| Train/val strategy | val size | Inflation (val − true) | Positive |
|---|---|---|---|
| Random dish-level split | 20% of pool | +2.2pt | 9/10 |
| **Official `dish_ids/splits`** | 16% of pool | **+3.5pt** | **9/10** |
| Session-grouped | 20% of pool | −0.0pt | 3/10 |

**The prediction registered before this ran was wrong.** The expectation was that official would
land between the other two — partial protection. It did not: its inflation is *at least as large*
as the random split's, in 9 of 10 runs, and the ordering holds within both architectures
separately (small: +3.7 official vs +3.1 leaky; big: +3.2 vs +1.3). The decision rule, also
fixed in advance ([`experiments/summarize_arms.py`](../experiments/summarize_arms.py)), returns
its verdict B: **the shipped split does not protect model selection.**

That is the claim. A stronger one — that the official split is *worse* than a random split — is
**not** supported here, because of a confound stated in the analysis before the numbers existed:
official val is 16% of the pool against 20% for the other arms, since its size is fixed by the
shipped assignment rather than chosen. A smaller validation set makes checkpoint selection
noisier, which inflates on its own. Equalising val size would settle it; that run has not been
done, and until it is, "no better than random" is the defensible reading and "worse" is not.

Note also what this does *not* undermine: the session-grouped arm sits at −0.0pt with only 3 of
10 positive. Whatever the official split is doing, grouping by session removes it.

## What to do about it

**If you use Nutrition5k:** start from the official split files under `dish_ids/splits/` rather
than `train_test_split` — that alone removes most of the contamination. To close the rest, derive
sessions from the `dish_id` timestamps and group by them
([`src/data/nutrition5k.py`](../src/data/nutrition5k.py) implements it; scikit-learn's
`GroupShuffleSplit` also works). It is one line of difference and it changes what your test
number measures.

**The general lesson:** *split by the process that generated the data, not by the row ID.*
Sample-ID overlap checks pass whenever rows are superficially unique — patients with multiple
scans, users with multiple sessions, sensors with multiple readings, dishes photographed in
batches. If any identifiable process makes samples correlated (a capture session, a hospital, a
device, a day), that process is the unit that must not span train and test. This failure class
is documented across fields (Kapoor & Narayanan, *Patterns*, 2023); the contribution here is a
measured, controlled instance of it in a widely used dataset, with the mechanism isolated.

## Limitations

- Training for the 20-cell grid ran at 160px, RGB-only, without augmentation, to isolate the
  raw effect at feasible cost. Relative inflation, not absolute accuracy, is the measured
  quantity; absolute numbers elsewhere in the repo use 224px.
- Augmentation (flips, rotations, color jitter) destroys some session cues and partially masks
  the effect — our first (confounded) experiment used heavy augmentation, which is consistent
  with its null result. Mitigation-by-augmentation is real but incidental; the grouped split is
  the fix.
- The 120-second session threshold is a judgment call from the gap distribution (median 41s,
  75th percentile 105s). `sessions_for` takes it as a parameter so sensitivity can be measured
  rather than assumed.
- Contamination percentages vary about a point with manifest filtering and split details
  (94.0%–96.5% across reasonable variants); we report the RGB-manifest, seed-42,
  versus-train-set figure and pin the method in code.
- One dataset, one task. The mechanism (correlated capture ⇒ contaminated validation ⇒
  inflated selection) is general; the +3.3pt magnitude is specific to this setup.
- The official arm's validation set is 16% of the pool against 20% for the other two, because
  its size is fixed by the shipped assignment rather than chosen. Smaller validation sets make
  checkpoint selection noisier, which inflates independently of leakage. This is why the claim
  is "no better than random" and not "worse than random" — equalising val size (the
  `--val-fraction` flag exists for exactly this) would settle it, and that run has not been done.
- Inflation magnitudes vary with hardware. Identical code and seeds gave leaky +3.3pt on Apple
  MPS and +2.2pt on a Kaggle P100. Only within-run comparisons are meaningful; no cross-run
  magnitude comparison is made anywhere in this document.
- Session membership is inferred from ID timestamps, not recorded ground truth. The dataset
  publishes no session field, so every session figure in this document — mine and the official
  split's alike — is a reconstruction. It is applied identically to both, so the comparison is
  fair even where the absolute number is uncertain.

## Reproducing

```bash
# environment
pip install -r requirements.txt

# the headline statistics (94.7% / 96.9% / 25.4%) — from the raw dataset
python - <<'PY'
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.nutrition5k import load_manifest, sessions_for
df = sessions_for(load_manifest(Path("<dataset-root>"), require_depth=False))
df["label"] = pd.qcut(df.calories, 3, labels=False)
tv, test = train_test_split(df, test_size=.15, stratify=df.label, random_state=42)
train, _ = train_test_split(tv, test_size=.15/.85, stratify=tv.label, random_state=42)
print(f"contaminated: {test.session.isin(set(train.session)).mean():.1%}")
PY

# the 20-cell controlled experiment (writes reports/leakage_rigorous.json)
python experiments/leakage_rigorous.py --data-root <dataset-root>

# the official-split comparison (34.7% and the threshold sweep).
# Needs no dataset download — it fetches the official ID lists and does arithmetic on the
# timestamps. Uses .cache/dish_ids_*.npy to restrict to this project's manifest if present.
python experiments/official_split_check.py     # writes reports/official_split_check.json
```

## Discussion

A condensed version of this is posted on the dataset's Kaggle page, where the people most likely
to hit the problem will actually see it:
[Nutrition5k ships official train/test splits — most notebooks ignore them, and even they leave
34.7% session leakage](https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset/discussion/733741)
(August 2026). Corrections and counterarguments are welcome there or as an issue on this repo —
the limitations above are the places I'd attack first.

## References

- Thames, Q., et al. *Nutrition5k: Towards Automatic Nutritional Understanding of Generic
  Food.* CVPR 2021.
- Kapoor, S., & Narayanan, A. *Leakage and the Reproducibility Crisis in ML-based Science.*
  Patterns, 2023.
