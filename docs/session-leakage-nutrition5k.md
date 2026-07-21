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
2. The remaining pool is split into train/val two ways: **leaky** (dish-level random — the
   standard practice) and **clean** (session-grouped).
3. Train, select the best checkpoint on validation accuracy — the universal procedure — then
   evaluate that checkpoint once on the common test set.
4. The measured quantity is **inflation = validation accuracy − common-test accuracy**: how
   much the split's own validation score overstates the truth.
5. Run the full grid: **5 seeds × 2 architectures × 2 split strategies = 20 training runs.**
   The two architectures bracket the design space: a small CNN with global average pooling
   (989k parameters) and a faithful reproduction of the common flatten-into-big-FC design
   (1.7M parameters). No augmentation, so the raw effect is not partially masked (augmentation
   destroys some session cues — see [Limitations](#limitations)).

### Results

From [`reports/leakage_rigorous.json`](../reports/leakage_rigorous.json) (20 cells):

| Arm | Mean inflation (val − common test) | Cells positive |
|---|---|---|
| **Leaky split, small CNN** | **+4.3 pts** | 5 / 5 |
| **Leaky split, big CNN** | **+2.4 pts** | 5 / 5 |
| **Leaky split, combined** | **+3.3 pts** | **10 / 10** |
| Clean split, combined | −0.8 pts | 2 / 10 |

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

## What to do about it

**If you use Nutrition5k:** derive sessions from the `dish_id` timestamps and use a grouped
split ([`src/data/nutrition5k.py`](../src/data/nutrition5k.py) implements it; scikit-learn's
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
```

## References

- Thames, Q., et al. *Nutrition5k: Towards Automatic Nutritional Understanding of Generic
  Food.* CVPR 2021.
- Kapoor, S., & Narayanan, A. *Leakage and the Reproducibility Crisis in ML-based Science.*
  Patterns, 2023.
