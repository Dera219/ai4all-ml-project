# AI4ALL — Nutrition5k (my independent work)

Calorie classification from food images, built from the raw Nutrition5k dataset — **not** derived
from the group's notebooks. My contribution: a from-scratch model and pipeline, and a
**data-leakage audit** of the split the group used — which turned up a real structural leak the
team's own check missed, then tested honestly whether it actually inflates accuracy (for my model,
it didn't — see Results, which reports that straight rather than burying it).

> The team's shared repo is
> [MalikSCole/AI4all-Group-Project](https://github.com/MalikSCole/AI4all-Group-Project). This
> repo is my own work on the same dataset. Everything here — the pipeline, the model, the
> experiment — I wrote from the raw CSVs and images.

## The finding

The group verified there is **zero `dish_id` overlap** between train, validation, and test. That
check is correct, and under it the split looks clean.

It answers the wrong question.

Nutrition5k dishes were photographed in rapid **capture sessions**: `dish_id` is literally
`dish_<unix_timestamp>`, and the median gap between consecutive dishes is **41 seconds**. Sorting
by timestamp reveals that **96.9% of dishes were shot in a session with other dishes** — same
table, same lighting, same camera pose, same prep batch. Those dishes are not independent samples.

Under the group's stratified-random split:

- `dish_id` overlap between train and test: **0** ✓ (the check they ran)
- test dishes that share a capture session with a training dish: **93.6%** ✗ (the check they didn't)
- fraction of label variance explained by session identity alone: **~24%**

The *risk* this creates: a model can score well by learning "this is session 214's lighting, and
session 214 is Medium" without learning anything about food — correlated images, not identical
ones, which is enough to leak. Whether a given model actually does this is an empirical question,
which is why the experiment below tests it rather than asserting it.

**The fix:** split on capture session, so every dish from a session lands entirely in one set.
[`src/data/nutrition5k.py`](src/data/nutrition5k.py) implements both splits; the experiment trains
the same model under each and measures the gap.

## Results, reported straight

Two parts, and they point in different directions. I'm keeping them separate because conflating
them would be exactly the kind of spin this project exists to avoid.

### Part 1 — the structural leak is real (proven, model-independent)

These numbers are deterministic — they come from the data and the split, not from any trained
model, so there's no seed and no noise:

| | random split | grouped split |
|---|---|---|
| `dish_id` overlap, train↔test | 0 | 0 |
| **test dishes sharing a session with train** | **94.7%** | **0%** |
| label variance explained by session alone | ~24% | — |

The random split — the one the group used, the one that passes a `dish_id` overlap check — leaks
94.7% of its test set through shared capture sessions. That finding stands regardless of anything
below.

### Part 2 — but the leak did *not* inflate my model's accuracy

I expected the leaky split to score higher. It didn't — across every seed I ran, the *clean*
split scored higher. Same model (990K params), only the split differs:

| seed | random (leaky) | grouped (clean) | gap |
|---|---|---|---|
| 42 | 0.724 | 0.734 | −0.9 pt |
| 2 | 0.731 | 0.774 | −4.3 pt |
| mean | 0.727 | **0.754** | **−2.6 pt** |

The clean split scored higher on **both** seeds. The leaky split is never the higher number. At
n≈487 per test set the per-seed gaps are within noise (~2 pt SE), but the *direction* is
consistent, and it is the opposite of "leakage inflates accuracy." **For this model, the session
leak did not inflate test accuracy.** Raw per-seed numbers: `reports/leakage_seed*.json`.

### What I think is going on, and what I won't claim

The honest reading: the leak is structurally present but this model can't exploit it. That is
plausibly *because* of the architecture choices — global average pooling and aggressive
augmentation (flips, 90° rotations, brightness) destroy exactly the session-specific pose and
lighting cues a model would memorize to cheat. A small, heavily-regularized model resists the leak.

The group's model is the opposite: ~3.3M parameters, a `flatten → Linear(25088,128)` trunk, less
augmentation. That architecture is far better equipped to memorize session cues — so the leak may
well inflate *their* 74.1%, even though it didn't inflate my 73.4%. **I can't claim that without
training their model on both splits, so I don't.** It's a hypothesis the experiment sets up, not a
result it delivers.

### What this does and doesn't establish

- **Does:** the split is leaky (proven); a clean, session-grouped evaluation exists; my model hits
  **73.4% on that clean split** — 2.2× the 33% baseline, and honestly measured.
- **Doesn't:** that leakage inflates accuracy in general. For my model it didn't. Whether it
  inflates the group's remains open, and I've said so rather than assumed the convenient answer.

That last line is the whole point. I flagged the group's monotonically-climbing test accuracy as a
red flag; it would be hypocritical to then report a leakage "inflation" number my own experiment
didn't support.

## The model

`CalorieCNN` ([src/models/cnn.py](src/models/cnn.py)) — built for this dataset, with three
deliberate departures from the group's architecture:

| | Group's | Mine | Why |
|---|---|---|---|
| Spatial collapse | `flatten -> Linear(25088, 128)` | global average pool | Their FC is **3.2M params** trained on ~2,300 images. GAP does it with zero. |
| Normalization | none | BatchNorm every conv | Faster, more stable training. |
| Total params | ~3.3M | **~990K** | On data this small, size *is* overfitting. |
| Ordinal decode | `sum(passed thresholds)` | walk to first failure + flag incoherence | `[0,1]` is incoherent for ordered classes; summing hides it as "Medium". |

It keeps the group's good idea — an **ordinal** head (Low < Medium < High are ordered, so a 3-way
softmax is the wrong loss) — and adds an auxiliary calorie-regression head on `log1p(calories)`,
because the target spans 1–3,900 kcal with a long tail.

It also uses **depth**. Nutrition5k ships an overhead depth map per dish, and calories track mass,
mass tracks volume, and volume is exactly what depth sees and a photograph cannot. RGB+depth is a
4-channel input; the depth channel is scaled on a fixed physical range so absolute height survives.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Point at the extracted Nutrition5k archive (the dir with dish_nutrition_values.csv + imagery/)
python experiments/leakage.py --data-root ~/Downloads/archive --epochs 15
```

First run builds a preprocessed image cache (~650 MB, uint8, ~45s). `reports/leakage.json` holds
the numbers; the run prints the comparison table.

The test set is **locked**: `src/training/train.py` selects on validation only, and the experiment
touches test exactly once, at the end. This is a guard against the pattern visible in the group's
notebooks, where test accuracy appears six times climbing monotonically (66.9 → 74.1) — the
signature of selecting on the test set until it improves.

## Layout

```
src/data/nutrition5k.py    # manifest, session derivation, both split strategies, leakage report
src/data/cache.py          # one-time uint8 image cache (RGB + depth)
src/models/cnn.py          # CalorieCNN, ordinal decode, majority baseline
src/training/              # dataset (augmentation, normalization) + training loop
experiments/leakage.py     # the headline experiment
tests/                     # split correctness, model shapes, ordinal decode
```

## Ethics

Nutrition5k was captured with a fixed overhead camera in controlled lighting over one culinary
range. A model trained on it is least accurate on food least like that — phone photos at an angle,
cuisines the dataset under-represents. Food datasets skew Western by default, so the people this
would serve worst are those whose food it never saw. And the leakage point generalizes into an
ethics point: a model that looks accurate in-distribution because of leakage will fail quietly on
real users, which is worse than a model that is honestly, visibly mediocre.

## License

MIT — see [LICENSE](LICENSE).
