# AI4ALL — Nutrition5k (my work)

My personal workspace for the AI4ALL Nutrition5k project: model experiments, EDA, and the
evaluation artifacts I'm responsible for.

> **The team's graded repo is [MalikSCole/AI4all-Group-Project](https://github.com/MalikSCole/AI4all-Group-Project).**
> This repo is where I develop and keep my own experiments. Work that counts toward the project
> still needs to land there — the Week 8 notes flag **"Contributor only Malik?"**, so commits with
> my name on them in the group repo are what answer that. Develop here, push what lands to the
> group repo, and keep the cadence visible (Week 3 notes ask for regular commits, with
> `yzhao2433` added as a collaborator).

## The project

Team: Malik, Kai, Shyam, Chidera. Dataset:
[Nutrition5k](https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset/data) (Google
Research) — food images with nutritional annotations.

Where the team is: a 224×224 CNN reached **66.9%** test accuracy vs **65.9%** at 128×128. The gain
from doubling resolution was small, which is itself informative — it suggests resolution wasn't
the binding constraint, so pushing it further is likely a dead end. Headroom is more plausibly in
augmentation, architecture, or the depth modality.

## Timeline

| When | What | Status |
|---|---|---|
| Week 8 (Jul 8) | Model comparison table; address leakage | last check-in |
| **Week 9 (now)** | My model experiments + 1-slide comparison table | **in progress** |
| **Week 10** | **Streamlit deployment — due *before* the session** | not started |
| Week 12 | Final in-class presentation | — |

## My experiments

Week 8 asked each member to train models and report train/validation performance in a shared
table. Log every run here — including the ones that fail. A run that didn't help is evidence,
and "we tried X and it didn't move the metric" is a stronger presentation beat than silence.

| Run | Change | Train acc | Val acc | Notes |
|---|---|---|---|---|
| baseline | 224×224 CNN (team's) | — | — | reference point |
| | | | | |

Assigned exploration from Week 8:

- **Augmentation** — rotations and brightness jitter via `torchvision.transforms`. Apply to the
  *training* set only. Augmenting validation changes what you're measuring against.
- **Hyperparameter tuning with cross-validation** (Week 5 resources)
- **Multimodal** — depth to estimate food mass, possibly one iteration

## The leakage concern — read before reporting any number

The Week 8 concern was using the validation set for both refinement *and* selection. If you tune
against a set and then select against the same set, its score stops being an estimate of
generalization and becomes a number you optimized directly. The 66.9% may be optimistic for this
reason.

A genuine three-way split: **train** (fit) → **validation** (tune and select) → **test** (touched
exactly once, at the end, never iterated against).

If the test set has already been evaluated against repeatedly, it isn't a test set anymore. The
honest move is to say so in the presentation. Judges and instructors respond well to "here's the
methodological limitation we found and how we'd fix it" — and badly to a number that collapses
under one question.

## Evaluation checklist (from Ru)

- [ ] Exploratory data analysis
- [ ] Feature importance (top 5 if many variables)
- [ ] Confusion matrix
- [ ] ROC / AUC
- [ ] Metrics table: accuracy, precision, recall, specificity, F1

**Accuracy alone will mislead here.** If nutrition classes are imbalanced, always predicting the
majority class scores well while being useless. Lead with precision and recall — that's why
they're on the list.

## Ethics

Nutrition5k was captured in a controlled setting with a specific culinary range. A model trained
on it is least accurate on cuisines least represented in it, and food datasets skew Western by
default. If this were ever aimed at real dietary decisions, the people it serves worst would be
those whose food it never saw. Worth stating plainly — it reads as maturity, not weakness.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`data/` is gitignored. Never commit the dataset: it bloats history permanently and Nutrition5k
carries its own license terms.

## License

MIT — see [LICENSE](LICENSE).
