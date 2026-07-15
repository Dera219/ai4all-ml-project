# AI4ALL — Nutrition5k (personal workspace)

Personal experiment space for my AI4ALL group project: estimating nutritional content from food
photographs.

> ## ⚠️ This is NOT the graded repo
>
> The team's repo is **https://github.com/MalikSCole/AI4all-Group-Project**. That is where work
> gets graded and where contributions are counted.
>
> The Week 8 notes flag **"Contributor only Malik?"** — the instructor has noticed that commit
> history shows one contributor. Work done here is invisible to that. **Push model work to the
> group repo.** Use this repo for scratch experiments you don't want to inflict on teammates —
> nothing else.
>
> Instructor `yzhao2433` should be a collaborator on the group repo, and the Week 3 notes ask for
> regular commits — the cadence is part of what's being assessed, not just the final artifact.

## The project

Team: Malik, Kai, Shyam, Chidera. Dataset:
[Nutrition5k](https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset/data)
(Google Research) — food images with nutritional annotations.

Current results from the group's work: a 224×224 CNN reached **66.9%** test accuracy vs **65.9%**
at 128×128. Higher resolution helped, but modestly — which is itself a finding. It suggests
resolution wasn't the binding constraint, so pushing it further is likely a dead end. The
headroom is probably in one of: augmentation, architecture, or the depth modality.

## Timeline

| When | What | Status |
|---|---|---|
| Week 8 (Jul 8) | Model comparison table, address data leakage | ← last check-in |
| **Week 9 (now)** | Per-person model experiments + 1-slide comparison table | **in progress** |
| **Week 10** | **Streamlit deployment — due *before* the class session** | not started |
| Week 12 | Final in-class presentation; project complete | — |

## Open technical concerns (from Week 8)

**Data leakage — the one that actually matters.** The concern raised was using a validation set
for both model refinement *and* model selection. If you tune against validation and then select
against the same set, the validation score stops being an estimate of generalization and starts
being a number you optimized directly. The reported 66.9% may be optimistic for this reason.

The fix is a genuine three-way split: train (fit), validation (tune and select), test (touched
**once**, at the end, never iterated against). If the test set has already been looked at
repeatedly, it is no longer a test set, and the honest move is to say so in the presentation
rather than quietly report the number.

**Assigned exploration:**

- Augmentation — rotations and brightness jitter via `torchvision.transforms`
- Hyperparameter tuning with cross-validation (Week 5 resources)
- Possibly one iteration of a multimodal model using depth to estimate food mass

## Evaluation checklist (from Ru)

- [ ] Exploratory data analysis
- [ ] Feature importance (top 5 if many variables)
- [ ] Confusion matrix
- [ ] ROC / AUC curves
- [ ] Performance table: accuracy, precision, recall, specificity, F1

A note on that list: **accuracy alone will mislead here.** If nutrition classes are imbalanced,
a model that always predicts the majority class scores well while being useless. Precision and
recall are on the list for a reason — lead with them.

## Ethics

Nutrition5k was captured in a controlled setting with a specific culinary range. A model trained
on it will be least accurate on cuisines least represented in it — and food datasets skew Western
by default. If this tool were ever aimed at real dietary decisions, that skew would mean the
people it serves worst are the ones whose food it never saw. Worth stating plainly in the
presentation; it's the kind of limitation that reads as maturity rather than weakness.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`data/` is gitignored — never commit the dataset. It bloats history permanently and Nutrition5k
has its own license terms.

## License

MIT — see [LICENSE](LICENSE).
