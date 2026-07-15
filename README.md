# AI4ALL — Nutrition5k (my work)

My workspace for the AI4ALL Nutrition5k project: the **Streamlit deployment** (Week 10) plus my
own model experiments.

> The team's graded repo is
> [MalikSCole/AI4all-Group-Project](https://github.com/MalikSCole/AI4all-Group-Project). Work that
> counts still needs to land there — the Week 8 notes flag **"Contributor only Malik?"**, so
> commits with my name on them in the group repo are what answer that. The deployment in this repo
> is meant to be contributed upstream, not to live here.

## The task

Classify an overhead food photo as **Low / Medium / High** calorie. Three quantile-derived classes
(`pd.qcut(calories, q=3)`), roughly balanced at ~1081 each, so the random baseline is **33.3%**.

Data: [Nutrition5k](https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset/data). 4,768
dishes in `dish_nutrition_values.csv`, but only **3,244 have an overhead RGB image** — the other
1,524 are dropped.

Model: `ExtraLayerOrdinalMultiTaskCNN224` — four conv blocks → shared FC(128) → two heads:

- **Ordinal head** (2 logits) rather than a 3-way softmax. The classes are *ordered* — Low <
  Medium < High — and a softmax treats confusing Low with High as no worse than confusing Low with
  Medium. Ordinal encoding (`Low=[0,0]`, `Medium=[1,0]`, `High=[1,1]`) puts the ordering into the
  loss.
- **Regression head** (5 outputs: calories, mass, fat, carb, protein) as an auxiliary task, giving
  the shared trunk a richer signal than three labels alone.

Reported: **74.1% test / 77.6% best validation.** Against a 33.3% baseline that's ~2.2x — a real
result. See the caveat below before quoting it.

## Week 10 deployment — built, blocked on one thing

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app is written and tested ([app/streamlit_app.py](app/streamlit_app.py),
[src/models/ordinal_cnn.py](src/models/ordinal_cnn.py), 21 tests). It will not predict yet:

> ### ⚠️ The trained weights don't exist outside a Kaggle session
>
> Every `torch.save` in the notebooks writes to `/kaggle/working/`, which is scratch space. No
> `.pth` is committed anywhere — `git ls-files` in the group repo returns none.
>
> **Nothing deploys until someone exports `final_ordinal_multitask_model.pth` and
> `regression_target_scaler.pkl` from Kaggle.** That's the whole Week 10 critical path. See
> [EXPORT_WEIGHTS.md](EXPORT_WEIGHTS.md).

The app degrades honestly rather than crashing: without weights it renders exactly which two files
are missing and how to get them.

### Two details the app had to get exactly right

**No normalization.** The notebook's eval transform is `Resize((224,224)) + ToTensor()` and
nothing else. Adding the usual ImageNet `Normalize` — the reflex when writing inference code —
would feed the network a distribution it never saw and produce confident garbage with no error.
Pinned by `test_white_image_maps_to_one_not_a_normalized_value`.

**The scaler is not optional.** The regression head was trained on `StandardScaler`-transformed
targets, so its raw output is in standard deviations. Without `inverse_transform` the app would
display "calories: -0.42". If the scaler is missing the app shows nothing rather than nonsense.

## The leakage question — the real answer is uncomfortable

Kai's [`dish-level-data-leakage-check.ipynb`](https://github.com/MalikSCole/AI4all-Group-Project/blob/main/dish-level-data-leakage-check.ipynb)
verifies **0 `dish_id` overlap** across train/val/test. That work is correct and the conclusion
holds: each dish has exactly one overhead image, so the stratified random split is clean at the
dish level. His caveat is also right — if the team ever adds side-angle or multi-view images,
this must become a group split by `dish_id`.

**But that isn't the concern Week 8 raised.** The note was about using the validation set for both
refinement *and* selection. And there's a sharper version of it visible in the notebooks:

```
FoodImageCalorieEstimator.ipynb — Test Accuracy: 0.6694
FoodImageCalorieEstimator.ipynb — Test Accuracy: 0.7002
FoodImageCalorieEstimator.ipynb — Test Accuracy: 0.7125
FoodImageCalorieEstimator.ipynb — Test Accuracy: 0.7146
FoodImageCalorieEstimator.ipynb — Test Accuracy: 0.7228
Seeded Model.ipynb              — test_accuracy: 0.7413
```

**The test set has been evaluated against at least six times**, and each reported number is higher
than the last. That monotonic climb is the signature of selecting on the test set: try a change,
check test, keep it if test improved. Once you've done that, the test set has been optimized
against and is no longer an estimate of generalization — it's a second validation set.

Structural leakage is clean. **Selection leakage is not.** 74.1% is optimistic.

The fix is cheap and it is honesty, not code: say so. *"We iterated against the test set across
model versions, so our true out-of-sample accuracy is likely somewhat below 74.1%. With more time
we'd hold out a fresh split touched exactly once."* That reads as methodological maturity. A
number that collapses when an instructor asks "how many times did you look at the test set?" does
the opposite — and that is a very natural question to ask when the answer is sitting in the commit
history.

## Evaluation checklist (from Ru)

- [ ] Exploratory data analysis
- [ ] Feature importance (top 5)
- [ ] Confusion matrix
- [ ] ROC / AUC
- [ ] Metrics table: accuracy, precision, recall, specificity, F1

**Per-class, not just overall.** The classes are balanced, so accuracy isn't as misleading as it
often is — but with ordered classes the interesting question is *which* mistakes happen. Confusing
Medium with High is a near-miss; confusing Low with High means the model learned nothing about
that image. A confusion matrix shows the difference and a single accuracy number hides it. If the
ordinal head is working, off-by-two errors should be rare — that's a claim worth checking and a
good slide.

## Ethics

Nutrition5k was captured in a controlled setting: fixed overhead camera, consistent lighting, a
specific culinary range. A model trained on it is least accurate on food least like that — phone
photos at an angle, and cuisines the dataset under-represents. Food datasets skew Western by
default, so the people this would serve worst are those whose food it never saw.

Worth stating plainly in the presentation. It reads as maturity, not weakness.

## Layout

```
app/streamlit_app.py        # Week 10 deployment
src/models/ordinal_cnn.py   # Architecture + inference (transcribed from the notebook)
tests/test_inference.py     # 21 tests; run without the real weights
models/                     # gitignored — see EXPORT_WEIGHTS.md
data/                       # gitignored — never commit Nutrition5k
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## License

MIT — see [LICENSE](LICENSE).
