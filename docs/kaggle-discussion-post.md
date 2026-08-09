<!--
Ready to paste as a Kaggle discussion topic on the dataset page:
https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset/discussion  →  "New Topic"

Title goes in the title field; everything below the rule is the body (Kaggle renders Markdown).
-->

**Title:** Nutrition5k ships official train/test splits — most notebooks ignore them, and even they leave 34.7% session leakage

---

Two things about splitting this dataset that cost me real accuracy points, in case they save
someone else the trouble.

**1. The dataset ships official splits, and almost nobody uses them.**

Nutrition5k publishes train/test ID lists at `dish_ids/splits/` in the GCS bucket, and the
README explains why they exist: *"All incremental scans that compose a unique plate are held
within the same split, to avoid overlap between the train and test splits."* Dishes here are
built up incrementally — the same plate re-photographed after each ingredient is added — so
those scans are near-duplicates, and the official split deliberately keeps them together.

Nearly every public notebook on this dataset, mine included, called `train_test_split` on the
manifest and never opened that directory. If you do that, you throw away a protection the
authors built for you.

**2. Grouping by *plate* isn't the same as grouping by *session*.**

`dish_id` is literally the string `dish_` followed by a unix timestamp. Sort by ID and the
capture process appears: the median gap
between consecutive dishes is **41 seconds**. Plates were photographed in batches — same table,
same lights, same camera pose, one sitting. Two *different* plates four minutes apart share all
of that, and the official grouping doesn't cover them, because they are genuinely different
plates.

Measured on the same 3,239-dish RGB manifest, treating a >120s gap as a session boundary:

| Split | Test dishes sharing a capture session with a training dish |
|---|---|
| Stratified random dish-level split (seed 42) | **94.7%** |
| Official `rgb_train_ids` / `rgb_test_ids` | **34.7%** |

So the shipped files remove most of it and not all of it.

The absolute number depends on where you draw the session boundary, so here's the sweep rather
than just the most striking row — official split: 9.4% at 60s, 34.7% at 120s, 66.3% at 300s,
84.7% at 600s.

**Why it matters.** Session identity alone explains **25.4%** of calorie-tertile label variance.
Dishes in one session came from the same meal service, so their labels correlate — and their
pixels correlate through lighting, background, and pose. A model can raise its score by learning
"this is session 214's lighting, and session 214 ran heavy." The images aren't duplicated,
they're *correlated*, which is exactly what a `dish_id` overlap check cannot see.

**What it costs.** 5 seeds × 2 architectures × 3 split strategies = 30 training runs, every
model scored on one common session-disjoint test set. Inflation is validation accuracy minus
true accuracy — how much the split's own number overstates reality:

| Train/val split | Inflation | Positive |
|---|---|---|
| Random dish-level | +2.2pt | 9/10 |
| **Official `dish_ids/splits`** | **+3.5pt** | **9/10** |
| Session-grouped | −0.0pt | 3/10 |

I expected the official split to land between the other two. It didn't — its inflation is at
least as large as the random split's, and the ordering holds within both architectures
separately. The damage is a contaminated *validation* set rewarding the wrong checkpoint, not a
particular model memorizing images, so it shows up whatever backbone you use.

**What I'd suggest.** Start from the official split files. Then group by session on top:

```python
# session id: new session when the gap to the previous dish exceeds 120s
ts = df.dish_id.str.removeprefix("dish_").astype(int).sort_values()
session = (ts.diff() > 120).cumsum()

from sklearn.model_selection import GroupShuffleSplit
train_idx, test_idx = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, groups=session)
)
```

Your accuracy will go *down*. That's the point — the lower number is the one that survives
contact with a photo taken in a different room.

**Check the 34.7% yourself in about ten seconds.** It needs no images and no download — dish IDs
carry the timestamps, so it's arithmetic on the official ID lists:
https://github.com/Dera219/ai4all-ml-project/blob/main/experiments/official_split_check.py

Full write-up — methodology, limitations, reproduction commands:
https://github.com/Dera219/ai4all-ml-project/blob/main/docs/session-leakage-nutrition5k.md

Three honest caveats, because two of these could change how you read the table.

**I am not claiming the official split is worse than random.** Its validation set is 16% of the
pool against 20% for the other arms — its size is fixed by the shipped assignment, not chosen —
and a smaller validation set makes checkpoint selection noisier all on its own. That confound is
uncontrolled. "No better than random" is what the data supports; "worse" is not.

**Inflation magnitude is hardware-dependent.** The same code and seeds give +3.3pt on Apple MPS
and +2.2pt on the Kaggle P100 these numbers came from. Only comparisons *within* one run mean
anything — which is why all 30 cells above come from a single machine.

**Sessions are inferred from timestamps, not recorded.** The dataset publishes no session field,
so every session number here is a reconstruction, applied identically to all three arms.

Also worth flagging: my first attempt at measuring the effect was confounded — I compared a
leaky test set against a *different* clean one, mixing "how much does leakage inflate scores"
with "which dishes are harder," and it reached the wrong conclusion. The write-up shows that
detour too. If you spot a hole in the corrected design I'd genuinely like to hear it.
