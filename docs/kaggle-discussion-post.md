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

`dish_id` is `dish_<unix_timestamp>`. Sort by ID and the capture process appears: the median gap
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

**What it costs.** 5 seeds × 2 architectures, every model scored on one common session-disjoint
test set. Selecting the checkpoint on a session-contaminated split overstated true accuracy by
**+3.3 points on average, positive in 10 of 10 runs**; selecting on a session-grouped split
tracked truth (−0.8, positive in 2 of 10). It's architecture-independent — the damage is a
contaminated *validation* set rewarding the wrong checkpoint, not a particular model memorizing
images — so it shows up whatever backbone you use.

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

Two honest caveats. The 34.7% measures **contamination, not accuracy** — I haven't re-run the
controlled experiment on the official partition, so I can't tell you how many points it costs
there, only that the contamination is present. And sessions are **inferred from timestamps**,
not recorded; the dataset publishes no session field, so every session number here is a
reconstruction, applied identically to both splits.

Also worth flagging: my first attempt at measuring the effect was confounded — I compared a
leaky test set against a *different* clean one, mixing "how much does leakage inflate scores"
with "which dishes are harder," and it reached the wrong conclusion. The write-up shows that
detour too. If you spot a hole in the corrected design I'd genuinely like to hear it.
