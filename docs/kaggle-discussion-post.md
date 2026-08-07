<!--
Ready to paste as a Kaggle discussion topic on the dataset page:
https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset/discussion  →  "New Topic"

Title goes in the title field; everything below the rule is the body (Kaggle renders Markdown).
-->

**Title:** Heads-up: a random dish-level split leaks — 94.7% of test dishes share a capture session with training dishes

---

If you're splitting this dataset randomly at the dish level and checking for `dish_id` overlap
between train and test, that check passes and the split still leaks. I measured how much, and
wanted to leave it here since this dataset gets used for a lot of calorie-estimation projects.

**What's going on.** `dish_id` is `dish_<unix_timestamp>`. Sort by ID and the capture process
shows up: the median gap between consecutive dishes is **41 seconds**. Dishes were photographed
in batches — same table, same lighting, same camera pose, plate after plate in one sitting.
Treating a gap over 120 seconds as a session boundary (results aren't sensitive to the exact
threshold):

| Statistic (raw data, RGB manifest, n = 3,241) | Value |
|---|---|
| Dishes in a session containing other dishes | 96.9% |
| Test dishes sharing a session with a *training* dish (stratified random split, seed 42) | **94.7%** |
| Calorie-tertile label variance explained by session identity alone | 25.4% |

That last row is the mechanism. Dishes in one session came from the same meal service, so their
calorie labels correlate — and their pixels correlate through lighting, background, and camera
pose. A model can raise its score by learning "this is session 214's lighting, and session 214
ran heavy" without learning anything about food. The images aren't duplicated, they're
*correlated*, which is exactly what a dish-ID overlap check cannot see.

**How much it costs.** 5 seeds × 2 architectures, all evaluated against one common
session-disjoint test set. Selecting the checkpoint on the session-contaminated split overstated
true accuracy by **+3.3 points on average, positive in 10 of 10 runs**. Selecting on a
session-grouped split tracked truth (−0.8, positive in 2 of 10). It's architecture-independent —
the damage comes from a contaminated *validation* set rewarding the wrong checkpoint, not from a
particular model memorizing images. So it will show up in your numbers whatever backbone you use.

**The fix is one line** — group the split by session instead of splitting dishes independently:

```python
# session id: start a new session when the gap to the previous dish exceeds 120s
ts = df.dish_id.str.removeprefix("dish_").astype(int).sort_values()
session = (ts.diff() > 120).cumsum()

from sklearn.model_selection import GroupShuffleSplit
train_idx, test_idx = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, groups=session)
)
```

Your accuracy will go *down*. That's the point — the lower number is the one that survives
contact with a photo taken in a different room.

Full write-up with methodology, the limitations, and reproduction commands:
https://github.com/Dera219/ai4all-ml-project/blob/main/docs/session-leakage-nutrition5k.md

One thing worth flagging honestly: my first attempt at measuring this was confounded — I compared
a leaky test set against a *different* clean one, so "how much does leakage inflate scores" got
mixed up with "which dishes are harder," and it reached the wrong conclusion. The write-up shows
that detour too. If you spot a hole in the corrected design I'd genuinely like to hear it.
