# Exporting the trained weights

> **RESOLVED (July 2026).** The weights in `models/` no longer come from the group notebook at
> all. They were produced by a self-owned Kaggle kernel
> (`chideraonyebu/nutrition5k-clean-training`) that trains the same architecture on this repo's
> **session-grouped, leak-free split** using this repo's `src/` as the single source of truth.
> Result: **test accuracy 0.7423**, evaluated once, best-val weights. To reproduce or retrain:
> push the kernel again and download `final_ordinal_multitask_model.pth` +
> `regression_target_scaler.pkl` from its output into `models/`.
> The document below is kept for the record of the original problem.

## The problem, stated plainly

**The trained model does not exist anywhere except a Kaggle session that has probably already
expired.**

Every `torch.save` in the group's notebooks writes to `/kaggle/working/`:

```python
best_model_path = "/kaggle/working/best_ordinal_multitask_224.pth"
final_model_path = "/kaggle/working/final_ordinal_multitask_model.pth"
scaler_path = "/kaggle/working/regression_target_scaler.pkl"
```

`/kaggle/working/` is scratch space. It survives only if you "Save & Run All" to create a version,
and it is **not** in the repo — `git ls-files` returns no `.pth`, `.pt`, or `.pkl`.

**Nothing can be deployed until someone exports those two files.** This is the Week 10 blocker,
and it is worth doing before writing another line of anything else.

## What you need

| File | What it is | Why it's needed |
|---|---|---|
| `final_ordinal_multitask_model.pth` | Packaged checkpoint: weights + metadata | The model |
| `regression_target_scaler.pkl` | The `StandardScaler` fit on train targets | Without it the regression head reports standardized units — "calories: -0.42" |

Both come from the final cells of **`Seeded Model.ipynb`**.

## Steps

1. Open `Seeded Model.ipynb` on Kaggle.
2. Attach the Nutrition5k dataset.
3. **Save & Run All (Commit)** — this persists `/kaggle/working/` as a version output. Running
   interactively and closing the tab does not.
4. Open the completed version → **Output** tab.
5. Download `final_ordinal_multitask_model.pth` and `regression_target_scaler.pkl`.
6. Put both in `models/` in this repo.
7. `streamlit run app/streamlit_app.py` — the error banner should be replaced by the uploader.

If the session is gone and the notebook has to be re-run: it is seeded (`SEED = 42`,
`random_state=42`), so the split is reproducible. Training is not bit-for-bit deterministic on
GPU by default, so expect the accuracy to land near 74.1% rather than exactly on it. **If it lands
somewhere materially different, report the number you actually got** — do not present 74.1%
because that's what the old run said.

## Do NOT commit the weights to git

The checkpoint is ~13 MB. Git stores every version forever, so a few re-trains and the repo is
bloated permanently — exactly what happened to `Trading_Algorithm`, which is 12 MB of committed
`node_modules` that can now never be removed from its history without a rewrite.

`models/*.pth` and `models/*.pkl` are gitignored. For sharing, use the Kaggle version output,
Google Drive, or a GitHub Release (which stores binaries outside the git object graph).

## Deploying to Streamlit Community Cloud

The weights being gitignored creates a real tension: Streamlit Cloud deploys **from the repo**, so
a gitignored file won't be there.

Three options, in order of preference:

1. **GitHub Release asset** — upload the `.pth` to a release, and have the app download it on
   first run and cache it. Keeps git clean; adds ~15 lines to `load_model`.
2. **Git LFS** — designed for this, works with Streamlit Cloud, but has a bandwidth quota on the
   free tier.
3. **Commit it anyway** — works, and permanently costs 13 MB of history. Acceptable for a project
   that ends in Week 12; a bad habit to form.

Given the deadline, option 3 is the pragmatic choice **if** you're out of time. Just make it a
decision rather than an accident, and say so in the presentation if asked.

## Sanity check after exporting

```bash
python -c "
import torch
ckpt = torch.load('models/final_ordinal_multitask_model.pth', map_location='cpu', weights_only=False)
print('model     :', ckpt['model_name'])
print('classes   :', ckpt['class_names'])
print('test acc  :', ckpt['test_accuracy'])
print('image size:', ckpt['image_size'])
"
```

Expected:

```
model     : ExtraLayerOrdinalMultiTaskCNN224
classes   : ['Low', 'Medium', 'High']
test acc  : 0.7412731006160165
image size: 224
```

Then `pytest` — 21 tests cover the inference path and run without the real weights.
