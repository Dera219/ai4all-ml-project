# AI4ALL Project

> **Scaffold status:** structure only. The **`TODO`** sections need your project's specifics —
> left blank on purpose rather than guessed at.

## Problem statement

`TODO` — AI4ALL frames projects around a social-impact question. What is yours, and who is
affected by it?

## Data

`TODO` — dataset, source, license, and size. Note any collection bias up front: who is in this
data, who is missing, and how that skews what a model trained on it can fairly claim.

`data/` is gitignored. Raw data should never be committed — it bloats history permanently and can
leak licensed or personal information. Document how to fetch it in `src/data/` instead.

## Approach

`TODO` — baseline first, then the model you are actually testing. A dumb baseline (majority class,
simple linear model) is not a formality; it is the only thing that tells you whether the fancy
model earned its complexity.

## Results

`TODO` — fill in after evaluation. Report the metric that matches the stakes: accuracy is
misleading on imbalanced data, and if errors cost different amounts in different directions, say
so and measure accordingly.

## Ethics

`TODO` — required framing for AI4ALL, and worth real thought rather than boilerplate: who could
this model harm if it is wrong, and is it wrong more often for some groups than others?

## Layout

```
data/{raw,interim,processed}   # gitignored
notebooks/                     # Exploration
src/data/                      # Loading, fetching, cleaning
src/features/                  # Feature engineering
src/models/                    # Train + predict
src/visualization/             # Plots
models/                        # Serialized artifacts (gitignored)
reports/figures/               # Final figures
tests/
```

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set a random seed everywhere and record it. A result you cannot reproduce is a result you cannot
defend when someone asks.

## License

MIT — see [LICENSE](LICENSE).
