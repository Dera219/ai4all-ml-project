# Official Nutrition5k split IDs

Vendored copies of the dataset's own train/test lists, fetched from

    https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/

on 2026-08-08. They are ~100 KB of plain text and they never change, so vendoring them makes
the `official` split arm reproducible on a machine with no internet — which is the normal
configuration for a Kaggle kernel, and the environment this experiment actually runs in.

`src/data/nutrition5k.official_split_ids()` prefers this directory, falls back to a local
`.cache/` copy, and only reaches for the network if neither exists.
