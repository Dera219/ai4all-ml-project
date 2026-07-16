"""Preprocess images once into a memmapped uint8 cache.

Decoding 3,243 PNGs per epoch would make the GPU wait on libpng. Resizing once and storing
uint8 gives fast epochs on a laptop.

uint8, not float32, on purpose: at 224x224 the RGB cache is ~488 MB as uint8 and ~1.9 GB as
float32. On an 8 GB machine that difference is the difference between a cache that fits beside
the training process and one that doesn't. Normalization to float happens per-batch, where it's
cheap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SIZE = 224


def _resize_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _resize_depth(path: Path, size: int) -> np.ndarray:
    """Load a 16-bit depth map and rescale it to uint8.

    Depth is stored as uint16 millimetres. Naively casting to uint8 would wrap around and turn a
    smooth surface into stripes. Per-image min/max scaling is also wrong — it destroys absolute
    scale, which is the whole reason depth is interesting for estimating mass.

    So: clip to a fixed physical range and scale linearly. The camera sits a fixed distance above
    the table, so a shared range is meaningful across dishes, and 0 stays "no reading".
    """
    with Image.open(path) as img:
        raw = np.asarray(img, dtype=np.float32)

    # Observed range for the overhead rig; clipping keeps the plate's height variation in the
    # informative part of the scale instead of compressing it against far-field noise.
    near, far = 0.0, 1200.0
    clipped = np.clip(raw, near, far)
    scaled = (clipped - near) / (far - near) * 255.0
    depth = scaled.astype(np.uint8)
    return np.asarray(Image.fromarray(depth).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def build_cache(
    manifest: pd.DataFrame,
    out_dir: Path | str,
    *,
    size: int = IMAGE_SIZE,
    with_depth: bool = True,
    force: bool = False,
) -> dict[str, Path]:
    """Write `rgb.npy` (N,H,W,3) and optionally `depth.npy` (N,H,W) plus `dish_ids.npy`.

    Row order matches `manifest`, so downstream code can index by position. `dish_ids.npy` is
    written alongside so that assumption is checkable rather than implicit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = out_dir / f"rgb_{size}.npy"
    depth_path = out_dir / f"depth_{size}.npy"
    ids_path = out_dir / f"dish_ids_{size}.npy"

    paths = {"rgb": rgb_path, "ids": ids_path}
    if with_depth:
        paths["depth"] = depth_path

    if not force and all(p.exists() for p in paths.values()):
        cached_ids = np.load(ids_path, allow_pickle=True)
        if len(cached_ids) == len(manifest) and (cached_ids == manifest.dish_id.to_numpy()).all():
            return paths
        # A stale cache silently trains on the wrong labels — rebuild rather than trust it.

    n = len(manifest)
    rgb = np.lib.format.open_memmap(rgb_path, mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    depth = (
        np.lib.format.open_memmap(depth_path, mode="w+", dtype=np.uint8, shape=(n, size, size))
        if with_depth
        else None
    )

    for i, row in enumerate(manifest.itertuples()):
        rgb[i] = _resize_rgb(Path(row.rgb_path), size)
        if depth is not None:
            depth[i] = _resize_depth(Path(row.depth_path), size)
        if (i + 1) % 500 == 0:
            print(f"  cached {i + 1}/{n}")

    rgb.flush()
    if depth is not None:
        depth.flush()
    np.save(ids_path, manifest.dish_id.to_numpy())
    return paths
