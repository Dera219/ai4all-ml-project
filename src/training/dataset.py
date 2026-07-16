"""Torch dataset over the cached arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

#: ImageNet statistics. Used because they're a reasonable, standard normalization for natural
#: photographs. Note the group's pipeline used ToTensor() only (no normalization) — either is
#: defensible, but the choice must match between training and inference or the model is silently
#: wrong at deploy time.
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#: Depth is its own distribution, not a colour channel; normalize it separately.
DEPTH_MEAN = torch.tensor([0.5]).view(1, 1, 1)
DEPTH_STD = torch.tensor([0.25]).view(1, 1, 1)


class DishDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Serves (image, label, log_calories).

    `augment` applies flips and small rotations. Crucially it is applied to TRAIN ONLY — augmenting
    validation changes what you are measuring against, which makes the number incomparable across
    epochs for reasons that have nothing to do with the model.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        index_of: dict[str, int],
        *,
        augment: bool = False,
        use_depth: bool = True,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.rgb = rgb
        self.depth = depth if use_depth else None
        self.rows = np.array([index_of[d] for d in self.frame.dish_id], dtype=np.int64)
        self.labels = torch.tensor(self.frame.label.to_numpy(), dtype=torch.long)
        self.log_calories = torch.tensor(
            np.log1p(self.frame.calories.to_numpy()), dtype=torch.float32
        )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[i]
        # np.array (not asarray) forces a writable copy out of the read-only memmap. Without it,
        # torch.from_numpy warns about undefined behavior on a non-writable buffer — harmless
        # while we only read, but the augmentation path below writes in place, so the copy is
        # also correctness, not just quiet.
        rgb = torch.from_numpy(np.array(self.rgb[row])).permute(2, 0, 1).float().div_(255.0)
        rgb = (rgb - RGB_MEAN) / RGB_STD

        if self.depth is not None:
            depth = torch.from_numpy(np.array(self.depth[row])).unsqueeze(0).float().div_(255.0)
            depth = (depth - DEPTH_MEAN) / DEPTH_STD
            image = torch.cat([rgb, depth], dim=0)
        else:
            image = rgb

        if self.augment:
            image = self._augment(image)

        return image, self.labels[i], self.log_calories[i]

    @staticmethod
    def _augment(image: torch.Tensor) -> torch.Tensor:
        """Flips and 90-degree rotations.

        Safe here specifically because the camera is directly overhead: a plate rotated 90° or
        mirrored is still a plausible plate. The same augmentation on side-angle photographs
        would teach the model that food falls upward.

        Brightness jitter is applied to RGB only — scaling the depth channel would be claiming
        the food changed height, which is a lie about the physics the channel encodes.
        """
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            image = torch.rot90(image, k, dims=[1, 2])
        if torch.rand(1).item() < 0.3:
            factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
            image[:3] = image[:3] * factor
        return image
