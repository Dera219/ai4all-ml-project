"""The trained architecture, and the inference path that must mirror the notebook exactly.

## Why this file is a transcription, not a redesign

The weights in `final_ordinal_multitask_model.pth` were produced by
`ExtraLayerOrdinalMultiTaskCNN224` in `Seeded Model.ipynb`. `load_state_dict` matches on layer
names and shapes, so every attribute name here (`conv1`, `shared_fc`, `ordinal_head`, ...) is
load-bearing. Renaming `shared_fc` to something tidier makes the checkpoint fail to load.

The preprocessing is the subtler trap. The notebook's eval transform is:

    transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

There is **no Normalize**. Adding the usual ImageNet normalization here — the reflex when writing
inference code — would feed the network inputs from a distribution it never saw and produce
confident garbage, with no error to warn you. Preprocessing must match training exactly or the
model is silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import nn
from torchvision import transforms

#: Must match `image_size` in the checkpoint.
IMAGE_SIZE = 224

#: Ordinal targets: Low=[0,0], Medium=[1,0], High=[1,1].
CLASS_NAMES = ("Low", "Medium", "High")

#: Order the StandardScaler was fit on. Do not reorder — inverse_transform is positional.
REGRESSION_TARGETS = ("calories", "mass", "fat", "carb", "protein")


class ExtraLayerOrdinalMultiTaskCNN224(nn.Module):
    """Four conv blocks into a shared FC, then two heads.

    Ordinal head (2 logits) rather than a 3-way softmax because the classes are ordered: Low <
    Medium < High. A softmax treats confusing Low with High as no worse than confusing Low with
    Medium. The ordinal encoding makes the model predict "is it above threshold 1?" and "above
    threshold 2?", which bakes the ordering into the loss.

    Regression head (5 outputs) is auxiliary: predicting calories/mass/fat/carb/protein gives the
    shared trunk a richer signal than three class labels alone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # 224 -> 112 -> 56 -> 28 -> 14 after four pools.
        self.shared_fc = nn.Linear(128 * 14 * 14, 128)
        self.dropout = nn.Dropout(0.3)

        self.ordinal_head = nn.Linear(128, 2)
        self.regression_head = nn.Linear(128, 5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        features = F.relu(self.shared_fc(x))
        features = self.dropout(features)

        return self.ordinal_head(features), self.regression_head(features)


def eval_transform() -> transforms.Compose:
    """Inference preprocessing. Mirrors `eval_transform_mt` in the notebook.

    Resize + ToTensor only. See the module docstring on why there is no Normalize.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def decode_ordinal(ordinal_logits: torch.Tensor) -> tuple[int, bool]:
    """Turn 2 threshold logits into a class index.

    Returns `(class_index, is_consistent)`.

    The notebook decodes with `(sigmoid(logits) >= 0.5).sum()`. That is standard, but it silently
    accepts an **inconsistent** prediction: `[0, 1]` means "not above the first threshold, but
    above the second," which is incoherent for ordered classes. Summing maps it to 1 (Medium) as
    if nothing happened.

    We reproduce the notebook's class index exactly — changing it would make this app disagree
    with the reported 74.1% — but we also return whether the thresholds were coherent, so the UI
    can say when the model is confused rather than hiding it behind a confident label.
    """
    probabilities = torch.sigmoid(ordinal_logits)
    passed = (probabilities >= 0.5).int()
    class_index = int(passed.sum().item())
    # Coherent patterns: [0,0], [1,0], [1,1]. Incoherent: [0,1].
    is_consistent = not (passed[0].item() == 0 and passed[1].item() == 1)
    return class_index, is_consistent


@dataclass(frozen=True, slots=True)
class Prediction:
    class_index: int
    class_name: str
    threshold_probabilities: tuple[float, float]
    is_consistent: bool
    #: Only populated when the scaler is available; otherwise the raw head output is
    #: standardized and meaningless as nutrition figures.
    nutrition: dict[str, float] | None


class CalorieClassifier:
    """Loads the checkpoint package and predicts on a single image."""

    def __init__(self, model: nn.Module, metadata: dict[str, Any], scaler: Any | None) -> None:
        self.model = model
        self.metadata = metadata
        self.scaler = scaler
        self._transform = eval_transform()

    @classmethod
    def load(cls, checkpoint_path: Path | str, scaler_path: Path | str | None = None) -> CalorieClassifier:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path}. The weights are NOT in this repo — they "
                f"were written to /kaggle/working/ and are not committed. See EXPORT_WEIGHTS.md."
            )

        # weights_only=False: the checkpoint is a dict of metadata alongside the tensors, and we
        # wrote it ourselves. Never point this at a checkpoint from an untrusted source —
        # unpickling arbitrary files executes arbitrary code.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"{checkpoint_path} has no 'model_state_dict'. Expected the packaged checkpoint "
                f"from the final cell of 'Seeded Model.ipynb', not a bare state_dict."
            )

        model = ExtraLayerOrdinalMultiTaskCNN224()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        scaler = None
        if scaler_path is not None and Path(scaler_path).exists():
            import joblib

            scaler = joblib.load(scaler_path)

        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        return cls(model=model, metadata=metadata, scaler=scaler)

    def predict(self, image: Image.Image) -> Prediction:
        tensor = self._transform(image.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            ordinal_logits, regression_output = self.model(tensor)

        logits = ordinal_logits[0]
        class_index, is_consistent = decode_ordinal(logits)
        probabilities = torch.sigmoid(logits)

        nutrition: dict[str, float] | None = None
        if self.scaler is not None:
            # The regression head was trained on StandardScaler-transformed targets, so its raw
            # output is in standardized units. Reporting it directly would show "calories: -0.4".
            unscaled = self.scaler.inverse_transform(regression_output.numpy())[0]
            nutrition = dict(zip(REGRESSION_TARGETS, (float(v) for v in unscaled), strict=True))

        return Prediction(
            class_index=class_index,
            class_name=CLASS_NAMES[class_index],
            threshold_probabilities=(float(probabilities[0]), float(probabilities[1])),
            is_consistent=is_consistent,
            nutrition=nutrition,
        )
