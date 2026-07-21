"""Grad-CAM for the ordinal calorie classifier.

Answers "where did the evidence for this prediction come from?" as a heatmap over the input
image. The method (Selvaraju et al., 2017): take the gradient of one output logit with respect
to the last conv block's feature maps, average it spatially into per-channel weights, and sum
the weighted maps. Regions that pushed the logit up glow; everything else doesn't.

## Which logit to explain

This model has no class logits — it has two ordinal *threshold* logits ("above Low?", "above
Medium?"). Explaining a threshold is actually more informative than explaining a class: for a
Medium prediction, threshold 0 shows what made the model say "more than Low" and threshold 1
shows what stopped it saying "High". The app shows both.

## What a heatmap is NOT

Grad-CAM shows where gradient-weighted evidence concentrated, not a verified causal
explanation. A heatmap on the plate is reassuring; a heatmap on the table edge or a shadow is
a red flag (and with session-level leakage, background heatmaps are exactly what you'd expect
to see — the model reading the room, not the food). That diagnostic use is the point of
shipping this, not decoration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import nn

from src.models.ordinal_cnn import IMAGE_SIZE, eval_transform


class GradCAM:
    """Grad-CAM over a chosen conv layer (default: the model's last, `conv4`).

    Stateless between calls: activations are captured per-call through a temporary forward
    hook, and gradients come from `torch.autograd.grad`, which touches no `.grad` fields on
    the model parameters — running this cannot contaminate any later training or inference.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.conv4

    def heatmap(self, image_tensor: torch.Tensor, threshold_index: int) -> np.ndarray:
        """CAM for one ordinal threshold logit, as a (IMAGE_SIZE, IMAGE_SIZE) array in [0, 1].

        `image_tensor` is a single preprocessed image, shape (1, 3, H, W).
        """
        if threshold_index not in (0, 1):
            raise ValueError(f"threshold_index must be 0 or 1, got {threshold_index}")
        if image_tensor.ndim != 4 or image_tensor.size(0) != 1:
            raise ValueError(f"expected shape (1, 3, H, W), got {tuple(image_tensor.shape)}")

        captured: list[torch.Tensor] = []

        def capture(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
            captured.append(output)

        handle = self.target_layer.register_forward_hook(capture)
        try:
            # Inference runs under no_grad everywhere else; here the graph is the product.
            with torch.enable_grad():
                ordinal_logits, _ = self.model(image_tensor)
                logit = ordinal_logits[0, threshold_index]
                activations = captured[0]
                (grads,) = torch.autograd.grad(logit, activations)
        finally:
            handle.remove()

        # Spatially-averaged gradients weight each channel; ReLU keeps only positive evidence
        # (what argued FOR the threshold, not against it — that is the Grad-CAM definition).
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

        cam = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0, 0]

        cam_np = cam.detach().numpy()
        peak = cam_np.max()
        if peak <= 0:
            # No positive evidence anywhere — a uniform zero map is the honest picture.
            return np.zeros_like(cam_np)
        return cam_np / peak


def overlay(image: Image.Image, cam: np.ndarray, *, alpha: float = 0.45) -> Image.Image:
    """Blend a CAM onto the image using matplotlib's magma colormap.

    The base image is resized to the model's input size first, so the heatmap aligns with what
    the network actually saw — overlaying onto the original aspect ratio would smear the map
    across pixels the model never received.
    """
    from matplotlib import colormaps

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    base = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    colored = (colormaps["magma"](cam)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(colored)
    return Image.blend(base, heat, alpha)


def explain(model: nn.Module, image: Image.Image) -> list[tuple[str, Image.Image]]:
    """Both threshold explanations for one PIL image, ready to display.

    Returns [(label, overlay_image), ...] in threshold order. Uses the same eval transform as
    prediction, so the maps explain the exact tensor the model classified.
    """
    tensor = eval_transform()(image.convert("RGB")).unsqueeze(0)
    cam_engine = GradCAM(model)
    labels = ('evidence for "above Low"', 'evidence for "above Medium"')
    return [
        (label, overlay(image, cam_engine.heatmap(tensor, index)))
        for index, label in enumerate(labels)
    ]
