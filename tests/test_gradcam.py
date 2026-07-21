"""Grad-CAM: shape, range, isolation, and honesty of the zero-evidence case.

These run on a randomly-initialized model — Grad-CAM's mechanics (hooks, gradients, upsampling,
normalization) are weight-independent, and random weights keep the suite runnable without the
checkpoint, same as the rest of the tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.gradcam import GradCAM, explain, overlay
from src.models.ordinal_cnn import IMAGE_SIZE, ExtraLayerOrdinalMultiTaskCNN224, eval_transform


@pytest.fixture(scope="module")
def model() -> ExtraLayerOrdinalMultiTaskCNN224:
    torch.manual_seed(7)
    m = ExtraLayerOrdinalMultiTaskCNN224()
    m.eval()
    return m


@pytest.fixture(scope="module")
def food_photo() -> Image.Image:
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), "RGB")


@pytest.fixture
def tensor(food_photo: Image.Image) -> torch.Tensor:
    return eval_transform()(food_photo).unsqueeze(0)


class TestHeatmap:
    def test_shape_range_and_finiteness(self, model, tensor) -> None:
        for index in (0, 1):
            cam = GradCAM(model).heatmap(tensor, index)
            assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE)
            assert np.isfinite(cam).all()
            assert cam.min() >= 0.0
            assert cam.max() <= 1.0

    def test_thresholds_get_different_maps(self, model, tensor) -> None:
        cam0 = GradCAM(model).heatmap(tensor, 0)
        cam1 = GradCAM(model).heatmap(tensor, 1)
        assert not np.allclose(cam0, cam1)

    def test_rejects_bad_inputs(self, model, tensor) -> None:
        with pytest.raises(ValueError, match="threshold_index"):
            GradCAM(model).heatmap(tensor, 2)
        with pytest.raises(ValueError, match="shape"):
            GradCAM(model).heatmap(tensor[0], 0)  # missing batch dim

    def test_leaves_no_trace_on_the_model(self, model, tensor) -> None:
        """Explaining must not contaminate the model: no parameter gradients, no lingering
        hooks, and predictions identical before and after."""
        with torch.no_grad():
            before = model(tensor)[0].clone()

        GradCAM(model).heatmap(tensor, 0)

        assert all(p.grad is None for p in model.parameters())
        assert not model.conv4._forward_hooks  # hook removed even on the happy path

        with torch.no_grad():
            after = model(tensor)[0]
        assert torch.equal(before, after)

    def test_hook_removed_even_when_forward_raises(self, model) -> None:
        bad_input = torch.rand(1, 3, 16, 16)  # too small for the FC layer → forward raises
        with pytest.raises(RuntimeError):
            GradCAM(model).heatmap(bad_input, 0)
        assert not model.conv4._forward_hooks


class TestOverlayAndExplain:
    def test_overlay_returns_aligned_rgb(self, food_photo) -> None:
        cam = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        out = overlay(food_photo, cam)
        assert out.mode == "RGB"
        assert out.size == (IMAGE_SIZE, IMAGE_SIZE)

    def test_overlay_rejects_bad_alpha(self, food_photo) -> None:
        cam = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        with pytest.raises(ValueError, match="alpha"):
            overlay(food_photo, cam, alpha=1.5)

    def test_explain_returns_both_thresholds(self, model, food_photo) -> None:
        results = explain(model, food_photo)
        assert [label for label, _ in results] == [
            'evidence for "above Low"',
            'evidence for "above Medium"',
        ]
        for _, img in results:
            assert img.size == (IMAGE_SIZE, IMAGE_SIZE)
