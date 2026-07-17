"""Inference-path tests.

These guard the ways a deployment can be *silently* wrong. None of them need the real weights —
they run against a random-weight checkpoint built in the exact packaged format, because what's
being tested is the plumbing, not the model's accuracy.

The failure mode they exist for: an app that loads, accepts an image, and confidently returns a
label computed from mis-scaled inputs. No error, no crash, just wrong — which is exactly the
class of bug that survives a demo and dies under a question.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler

from src.models.ordinal_cnn import (
    CLASS_NAMES,
    IMAGE_SIZE,
    REGRESSION_TARGETS,
    CalorieClassifier,
    ExtraLayerOrdinalMultiTaskCNN224,
    decode_ordinal,
    eval_transform,
)

BIG = 10.0  # a logit large enough that sigmoid saturates


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint in the exact format `Seeded Model.ipynb` writes — random weights."""
    out = tmp_path_factory.mktemp("ckpt")
    torch.manual_seed(0)
    model = ExtraLayerOrdinalMultiTaskCNN224()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "ExtraLayerOrdinalMultiTaskCNN224",
            "image_size": 224,
            "class_names": list(CLASS_NAMES),
            "ordinal_encoding": {"Low": [0, 0], "Medium": [1, 0], "High": [1, 1]},
            "regression_targets": list(REGRESSION_TARGETS),
            "test_accuracy": 0.7412731006160165,
            "best_validation_accuracy": 0.7761806981519507,
            "seed": 42,
        },
        out / "final_ordinal_multitask_model.pth",
    )
    scaler = StandardScaler()
    rng = np.random.default_rng(0)
    scaler.fit(
        np.column_stack(
            [
                rng.normal(250, 150, 500),
                rng.normal(200, 90, 500),
                rng.normal(12, 8, 500),
                rng.normal(20, 12, 500),
                rng.normal(18, 10, 500),
            ]
        )
    )
    joblib.dump(scaler, out / "regression_target_scaler.pkl")
    return out


def an_image(size: tuple[int, int] = (640, 480)) -> Image.Image:
    return Image.new("RGB", size, (190, 120, 70))


class TestArchitectureMatchesCheckpoint:
    """Layer names and shapes are load-bearing: `load_state_dict` matches on them. Renaming
    `shared_fc` to something tidier silently breaks weight loading."""

    def test_head_shapes(self) -> None:
        ordinal, regression = ExtraLayerOrdinalMultiTaskCNN224()(
            torch.zeros(2, 3, 224, 224)
        )
        assert ordinal.shape == (2, 2)
        assert regression.shape == (2, 5)

    def test_expected_layer_names_exist(self) -> None:
        keys = set(ExtraLayerOrdinalMultiTaskCNN224().state_dict())
        for expected in (
            "conv1.weight",
            "conv4.weight",
            "shared_fc.weight",
            "ordinal_head.weight",
            "regression_head.weight",
        ):
            assert expected in keys


class TestPreprocessingMatchesTraining:
    """The notebook's eval transform is Resize + ToTensor with NO Normalize. Adding
    normalization here — the reflex — feeds the net a distribution it never saw."""

    def test_output_is_correct_shape(self) -> None:
        assert eval_transform()(an_image()).shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_pixels_stay_in_unit_range(self) -> None:
        tensor = eval_transform()(an_image())
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_white_image_maps_to_one_not_a_normalized_value(self) -> None:
        # Under ImageNet normalization, white would land near +2.2, not 1.0. This pins the
        # absence of Normalize so nobody "helpfully" adds it later.
        tensor = eval_transform()(Image.new("RGB", (100, 100), (255, 255, 255)))
        assert torch.allclose(tensor.max(), torch.tensor(1.0))


class TestOrdinalDecoding:
    @pytest.mark.parametrize(
        ("logits", "expected_class"),
        [([-BIG, -BIG], 0), ([BIG, -BIG], 1), ([BIG, BIG], 2)],
    )
    def test_coherent_patterns_decode_to_documented_classes(
        self, logits: list[float], expected_class: int
    ) -> None:
        index, consistent = decode_ordinal(torch.tensor(logits))
        assert index == expected_class
        assert consistent is True

    def test_inconsistent_pattern_is_flagged(self) -> None:
        """[0, 1] = 'not above threshold 1, but above threshold 2' — incoherent for ordered
        classes. The notebook's sum() silently calls it Medium; we still return 1 to stay
        numerically identical to the reported accuracy, but we mark it."""
        index, consistent = decode_ordinal(torch.tensor([-BIG, BIG]))
        assert index == 1
        assert consistent is False


class TestLoading:
    def test_loads_packaged_checkpoint_and_metadata(self, checkpoint_dir: Path) -> None:
        clf = CalorieClassifier.load(
            checkpoint_dir / "final_ordinal_multitask_model.pth",
            checkpoint_dir / "regression_target_scaler.pkl",
        )
        assert clf.metadata["model_name"] == "ExtraLayerOrdinalMultiTaskCNN224"
        assert clf.metadata["test_accuracy"] == pytest.approx(0.7412731, abs=1e-6)
        assert clf.scaler is not None

    def test_missing_checkpoint_points_at_the_export_doc(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="EXPORT_WEIGHTS"):
            CalorieClassifier.load(tmp_path / "nope.pth")

    def test_bare_state_dict_is_rejected(self, tmp_path: Path) -> None:
        """A bare state_dict would load its tensors but carry no class names or accuracy, so the
        UI would render metadata that isn't there. Fail loudly instead."""
        path = tmp_path / "bare.pth"
        torch.save(ExtraLayerOrdinalMultiTaskCNN224().state_dict(), path)
        with pytest.raises(ValueError, match="model_state_dict"):
            CalorieClassifier.load(path)


class TestPrediction:
    def test_returns_a_real_class(self, checkpoint_dir: Path) -> None:
        clf = CalorieClassifier.load(
            checkpoint_dir / "final_ordinal_multitask_model.pth",
            checkpoint_dir / "regression_target_scaler.pkl",
        )
        prediction = clf.predict(an_image())
        assert prediction.class_name in CLASS_NAMES
        assert 0 <= prediction.class_index <= 2

    def test_threshold_probabilities_are_probabilities(
        self, checkpoint_dir: Path
    ) -> None:
        clf = CalorieClassifier.load(
            checkpoint_dir / "final_ordinal_multitask_model.pth"
        )
        for probability in clf.predict(an_image()).threshold_probabilities:
            assert 0.0 <= probability <= 1.0

    def test_nutrition_is_unscaled_into_plausible_units(
        self, checkpoint_dir: Path
    ) -> None:
        """Without inverse_transform the head's output is standardized — 'calories: -0.4'.
        Scaler fit around mean 250 kcal, so unscaled output must land in a sane range."""
        clf = CalorieClassifier.load(
            checkpoint_dir / "final_ordinal_multitask_model.pth",
            checkpoint_dir / "regression_target_scaler.pkl",
        )
        nutrition = clf.predict(an_image()).nutrition
        assert nutrition is not None
        assert set(nutrition) == set(REGRESSION_TARGETS)
        assert -500 < nutrition["calories"] < 1500

    def test_nutrition_suppressed_when_scaler_missing(
        self, checkpoint_dir: Path
    ) -> None:
        """Better to show nothing than to show standardized numbers labelled 'kcal'."""
        clf = CalorieClassifier.load(
            checkpoint_dir / "final_ordinal_multitask_model.pth"
        )
        assert clf.predict(an_image()).nutrition is None

    @pytest.mark.parametrize("size", [(64, 64), (1920, 1080), (300, 900)])
    def test_handles_arbitrary_input_sizes(
        self, checkpoint_dir: Path, size: tuple[int, int]
    ) -> None:
        clf = CalorieClassifier.load(
            checkpoint_dir / "final_ordinal_multitask_model.pth"
        )
        assert clf.predict(an_image(size)).class_name in CLASS_NAMES

    def test_grayscale_image_is_converted_not_crashed(
        self, checkpoint_dir: Path
    ) -> None:
        clf = CalorieClassifier.load(
            checkpoint_dir / "final_ordinal_multitask_model.pth"
        )
        assert clf.predict(Image.new("L", (200, 200), 128)).class_name in CLASS_NAMES

    def test_prediction_is_deterministic(self, checkpoint_dir: Path) -> None:
        """Dropout must be off. If eval() were forgotten, the same image would give different
        answers on refresh — which looks like magic and is actually a bug."""
        clf = CalorieClassifier.load(
            checkpoint_dir / "final_ordinal_multitask_model.pth"
        )
        image = an_image()
        first = clf.predict(image)
        second = clf.predict(image)
        assert first.class_index == second.class_index
        assert first.threshold_probabilities == second.threshold_probabilities
