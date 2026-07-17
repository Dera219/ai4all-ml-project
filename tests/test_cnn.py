"""Model tests. No trained weights needed — these check shapes and the ordinal decode logic."""

from __future__ import annotations

import pytest
import torch

from src.models.cnn import (
    CalorieCNN,
    GroupStyleCNN,
    MajorityBaseline,
    decode_ordinal,
    ordinal_targets,
)


class TestArchitecture:
    @pytest.mark.parametrize("channels", [3, 4])
    def test_forward_shapes(self, channels: int) -> None:
        model = CalorieCNN(in_channels=channels)
        ordinal, regression = model(torch.zeros(2, channels, 224, 224))
        assert ordinal.shape == (2, 2)
        assert regression.shape == (2,)

    def test_global_pool_makes_it_size_agnostic(self) -> None:
        """AdaptiveAvgPool2d means a different input size still produces valid output — the whole
        reason for using it over a fixed flatten->FC."""
        model = CalorieCNN(in_channels=4)
        ordinal, _ = model(torch.zeros(1, 4, 256, 256))
        assert ordinal.shape == (1, 2)

    def test_far_smaller_than_a_flatten_fc_trunk(self) -> None:
        """The design claim: GAP keeps the model small. The group's flatten->FC(25088,128) is
        3.2M params on its own; the whole model here should be well under that."""
        assert CalorieCNN(in_channels=4).n_parameters < 1_500_000


class TestGroupStyleModel:
    """The high-capacity comparison arm. It must (a) run at the experiment's size, (b) share the
    same (ordinal, regression) interface as CalorieCNN so one trainer drives both, and (c) be
    genuinely bigger — the whole reason it's in the experiment."""

    def test_same_output_interface_as_mine(self) -> None:
        model = GroupStyleCNN(in_channels=3)
        # LazyLinear materializes on first forward.
        ordinal, regression = model(torch.zeros(2, 3, 160, 160))
        assert ordinal.shape == (2, 2)
        assert regression.shape == (2,)

    def test_has_a_giant_flatten_fc(self) -> None:
        """After materializing, the flatten->FC must dominate — that's the capacity that lets it
        memorize, which is what the leakage experiment probes."""
        model = GroupStyleCNN(in_channels=3)
        model(torch.zeros(1, 3, 160, 160))  # materialize
        assert model.n_parameters > CalorieCNN(in_channels=3).n_parameters
        # The shared_fc layer alone should be the bulk of it.
        fc_params = sum(p.numel() for p in model.shared_fc.parameters())
        assert fc_params > 0.5 * model.n_parameters

    def test_decodes_with_the_same_ordinal_logic(self) -> None:
        model = GroupStyleCNN(in_channels=3)
        ordinal, _ = model(torch.zeros(1, 3, 160, 160))
        cls, coherent = decode_ordinal(ordinal)
        assert cls.shape == (1,)
        assert coherent.shape == (1,)


class TestOrdinalDecode:
    @pytest.mark.parametrize(
        ("logits", "expected_class", "coherent"),
        [
            ([-9.0, -9.0], 0, True),  # [0,0] Low
            ([9.0, -9.0], 1, True),  # [1,0] Medium
            ([9.0, 9.0], 2, True),  # [1,1] High
            ([-9.0, 9.0], 0, False),  # [0,1] incoherent -> first failure wins -> 0
        ],
    )
    def test_decode(
        self, logits: list[float], expected_class: int, coherent: bool
    ) -> None:
        cls, coh = decode_ordinal(torch.tensor([logits]))
        assert int(cls[0]) == expected_class
        assert bool(coh[0]) is coherent

    def test_incoherent_decodes_low_not_medium(self) -> None:
        """The departure from the group's decode: they sum passed thresholds, so [0,1] -> 1
        (Medium). Walking to the first failure gives 0 (Low) and flags it — because 'above the
        second threshold but not the first' is not evidence for the middle class."""
        cls, coherent = decode_ordinal(torch.tensor([[-5.0, 5.0]]))
        assert int(cls[0]) == 0
        assert not bool(coherent[0])

    def test_targets_round_trip(self) -> None:
        labels = torch.tensor([0, 1, 2])
        targets = ordinal_targets(labels)
        assert torch.equal(targets, torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
        # A confident model on those exact targets should decode back to the labels.
        logits = (targets - 0.5) * 20
        decoded, _ = decode_ordinal(logits)
        assert torch.equal(decoded, labels)


class TestBaseline:
    def test_predicts_the_majority_class(self) -> None:
        labels = torch.tensor([0, 0, 0, 1, 2])
        baseline = MajorityBaseline().fit(labels)
        assert baseline.majority == 0
        assert torch.equal(baseline.predict(3), torch.zeros(3, dtype=torch.long))

    def test_scores_near_chance_on_balanced_data(self) -> None:
        labels = torch.tensor([0, 1, 2] * 100)
        baseline = MajorityBaseline().fit(labels)
        accuracy = (baseline.predict(len(labels)) == labels).float().mean()
        assert abs(float(accuracy) - 1 / 3) < 0.05
