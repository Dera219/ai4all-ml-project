"""Tests for the split logic — the heart of the leakage contribution.

The claim being made is methodological: a random split on dish_id leaks, a grouped split on
capture-session does not. If these tests don't hold, the claim is wrong, so they matter more than
any accuracy number the model produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.nutrition5k import (
    add_calorie_classes,
    grouped_splits,
    random_splits,
    session_leakage_report,
    sessions_for,
)


def synthetic_manifest(n_sessions: int = 60, per_session: int = 5, seed: int = 0) -> pd.DataFrame:
    """Dishes in capture sessions, with the dish_id = dish_<timestamp> structure of the real data.

    Within a session, timestamps are ~40s apart; between sessions, hours apart. Calories are
    session-correlated (a real property of the data: dishes plated together share ingredients),
    which is exactly what makes the leak matter.
    """
    rng = np.random.default_rng(seed)
    rows = []
    ts = 1_556_000_000
    for s in range(n_sessions):
        session_mean = rng.uniform(50, 600)
        ts += rng.integers(3600, 36000)  # hours between sessions
        for _ in range(per_session):
            ts += rng.integers(20, 90)  # seconds within a session
            calories = max(1.0, rng.normal(session_mean, 30))
            rows.append({"dish_id": f"dish_{ts}", "calories": calories, "mass": calories * 1.5})
    return pd.DataFrame(rows)


@pytest.fixture
def manifest() -> pd.DataFrame:
    df = synthetic_manifest()
    df = sessions_for(df)
    df, _ = add_calorie_classes(df)
    return df


class TestSessionDerivation:
    def test_sessions_recovered_from_timestamps(self, manifest: pd.DataFrame) -> None:
        # 60 sessions were generated; the deriver should find about that many.
        assert 55 <= manifest.session.nunique() <= 65

    def test_dishes_within_a_session_are_close_in_time(self, manifest: pd.DataFrame) -> None:
        for _, group in manifest.groupby("session"):
            if len(group) > 1:
                span = group.ts.max() - group.ts.min()
                assert span < 120 * len(group)

    def test_gap_threshold_changes_session_count(self) -> None:
        df = synthetic_manifest()
        few = sessions_for(df, gap_seconds=10).session.nunique()
        many = sessions_for(df, gap_seconds=10_000).session.nunique()
        # A tiny gap splits everything apart; a huge gap merges into one. The parameter matters,
        # which is why it's exposed rather than hardcoded.
        assert few > many


class TestGroupedSplitIsClean:
    def test_no_session_spans_two_sets(self, manifest: pd.DataFrame) -> None:
        splits = grouped_splits(manifest, seed=1)
        train_s, val_s, test_s = (
            set(splits.train.session),
            set(splits.val.session),
            set(splits.test.session),
        )
        assert not (train_s & val_s)
        assert not (train_s & test_s)
        assert not (val_s & test_s)

    def test_zero_session_contamination(self, manifest: pd.DataFrame) -> None:
        report = session_leakage_report(grouped_splits(manifest, seed=1))
        assert report["contaminated_fraction"] == 0.0
        assert report["shared_sessions"] == 0

    def test_every_dish_lands_somewhere_once(self, manifest: pd.DataFrame) -> None:
        splits = grouped_splits(manifest, seed=1)
        ids = (
            list(splits.train.dish_id) + list(splits.val.dish_id) + list(splits.test.dish_id)
        )
        assert len(ids) == len(manifest)
        assert len(set(ids)) == len(manifest), "a dish appeared in more than one set"

    def test_reproducible(self, manifest: pd.DataFrame) -> None:
        a = grouped_splits(manifest, seed=7)
        b = grouped_splits(manifest, seed=7)
        assert set(a.test.dish_id) == set(b.test.dish_id)


class TestRandomSplitLeaks:
    """The control arm — this SHOULD leak. If it stops, the experiment lost its contrast."""

    def test_dish_id_overlap_is_zero_yet_sessions_leak(self, manifest: pd.DataFrame) -> None:
        splits = random_splits(manifest, seed=1)

        # The check the group ran — passes.
        assert not (set(splits.train.dish_id) & set(splits.test.dish_id))

        # The check they didn't — fails badly, which is the whole point.
        report = session_leakage_report(splits)
        assert report["contaminated_fraction"] > 0.5, (
            "the random split should leak most of its test set via shared sessions; "
            "if it doesn't, the synthetic data no longer resembles Nutrition5k"
        )


class TestClassEdges:
    def test_edges_from_train_applied_to_test(self, manifest: pd.DataFrame) -> None:
        """Val/test must be binned with TRAIN's edges, or the test distribution leaks into the
        class boundaries."""
        splits = grouped_splits(manifest, seed=1)
        train_frame, edges = add_calorie_classes(splits.train)
        test_frame, edges_back = add_calorie_classes(splits.test, edges=edges)
        assert np.array_equal(edges, edges_back)
        assert set(test_frame.label.unique()).issubset({0, 1, 2})

    def test_outer_edges_are_infinite(self, manifest: pd.DataFrame) -> None:
        """An unusually large dish in test must still bin, not become NaN."""
        _, edges = add_calorie_classes(manifest)
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf


class TestManifestCleaning:
    def test_readable_check_catches_zero_byte_files(self, tmp_path) -> None:
        from src.data.nutrition5k import _is_readable

        empty = tmp_path / "empty.png"
        empty.touch()
        assert empty.exists()          # the trap: it exists
        assert not _is_readable(empty)  # but has no bytes

        real = tmp_path / "real.png"
        real.write_bytes(b"\x89PNG\r\n")
        assert _is_readable(real)
