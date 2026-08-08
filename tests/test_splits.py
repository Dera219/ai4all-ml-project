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


def synthetic_manifest(
    n_sessions: int = 60, per_session: int = 5, seed: int = 0
) -> pd.DataFrame:
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
            rows.append(
                {"dish_id": f"dish_{ts}", "calories": calories, "mass": calories * 1.5}
            )
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

    def test_dishes_within_a_session_are_close_in_time(
        self, manifest: pd.DataFrame
    ) -> None:
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
            list(splits.train.dish_id)
            + list(splits.val.dish_id)
            + list(splits.test.dish_id)
        )
        assert len(ids) == len(manifest)
        assert len(set(ids)) == len(manifest), "a dish appeared in more than one set"

    def test_reproducible(self, manifest: pd.DataFrame) -> None:
        a = grouped_splits(manifest, seed=7)
        b = grouped_splits(manifest, seed=7)
        assert set(a.test.dish_id) == set(b.test.dish_id)


class TestRandomSplitLeaks:
    """The control arm — this SHOULD leak. If it stops, the experiment lost its contrast."""

    def test_dish_id_overlap_is_zero_yet_sessions_leak(
        self, manifest: pd.DataFrame
    ) -> None:
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


class TestCommonTestDesign:
    """The rigorous experiment's setup: a held-out clean test shared across all arms, plus a
    repartitionable pool. These guarantees are what make that experiment's comparison valid."""

    def test_held_out_test_shares_no_session_with_pool(
        self, manifest: pd.DataFrame
    ) -> None:
        from src.data.nutrition5k import holdout_clean_test

        pool, test = holdout_clean_test(manifest, test_fraction=0.2, seed=0)
        assert not (set(pool.session) & set(test.session))
        assert len(pool) + len(test) == len(manifest)

    def test_leaky_train_val_shares_sessions(self, manifest: pd.DataFrame) -> None:
        from src.data.nutrition5k import holdout_clean_test, train_val_split

        pool, _ = holdout_clean_test(manifest, seed=0)
        train, val = train_val_split(pool, strategy="leaky", seed=0)
        # The whole point of the leaky arm: val overlaps train at the session level.
        assert set(train.session) & set(val.session)

    def test_clean_train_val_shares_no_sessions(self, manifest: pd.DataFrame) -> None:
        from src.data.nutrition5k import holdout_clean_test, train_val_split

        pool, _ = holdout_clean_test(manifest, seed=0)
        train, val = train_val_split(pool, strategy="clean", seed=0)
        assert not (set(train.session) & set(val.session))

    def test_both_strategies_draw_from_the_same_pool(
        self, manifest: pd.DataFrame
    ) -> None:
        """Only the partition differs — same dishes, so the split strategy is the sole variable."""
        from src.data.nutrition5k import holdout_clean_test, train_val_split

        pool, _ = holdout_clean_test(manifest, seed=0)
        leaky = train_val_split(pool, strategy="leaky", seed=0)
        clean = train_val_split(pool, strategy="clean", seed=0)
        leaky_ids = set(leaky[0].dish_id) | set(leaky[1].dish_id)
        clean_ids = set(clean[0].dish_id) | set(clean[1].dish_id)
        assert leaky_ids == clean_ids == set(pool.dish_id)

    def test_unknown_strategy_rejected(self, manifest: pd.DataFrame) -> None:
        from src.data.nutrition5k import holdout_clean_test, train_val_split

        pool, _ = holdout_clean_test(manifest, seed=0)
        with pytest.raises(ValueError, match="leaky.*clean"):
            train_val_split(pool, strategy="stratified", seed=0)


class TestManifestCleaning:
    def test_readable_check_catches_zero_byte_files(self, tmp_path) -> None:
        from src.data.nutrition5k import _is_readable

        empty = tmp_path / "empty.png"
        empty.touch()
        assert empty.exists()  # the trap: it exists
        assert not _is_readable(empty)  # but has no bytes

        real = tmp_path / "real.png"
        real.write_bytes(b"\x89PNG\r\n")
        assert _is_readable(real)


class TestOfficialStrategy:
    """The third arm: split by the dataset's own shipped files.

    These never touch the network. `official_split_ids` is monkeypatched where the assignment
    matters, and exercised against a pre-seeded cache where the loading itself is under test —
    a unit test that depends on a GCS bucket is a unit test that fails on a plane.
    """

    @staticmethod
    def _assign(pool: pd.DataFrame, val_share: float = 0.25) -> dict[str, set[str]]:
        """Fake an official assignment: the first `val_share` of dishes are 'official test'."""
        ids = list(pool.dish_id)
        cut = int(len(ids) * val_share)
        return {"test": set(ids[:cut]), "train": set(ids[cut:])}

    def test_val_is_exactly_the_official_test_side(
        self, manifest: pd.DataFrame, monkeypatch
    ) -> None:
        from src.data import nutrition5k as n5k

        pool, _ = n5k.holdout_clean_test(manifest, seed=0)
        assignment = self._assign(pool)
        monkeypatch.setattr(n5k, "official_split_ids", lambda **_: assignment)

        train, val = n5k.train_val_split(pool, strategy="official", seed=0)
        assert set(val.dish_id) == assignment["test"] & set(pool.dish_id)
        assert not (set(train.dish_id) & set(val.dish_id))

    def test_no_official_test_dish_is_moved_into_train(
        self, manifest: pd.DataFrame, monkeypatch
    ) -> None:
        """The boundary under test must never be crossed to hit a size target."""
        from src.data import nutrition5k as n5k

        pool, _ = n5k.holdout_clean_test(manifest, seed=0)
        assignment = self._assign(pool)
        monkeypatch.setattr(n5k, "official_split_ids", lambda **_: assignment)

        train, _ = n5k.train_val_split(pool, strategy="official", seed=0)
        assert not (set(train.dish_id) & assignment["test"])

    def test_draws_from_the_same_pool_as_the_other_arms(
        self, manifest: pd.DataFrame, monkeypatch
    ) -> None:
        from src.data import nutrition5k as n5k

        pool, _ = n5k.holdout_clean_test(manifest, seed=0)
        monkeypatch.setattr(n5k, "official_split_ids", lambda **_: self._assign(pool))

        train, val = n5k.train_val_split(pool, strategy="official", seed=0)
        assert set(train.dish_id) | set(val.dish_id) == set(pool.dish_id)

    def test_empty_side_is_an_error_not_a_silent_degenerate_split(
        self, manifest: pd.DataFrame, monkeypatch
    ) -> None:
        """If dish_ids don't match the official lists, fail loudly — a silently empty val
        would train fine and report a meaningless selection number."""
        from src.data import nutrition5k as n5k

        pool, _ = n5k.holdout_clean_test(manifest, seed=0)
        monkeypatch.setattr(
            n5k, "official_split_ids", lambda **_: {"test": set(), "train": set()}
        )
        with pytest.raises(ValueError, match="official split left one side empty"):
            n5k.train_val_split(pool, strategy="official", seed=0)

    def test_vendored_copy_resolves_without_network_or_cache(self) -> None:
        """The checked-in lists are the primary source — a Kaggle kernel has no internet."""
        from src.data.nutrition5k import official_split_ids

        out = official_split_ids(cache_dir="/nonexistent-on-purpose")
        assert len(out["train"]) == 4059
        assert len(out["test"]) == 709
        assert not (out["train"] & out["test"])

    def test_search_directories_are_consulted(self, tmp_path, monkeypatch) -> None:
        """Kaggle mounts the code dataset at an unpredictable path, so callers pass candidates."""
        from src.data import nutrition5k as n5k

        monkeypatch.setattr(n5k, "VENDORED_SPLITS", tmp_path / "absent")
        d = tmp_path / "mounted"
        d.mkdir()
        (d / "rgb_train_ids.txt").write_text("dish_1\ndish_2\n\n")
        (d / "rgb_test_ids.txt").write_text("dish_3\n")

        out = n5k.official_split_ids(cache_dir="/nonexistent-on-purpose", search=[d])
        assert out == {"train": {"dish_1", "dish_2"}, "test": {"dish_3"}}

    def test_falls_back_to_cache_when_nothing_vendored(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.data import nutrition5k as n5k

        monkeypatch.setattr(n5k, "VENDORED_SPLITS", tmp_path / "absent")
        cache = tmp_path / "official_splits"
        cache.mkdir()
        (cache / "rgb_train_ids.txt").write_text("dish_9\n")
        (cache / "rgb_test_ids.txt").write_text("dish_8\n")

        out = n5k.official_split_ids(cache_dir=tmp_path)
        assert out == {"train": {"dish_9"}, "test": {"dish_8"}}
