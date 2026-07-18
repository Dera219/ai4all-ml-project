"""Tests for the evaluation metrics — the math must be right before it goes on a slide.

Hand-computed expected values on tiny fixtures, so a wrong formula (a swapped precision/recall, a
specificity that forgets the true negatives) fails here rather than in the presentation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation import metrics as M


class TestConfusionMatrix:
    def test_counts_land_in_the_right_cells(self) -> None:
        y_true = np.array([0, 0, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 0])
        cm = M.confusion_matrix(y_true, y_pred)
        # row 0 (true Low): one predicted Low, one predicted Medium
        assert cm[0, 0] == 1 and cm[0, 1] == 1
        assert cm[1, 1] == 1  # true Medium predicted Medium
        assert cm[2, 2] == 1 and cm[2, 0] == 1  # true High: one right, one called Low
        assert cm.sum() == 5

    def test_perfect_predictions_are_diagonal(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        cm = M.confusion_matrix(y, y)
        assert np.array_equal(cm, np.diag([2, 2, 2]))
        assert M.accuracy(cm) == 1.0


class TestPerClassMetrics:
    def test_hand_computed_values(self) -> None:
        # Confusion matrix (rows=true, cols=pred), Low/Med/High:
        #        pred L  M  H
        # true L    8   2  0
        # true M    1   7  2
        # true H    0   1  9
        cm = np.array([[8, 2, 0], [1, 7, 2], [0, 1, 9]])
        m = {x.name: x for x in M.per_class_metrics(cm)}

        # Low: TP=8, FP=1 (the Med→Low), FN=2 → precision 8/9, recall 8/10
        assert m["Low"].precision == pytest.approx(8 / 9, abs=1e-4)
        assert m["Low"].recall == pytest.approx(0.8, abs=1e-4)
        # Low specificity = TN / (TN + FP). Total=30, true-Low=10, so negatives=20.
        # FP=1 (the Med→Low), so TN=19 → specificity 19/20.
        assert m["Low"].specificity == pytest.approx(19 / 20, abs=1e-4)

        # High: TP=9, FP=2 (the Med→High), FN=1 → precision 9/11, recall 9/10
        assert m["High"].precision == pytest.approx(9 / 11, abs=1e-4)
        assert m["High"].recall == pytest.approx(0.9, abs=1e-4)

    def test_f1_is_harmonic_mean(self) -> None:
        cm = np.array([[8, 2, 0], [1, 7, 2], [0, 1, 9]])
        low = M.per_class_metrics(cm)[0]
        expected = 2 * low.precision * low.recall / (low.precision + low.recall)
        assert low.f1 == pytest.approx(expected, abs=1e-6)

    def test_support_sums_to_total(self) -> None:
        cm = np.array([[8, 2, 0], [1, 7, 2], [0, 1, 9]])
        assert sum(m.support for m in M.per_class_metrics(cm)) == cm.sum()


class TestOrdinalProbabilities:
    def test_confident_thresholds_give_a_valid_distribution(self) -> None:
        big = 10.0
        logits = torch.tensor([
            [-big, -big],  # both thresholds off → Low
            [big, -big],   # above t1, below t2 → Medium
            [big, big],    # both on → High
        ])
        probs = M.ordinal_class_probabilities(logits)
        assert np.allclose(probs.sum(axis=1), 1.0)  # valid distributions
        assert probs[0].argmax() == 0  # Low
        assert probs[1].argmax() == 1  # Medium
        assert probs[2].argmax() == 2  # High

    def test_incoherent_thresholds_are_clipped_not_negative(self) -> None:
        # t2 > t1 would make P(Medium) negative without the clip.
        logits = torch.tensor([[-5.0, 5.0]])
        probs = M.ordinal_class_probabilities(logits)
        assert (probs >= 0).all()
        assert np.isclose(probs.sum(), 1.0)


class TestOffByTwo:
    def test_only_low_high_confusions_count(self) -> None:
        y_true = np.array([0, 0, 1, 2])
        y_pred = np.array([2, 1, 2, 0])  # L→H (dist 2), L→M (1), M→H (1), H→L (dist 2)
        assert M.off_by_two_rate(y_true, y_pred) == pytest.approx(0.5)  # 2 of 4

    def test_perfect_predictions_have_zero(self) -> None:
        y = np.array([0, 1, 2, 1, 0])
        assert M.off_by_two_rate(y, y) == 0.0


class TestROC:
    def test_auc_is_one_for_perfectly_separated_scores(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        # Probabilities that perfectly rank each class highest for its members.
        probs = np.array([
            [0.9, 0.05, 0.05], [0.8, 0.1, 0.1],   # Low
            [0.1, 0.85, 0.05], [0.05, 0.9, 0.05], # Medium
            [0.05, 0.05, 0.9], [0.1, 0.1, 0.8],   # High
        ])
        roc = M.roc_one_vs_rest(y_true, probs)
        for name in M.CLASS_NAMES:
            assert roc[name]["auc"] == pytest.approx(1.0)


class TestMarkdownTable:
    def test_renders_every_class_and_the_summary(self) -> None:
        cm = np.array([[8, 2, 0], [1, 7, 2], [0, 1, 9]])
        table = M.metrics_table_markdown(cm, M.per_class_metrics(cm), roc={})
        for name in M.CLASS_NAMES:
            assert name in table
        assert "Accuracy" in table and "Macro-F1" in table
