import math

import numpy as np
import pytest

from celtics.evaluate.metrics import accuracy, brier, calibration_table, log_loss, summarize


def test_coin_flip_scores():
    p = np.full(4, 0.5)
    y = np.array([1, 0, 1, 0])
    assert log_loss(p, y) == pytest.approx(math.log(2))
    assert brier(p, y) == pytest.approx(0.25)


def test_accuracy_threshold():
    assert accuracy(np.array([0.9, 0.2, 0.6]), np.array([1, 0, 0])) == pytest.approx(2 / 3)


def test_log_loss_clips_extremes():
    assert np.isfinite(log_loss(np.array([1.0, 0.0]), np.array([0, 1])))


def test_calibration_table_bins():
    p = np.array([0.05, 0.07, 0.95, 0.96])
    y = np.array([0, 1, 1, 1])
    table = calibration_table(p, y, n_bins=10)
    assert [row["n"] for row in table] == [2, 2]
    assert table[0]["observed"] == pytest.approx(0.5)
    assert table[1]["mean_pred"] == pytest.approx(0.955)


def test_summarize_keys():
    s = summarize(np.array([0.7, 0.3]), np.array([1, 0]))
    assert set(s) == {"n", "log_loss", "brier", "accuracy"} and s["n"] == 2
