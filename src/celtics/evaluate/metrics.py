"""Probabilistic forecast metrics."""

import numpy as np

EPS = 1e-15


def _arrays(p, y):
    return np.asarray(p, dtype=float), np.asarray(y, dtype=float)


def log_loss(p, y) -> float:
    p, y = _arrays(p, y)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    p, y = _arrays(p, y)
    return float(np.mean((p - y) ** 2))


def accuracy(p, y) -> float:
    p, y = _arrays(p, y)
    return float(np.mean((p > 0.5) == (y == 1)))


def calibration_table(p, y, n_bins: int = 10) -> list[dict]:
    p, y = _arrays(p, y)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "mean_pred": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return rows


def summarize(p, y) -> dict:
    return {
        "n": int(len(np.asarray(p))),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "accuracy": accuracy(p, y),
    }
