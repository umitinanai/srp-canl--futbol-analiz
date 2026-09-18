"""Calibration and accuracy metrics for backtested probability predictions.

Standard, well-established forecast-evaluation metrics only (Brier
score, log loss, expected calibration error) -- no invented or
project-specific scoring rules.
"""

from __future__ import annotations

import math
from typing import Sequence


class BacktestMetricsError(ValueError):
    """Raised when inputs to a backtest metric calculation are invalid."""


def _validate_predictions_and_outcomes(
    predicted_probabilities: Sequence[float], outcomes: Sequence[int]
) -> None:
    """Validate aligned prediction/outcome sequences for metric calculations.

    Args:
        predicted_probabilities: predicted probabilities, each expected
            in [0, 1].
        outcomes: binary outcomes (0 or 1), aligned with
            predicted_probabilities.

    Raises:
        BacktestMetricsError: if the sequences are misshapen, empty, or
            contain out-of-domain values.
    """
    if len(predicted_probabilities) != len(outcomes):
        raise BacktestMetricsError(
            "predicted_probabilities and outcomes must have equal length"
        )
    if len(predicted_probabilities) == 0:
        raise BacktestMetricsError("predicted_probabilities must not be empty")
    for p in predicted_probabilities:
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise BacktestMetricsError(f"predicted probability must be numeric, got {type(p)!r}")
        if math.isnan(p) or math.isinf(p):
            raise BacktestMetricsError(f"predicted probability must be finite, got {p}")
        if not (0.0 <= p <= 1.0):
            raise BacktestMetricsError(f"predicted probability must be within [0, 1], got {p}")
    for o in outcomes:
        if o not in (0, 1):
            raise BacktestMetricsError(f"outcome must be 0 or 1, got {o!r}")


def brier_score(predicted_probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Compute the Brier score: mean squared error between predictions and outcomes.

    Args:
        predicted_probabilities: predicted probabilities, each in [0, 1].
        outcomes: binary outcomes (0 or 1), aligned with
            predicted_probabilities.

    Returns:
        The Brier score, in [0, 1] (0 = perfect, 1 = worst possible).

    Raises:
        BacktestMetricsError: if inputs are invalid.
    """
    _validate_predictions_and_outcomes(predicted_probabilities, outcomes)
    n = len(predicted_probabilities)
    total = sum((p - o) ** 2 for p, o in zip(predicted_probabilities, outcomes))
    return total / n


def log_loss(
    predicted_probabilities: Sequence[float],
    outcomes: Sequence[int],
    epsilon: float = 1e-12,
) -> float:
    """Compute binary log loss (cross-entropy), with clipping to avoid log(0).

    Args:
        predicted_probabilities: predicted probabilities, each in [0, 1].
        outcomes: binary outcomes (0 or 1), aligned with
            predicted_probabilities.
        epsilon: clipping floor/ceiling applied to predictions before
            taking logarithms, to avoid -inf for p=0 or p=1 predictions.

    Returns:
        The mean log loss (>= 0; lower is better).

    Raises:
        BacktestMetricsError: if inputs are invalid, or epsilon is not
            a small strictly-positive number less than 0.5.
    """
    _validate_predictions_and_outcomes(predicted_probabilities, outcomes)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise BacktestMetricsError(f"epsilon must be numeric, got {type(epsilon)!r}")
    if not (0.0 < epsilon < 0.5):
        raise BacktestMetricsError(f"epsilon must be within (0, 0.5), got {epsilon}")

    n = len(predicted_probabilities)
    total = 0.0
    for p, o in zip(predicted_probabilities, outcomes):
        clipped = min(max(p, epsilon), 1.0 - epsilon)
        total += -(o * math.log(clipped) + (1 - o) * math.log(1.0 - clipped))
    return total / n


def calibration_error(
    predicted_probabilities: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> float:
    """Compute the expected calibration error (ECE) over equal-width probability bins.

    For each bin, the absolute difference between the mean predicted
    probability and the observed outcome frequency is computed, then
    averaged across bins weighted by bin occupancy.

    Args:
        predicted_probabilities: predicted probabilities, each in [0, 1].
        outcomes: binary outcomes (0 or 1), aligned with
            predicted_probabilities.
        n_bins: number of equal-width bins over [0, 1]. Must be a
            positive integer.

    Returns:
        The expected calibration error, in [0, 1] (0 = perfectly calibrated).

    Raises:
        BacktestMetricsError: if inputs are invalid, or n_bins is not a
            positive integer.
    """
    _validate_predictions_and_outcomes(predicted_probabilities, outcomes)
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise BacktestMetricsError(f"n_bins must be a positive int, got {n_bins!r}")

    n = len(predicted_probabilities)
    bin_sums_pred = [0.0] * n_bins
    bin_sums_outcome = [0.0] * n_bins
    bin_counts = [0] * n_bins

    for p, o in zip(predicted_probabilities, outcomes):
        bin_index = min(int(p * n_bins), n_bins - 1)
        bin_sums_pred[bin_index] += p
        bin_sums_outcome[bin_index] += o
        bin_counts[bin_index] += 1

    weighted_error = 0.0
    for count, sum_pred, sum_outcome in zip(bin_counts, bin_sums_pred, bin_sums_outcome):
        if count == 0:
            continue
        mean_pred = sum_pred / count
        mean_outcome = sum_outcome / count
        weighted_error += (count / n) * abs(mean_pred - mean_outcome)

    return weighted_error
