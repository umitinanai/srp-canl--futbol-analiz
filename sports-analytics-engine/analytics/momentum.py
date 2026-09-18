"""Momentum analytics: MI_rate, Z_MI and rolling momentum windows.

    MI_rate = delta_MI / delta_t
    Z_MI    = (MI_current - mean_MI) / std_MI

All public functions are pure and deterministic. Rolling functions
operate over caller-supplied bounded sequences (the caller -- e.g. a
MatchState's bounded deque -- is responsible for bounding history size;
this module does not retain any state between calls).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


class MomentumValidationError(ValueError):
    """Raised when an input to a momentum calculation is invalid."""


def _validate_finite_number(name: str, value: float) -> float:
    """Validate that value is a finite, non-NaN real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        MomentumValidationError: if value is not numeric, NaN, or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MomentumValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value):
        raise MomentumValidationError(f"{name} is NaN")
    if math.isinf(value):
        raise MomentumValidationError(f"{name} is infinite")
    return value


def calculate_momentum(mi_current: float, mi_previous: float) -> float:
    """Compute raw momentum: the change in MI between two observations.

    This is the un-normalized delta_MI (Section 11's "delta_MI"), as
    opposed to calculate_mi_rate() which additionally normalizes by
    elapsed time.

    Args:
        mi_current: the more recent momentum-indicator value.
        mi_previous: the earlier momentum-indicator value.

    Returns:
        mi_current - mi_previous.

    Raises:
        MomentumValidationError: if either input is invalid.
    """
    mi_current = _validate_finite_number("mi_current", mi_current)
    mi_previous = _validate_finite_number("mi_previous", mi_previous)
    return mi_current - mi_previous


def calculate_mi_rate(delta_mi: float, delta_t: float) -> float:
    """Compute MI_rate = delta_MI / delta_t.

    Args:
        delta_mi: change in the momentum indicator over the interval.
        delta_t: elapsed time over the interval. Must be strictly
            positive -- a rate over zero or negative elapsed time is
            undefined and is rejected explicitly rather than silently
            producing infinity or a sign-flipped rate.

    Returns:
        delta_mi / delta_t.

    Raises:
        MomentumValidationError: if either input is invalid, or if
            delta_t is not strictly positive.
    """
    delta_mi = _validate_finite_number("delta_mi", delta_mi)
    delta_t = _validate_finite_number("delta_t", delta_t)
    if delta_t <= 0:
        raise MomentumValidationError(
            f"delta_t must be strictly positive to compute a rate, got {delta_t}"
        )
    return delta_mi / delta_t


def calculate_mi_zscore(mi_current: float, mi_mean: float, mi_std: float) -> float:
    """Compute Z_MI = (mi_current - mi_mean) / mi_std.

    Args:
        mi_current: the current momentum-indicator (or MI_rate) value.
        mi_mean: the rolling mean of the momentum indicator.
        mi_std: the rolling standard deviation of the momentum
            indicator. May be zero.

    Returns:
        The z-score, or 0.0 if mi_std == 0 -- with zero dispersion, all
        observed values are identical to the mean, so a neutral (zero)
        z-score is the documented, safe fallback rather than a division
        by zero.

    Raises:
        MomentumValidationError: if any input is invalid, or if mi_std
            is negative (a standard deviation cannot be negative).
    """
    mi_current = _validate_finite_number("mi_current", mi_current)
    mi_mean = _validate_finite_number("mi_mean", mi_mean)
    mi_std = _validate_finite_number("mi_std", mi_std)
    if mi_std < 0:
        raise MomentumValidationError(f"mi_std must be non-negative, got {mi_std}")

    if mi_std == 0.0:
        return 0.0
    return (mi_current - mi_mean) / mi_std


@dataclass(frozen=True)
class RollingMomentumResult:
    """Vectorized rolling momentum output over a bounded observation window.

    Attributes:
        mi_rate: MI_rate for each consecutive pair of observations
            (length = len(mi_values) - 1). NaN where the corresponding
            delta_t was zero (protected division, see calculate_mi_rate
            for the strict single-pair policy; the rolling/batched path
            uses NaN instead of raising so that one bad interval does
            not invalidate the entire window).
        z_mi: rolling z-score of mi_rate at each position, using the
            trailing `window` values. NaN for positions with fewer than
            `window` preceding mi_rate observations (insufficient
            observations); 0.0 wherever the trailing window's standard
            deviation is exactly zero.
        window: the rolling window size used.
    """

    mi_rate: Tuple[float, ...]
    z_mi: Tuple[float, ...]
    window: int


def rolling_momentum(
    timestamps: Sequence[float],
    mi_values: Sequence[float],
    window: int = 5,
) -> RollingMomentumResult:
    """Compute MI_rate and a rolling Z_MI over a bounded observation history.

    Args:
        timestamps: strictly increasing observation timestamps.
        mi_values: momentum-indicator values aligned with timestamps
            (same length).
        window: number of trailing MI_rate observations used for each
            rolling z-score. Must be a positive integer.

    Returns:
        A RollingMomentumResult. See RollingMomentumResult for the NaN
        semantics used to represent zero-delta-t and insufficient
        observation conditions.

    Raises:
        MomentumValidationError: if inputs are misshapen, too short,
            non-finite, window is invalid, or timestamps are not
            strictly increasing.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise MomentumValidationError(f"window must be a positive int, got {window!r}")

    ts = np.asarray(timestamps, dtype=np.float64)
    mi = np.asarray(mi_values, dtype=np.float64)

    if ts.ndim != 1 or mi.ndim != 1:
        raise MomentumValidationError("timestamps and mi_values must be 1D sequences")
    if ts.shape[0] != mi.shape[0]:
        raise MomentumValidationError("timestamps and mi_values must have equal length")
    if ts.shape[0] < 2:
        raise MomentumValidationError(
            "at least 2 observations are required to compute momentum "
            "(insufficient observations)"
        )
    if not (np.all(np.isfinite(ts)) and np.all(np.isfinite(mi))):
        raise MomentumValidationError("timestamps and mi_values must be finite (no NaN/Inf)")
    if np.any(np.diff(ts) <= 0):
        raise MomentumValidationError("timestamps must be strictly increasing")

    delta_mi = np.diff(mi)
    delta_t = np.diff(ts)

    mi_rate = np.divide(
        delta_mi,
        delta_t,
        out=np.full_like(delta_mi, np.nan),
        where=delta_t != 0,
    )

    n = mi_rate.shape[0]
    z_mi = np.full(n, np.nan, dtype=np.float64)

    if n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(mi_rate, window)
        means = windows.mean(axis=1)
        stds = windows.std(axis=1)
        tail_values = mi_rate[window - 1 :]
        safe_stds = np.where(stds == 0.0, 1.0, stds)
        tail_z = np.where(stds == 0.0, 0.0, (tail_values - means) / safe_stds)
        z_mi[window - 1 :] = tail_z

    return RollingMomentumResult(
        mi_rate=tuple(mi_rate.tolist()),
        z_mi=tuple(z_mi.tolist()),
        window=window,
    )
