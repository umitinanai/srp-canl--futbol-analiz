"""xG proxy and shot-quality analytics.

IMPORTANT: this is an analytical xG PROXY built from a simple,
transparent, configurable linear combination of shot-volume statistics
-- it is explicitly NOT a professional provider-grade expected-goals
model. The exact coefficients (DEFAULT_XG_PROXY_WEIGHTS) are a Stage 2
analytics-layer assumption, isolated here as named, overridable
constants rather than buried magic numbers, since the source
specification does not define an exact xG coefficient set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

#: Default linear-combination weights for the xG proxy. ASSUMPTION:
#: not defined by the source specification; isolated here, documented,
#: and overridable via the `weights` parameter. Weights sum to 1.0.
DEFAULT_XG_PROXY_WEIGHTS: Mapping[str, float] = {
    "shots_on_target": 0.6,
    "dangerous_attacks": 0.4,
}

#: Default epsilon used to guard divisions by (potentially zero) shot counts.
DEFAULT_EPSILON = 1e-6


class XGProxyValidationError(ValueError):
    """Raised when an input to an xG proxy calculation is invalid."""


def _validate_finite_number(name: str, value: float) -> float:
    """Validate that value is a finite, non-NaN real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        XGProxyValidationError: if value is not numeric, NaN, or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise XGProxyValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value):
        raise XGProxyValidationError(f"{name} is NaN")
    if math.isinf(value):
        raise XGProxyValidationError(f"{name} is infinite")
    return value


def _validate_non_negative(name: str, value: float) -> float:
    """Validate that value is a finite, non-negative real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        XGProxyValidationError: if value is invalid or negative.
    """
    value = _validate_finite_number(name, value)
    if value < 0:
        raise XGProxyValidationError(f"{name} must be non-negative, got {value}")
    return value


def _validate_epsilon(epsilon: float) -> float:
    """Validate an epsilon guard value: finite and strictly positive.

    Args:
        epsilon: candidate epsilon.

    Returns:
        epsilon, coerced to float.

    Raises:
        XGProxyValidationError: if epsilon is invalid or non-positive.
    """
    epsilon = _validate_finite_number("epsilon", epsilon)
    if epsilon <= 0:
        raise XGProxyValidationError(f"epsilon must be strictly positive, got {epsilon}")
    return epsilon


def _validate_weights(weights: Mapping[str, float]) -> Mapping[str, float]:
    """Validate the xG proxy weight map.

    Args:
        weights: candidate weight mapping. Must contain exactly the
            keys "shots_on_target" and "dangerous_attacks", summing to
            ~1.0.

    Returns:
        weights, unchanged.

    Raises:
        XGProxyValidationError: if the key set or sum is invalid.
    """
    expected_keys = frozenset(DEFAULT_XG_PROXY_WEIGHTS.keys())
    actual_keys = frozenset(weights.keys())
    if actual_keys != expected_keys:
        raise XGProxyValidationError(
            f"weights keys must exactly equal {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )
    total = 0.0
    for key, value in weights.items():
        value = _validate_finite_number(f"weights[{key!r}]", value)
        total += value
    if abs(total - 1.0) > 1e-6:
        raise XGProxyValidationError(f"weights must sum to 1.0, got {total}")
    return weights


def calculate_xg_proxy(
    shots_on_target: float,
    dangerous_attacks: float,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Compute a transparent xG proxy from shot-volume statistics.

    xg_proxy = weights["shots_on_target"] * shots_on_target
             + weights["dangerous_attacks"] * dangerous_attacks

    Args:
        shots_on_target: count of shots on target (non-negative, finite).
        dangerous_attacks: count of dangerous attacks (non-negative,
            finite).
        weights: optional override for DEFAULT_XG_PROXY_WEIGHTS.

    Returns:
        The xG proxy value (non-negative).

    Raises:
        XGProxyValidationError: if any input is invalid.
    """
    shots_on_target = _validate_non_negative("shots_on_target", shots_on_target)
    dangerous_attacks = _validate_non_negative("dangerous_attacks", dangerous_attacks)
    weights = _validate_weights(weights if weights is not None else DEFAULT_XG_PROXY_WEIGHTS)

    return (
        weights["shots_on_target"] * shots_on_target
        + weights["dangerous_attacks"] * dangerous_attacks
    )


def calculate_sqp(xg_proxy: float, shots: float, epsilon: float = DEFAULT_EPSILON) -> float:
    """Compute the Shot Quality Proxy: SQP = xg_proxy / max(shots, epsilon).

    Args:
        xg_proxy: the xG proxy value (non-negative, finite).
        shots: total shot count (non-negative, finite). May be zero;
            division by zero is prevented via epsilon.
        epsilon: strictly positive floor applied to the shots
            denominator.

    Returns:
        The shot quality proxy value.

    Raises:
        XGProxyValidationError: if any input is invalid.
    """
    xg_proxy = _validate_non_negative("xg_proxy", xg_proxy)
    shots = _validate_non_negative("shots", shots)
    epsilon = _validate_epsilon(epsilon)
    return xg_proxy / max(shots, epsilon)


def calculate_shot_quality_ratio(
    shots_on_target: float, shots: float, epsilon: float = DEFAULT_EPSILON
) -> float:
    """Compute the ratio of shots on target to total shots.

    Args:
        shots_on_target: count of shots on target (non-negative, finite).
        shots: total shot count (non-negative, finite).
        epsilon: strictly positive floor applied to the shots
            denominator.

    Returns:
        shots_on_target / max(shots, epsilon).

    Raises:
        XGProxyValidationError: if any input is invalid, or if
            shots_on_target exceeds shots (a logically impossible stat
            combination).
    """
    shots_on_target = _validate_non_negative("shots_on_target", shots_on_target)
    shots = _validate_non_negative("shots", shots)
    epsilon = _validate_epsilon(epsilon)
    if shots_on_target > shots:
        raise XGProxyValidationError(
            f"shots_on_target ({shots_on_target}) cannot exceed shots ({shots})"
        )
    return shots_on_target / max(shots, epsilon)


@dataclass(frozen=True)
class RollingXGProxyResult:
    """Vectorized rolling xG-proxy output over a bounded observation window.

    Attributes:
        xg_proxy: xG proxy value at each observation.
        sqp: shot quality proxy at each observation.
        shot_quality_ratio: shots-on-target ratio at each observation.
        shot_quality_trend: instantaneous rate of change of sqp with
            respect to time, computed via a central-difference gradient
            (np.gradient). Represents whether shot quality is trending
            up or down.
    """

    xg_proxy: Tuple[float, ...]
    sqp: Tuple[float, ...]
    shot_quality_ratio: Tuple[float, ...]
    shot_quality_trend: Tuple[float, ...]


def rolling_xg_proxy(
    shots_on_target_series: Sequence[float],
    dangerous_attacks_series: Sequence[float],
    shots_series: Sequence[float],
    timestamps: Sequence[float],
    weights: Optional[Mapping[str, float]] = None,
    epsilon: float = DEFAULT_EPSILON,
) -> RollingXGProxyResult:
    """Compute xG proxy, SQP, shot-quality ratio and trend over an observation history.

    Args:
        shots_on_target_series: shots-on-target count per observation.
        dangerous_attacks_series: dangerous-attacks count per observation.
        shots_series: total shots count per observation.
        timestamps: strictly increasing observation timestamps, aligned
            with the series above (all same length).
        weights: optional override for DEFAULT_XG_PROXY_WEIGHTS.
        epsilon: strictly positive floor applied to shot-count
            denominators.

    Returns:
        A RollingXGProxyResult with one entry per observation for each
        field.

    Raises:
        XGProxyValidationError: if inputs are misshapen, too short,
            invalid, or timestamps are not strictly increasing.
    """
    weights = _validate_weights(weights if weights is not None else DEFAULT_XG_PROXY_WEIGHTS)
    epsilon = _validate_epsilon(epsilon)

    sot = np.asarray(shots_on_target_series, dtype=np.float64)
    das = np.asarray(dangerous_attacks_series, dtype=np.float64)
    shots = np.asarray(shots_series, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)

    lengths = {sot.shape[0], das.shape[0], shots.shape[0], ts.shape[0]}
    if len(lengths) != 1:
        raise XGProxyValidationError("all input series must have equal length")
    if ts.shape[0] < 1:
        raise XGProxyValidationError("at least 1 observation is required")
    if not (
        np.all(np.isfinite(sot))
        and np.all(np.isfinite(das))
        and np.all(np.isfinite(shots))
        and np.all(np.isfinite(ts))
    ):
        raise XGProxyValidationError("all inputs must be finite (no NaN/Inf)")
    if np.any(sot < 0) or np.any(das < 0) or np.any(shots < 0):
        raise XGProxyValidationError("shot-related counts must be non-negative")
    if np.any(sot > shots):
        raise XGProxyValidationError("shots_on_target cannot exceed shots at any observation")
    if ts.shape[0] >= 2 and np.any(np.diff(ts) <= 0):
        raise XGProxyValidationError("timestamps must be strictly increasing")

    xg_proxy = weights["shots_on_target"] * sot + weights["dangerous_attacks"] * das

    safe_shots = np.maximum(shots, epsilon)
    sqp = xg_proxy / safe_shots
    shot_quality_ratio = sot / safe_shots

    if ts.shape[0] >= 2:
        shot_quality_trend = np.gradient(sqp, ts)
    else:
        shot_quality_trend = np.zeros_like(sqp)

    return RollingXGProxyResult(
        xg_proxy=tuple(xg_proxy.tolist()),
        sqp=tuple(sqp.tolist()),
        shot_quality_ratio=tuple(shot_quality_ratio.tolist()),
        shot_quality_trend=tuple(shot_quality_trend.tolist()),
    )
