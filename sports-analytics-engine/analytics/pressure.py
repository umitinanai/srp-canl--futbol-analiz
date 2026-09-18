"""Pressure analytics: component-based Pressure Index and its acceleration.

Pressure is deliberately not a single raw metric. It is built from
normalized input components (shots, shots_on_target, dangerous_attacks,
corners, possession_changes, xg_proxy), combined via configurable
weights:

    Pressure = sum(weight_i * normalize(component_i))
    PAI      = delta_Pressure / delta_t

The expected component/weight key set is imported directly from
config.settings.EXPECTED_PRESSURE_WEIGHT_KEYS, so this module and the
Stage 1.1 Settings contract can never silently drift apart -- weight
keys are defined in exactly one place (config.settings).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from config.settings import EXPECTED_PRESSURE_WEIGHT_KEYS

#: Reference "half-saturation" scale per component, used by the
#: normalize(x) = x / (x + scale) squashing function: normalize(scale)
#: == 0.5, normalize(0) == 0, normalize(x) -> 1 as x -> infinity. These
#: are Stage 2 analytics-layer defaults (not part of the Stage 1.1
#: config contract), named and overridable via the `scales` parameter
#: rather than being unexplained inline magic numbers.
DEFAULT_COMPONENT_SCALES: Mapping[str, float] = {
    "shots": 10.0,
    "shots_on_target": 5.0,
    "dangerous_attacks": 15.0,
    "corners": 6.0,
    "possession_changes": 20.0,
    "xg_proxy": 2.0,
}


class PressureValidationError(ValueError):
    """Raised when an input to a pressure calculation is invalid."""


def _validate_finite_number(name: str, value: float) -> float:
    """Validate that value is a finite, non-NaN real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        PressureValidationError: if value is not numeric, NaN, or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PressureValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value):
        raise PressureValidationError(f"{name} is NaN")
    if math.isinf(value):
        raise PressureValidationError(f"{name} is infinite")
    return value


def _validate_weights(weights: Mapping[str, float]) -> Mapping[str, float]:
    """Validate a pressure weight map against the shared Stage 1.1 key contract.

    Args:
        weights: candidate weight mapping.

    Returns:
        weights, unchanged.

    Raises:
        PressureValidationError: if the key set does not exactly match
            EXPECTED_PRESSURE_WEIGHT_KEYS, if any weight is invalid, or
            if the weights do not sum to ~1.0.
    """
    actual_keys = frozenset(weights.keys())
    if actual_keys != EXPECTED_PRESSURE_WEIGHT_KEYS:
        missing = EXPECTED_PRESSURE_WEIGHT_KEYS - actual_keys
        unexpected = actual_keys - EXPECTED_PRESSURE_WEIGHT_KEYS
        raise PressureValidationError(
            f"weights keys must exactly equal {sorted(EXPECTED_PRESSURE_WEIGHT_KEYS)}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    total = 0.0
    for key, value in weights.items():
        value = _validate_finite_number(f"weights[{key!r}]", value)
        total += value
    if abs(total - 1.0) > 1e-6:
        raise PressureValidationError(f"weights must sum to 1.0, got {total}")
    return weights


def _validate_scales(scales: Mapping[str, float]) -> Mapping[str, float]:
    """Validate a component normalization-scale map.

    Args:
        scales: candidate scale mapping.

    Returns:
        scales, unchanged.

    Raises:
        PressureValidationError: if the key set does not exactly match
            EXPECTED_PRESSURE_WEIGHT_KEYS, or if any scale is invalid
            (non-finite or non-positive).
    """
    actual_keys = frozenset(scales.keys())
    if actual_keys != EXPECTED_PRESSURE_WEIGHT_KEYS:
        missing = EXPECTED_PRESSURE_WEIGHT_KEYS - actual_keys
        unexpected = actual_keys - EXPECTED_PRESSURE_WEIGHT_KEYS
        raise PressureValidationError(
            f"scales keys must exactly equal {sorted(EXPECTED_PRESSURE_WEIGHT_KEYS)}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for key, value in scales.items():
        value = _validate_finite_number(f"scales[{key!r}]", value)
        if value <= 0:
            raise PressureValidationError(f"scales[{key!r}] must be strictly positive, got {value}")
    return scales


def _validate_components(components: Mapping[str, float]) -> Mapping[str, float]:
    """Validate a raw pressure-component map.

    Missing components follow an explicit missing-data policy: rather
    than silently substituting zero (which would understate pressure
    without any signal that data was absent), a missing required
    component raises immediately, naming exactly which key(s) are
    absent.

    Args:
        components: candidate raw component mapping (shots,
            shots_on_target, dangerous_attacks, corners,
            possession_changes, xg_proxy).

    Returns:
        components, unchanged.

    Raises:
        PressureValidationError: if any expected component is missing,
            an unexpected key is present, or any value is invalid
            (non-finite or negative).
    """
    actual_keys = frozenset(components.keys())
    if actual_keys != EXPECTED_PRESSURE_WEIGHT_KEYS:
        missing = EXPECTED_PRESSURE_WEIGHT_KEYS - actual_keys
        unexpected = actual_keys - EXPECTED_PRESSURE_WEIGHT_KEYS
        raise PressureValidationError(
            f"components keys must exactly equal {sorted(EXPECTED_PRESSURE_WEIGHT_KEYS)}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for key, value in components.items():
        value = _validate_finite_number(f"components[{key!r}]", value)
        if value < 0:
            raise PressureValidationError(f"components[{key!r}] must be non-negative, got {value}")
    return components


def _normalize_component(raw_value: float, scale: float) -> float:
    """Squash a non-negative raw component value into [0, 1).

    normalize(x) = x / (x + scale)

    Args:
        raw_value: the raw, non-negative component value.
        scale: the strictly positive half-saturation reference scale.

    Returns:
        A normalized value in [0, 1).
    """
    return raw_value / (raw_value + scale)


def calculate_pressure_index(
    components: Mapping[str, float],
    weights: Mapping[str, float],
    scales: Optional[Mapping[str, float]] = None,
) -> float:
    """Compute the weighted, normalized Pressure Index for one snapshot.

    Pressure = sum(weight_i * normalize(component_i))

    Args:
        components: raw component values. Must contain exactly the keys
            in config.settings.EXPECTED_PRESSURE_WEIGHT_KEYS.
        weights: component weights. Must contain exactly the same key
            set and sum to ~1.0 (see config.settings.Settings.pressure_weights
            for the Stage 1.1-managed default).
        scales: optional per-component normalization scales; defaults
            to DEFAULT_COMPONENT_SCALES.

    Returns:
        The pressure index, bounded in [0, 1) since each normalized
        component is in [0, 1) and the weights sum to 1.

    Raises:
        PressureValidationError: if components, weights, or scales are
            invalid (see the respective validators).
    """
    components = _validate_components(components)
    weights = _validate_weights(weights)
    scales = _validate_scales(scales if scales is not None else DEFAULT_COMPONENT_SCALES)

    pressure = 0.0
    for key in EXPECTED_PRESSURE_WEIGHT_KEYS:
        normalized = _normalize_component(components[key], scales[key])
        pressure += weights[key] * normalized
    return pressure


def calculate_pressure_acceleration(delta_pressure: float, delta_t: float) -> float:
    """Compute PAI = delta_Pressure / delta_t.

    Args:
        delta_pressure: change in pressure index over the interval.
        delta_t: elapsed time over the interval. Must be strictly
            positive.

    Returns:
        delta_pressure / delta_t.

    Raises:
        PressureValidationError: if either input is invalid, or if
            delta_t is not strictly positive.
    """
    delta_pressure = _validate_finite_number("delta_pressure", delta_pressure)
    delta_t = _validate_finite_number("delta_t", delta_t)
    if delta_t <= 0:
        raise PressureValidationError(
            f"delta_t must be strictly positive to compute acceleration, got {delta_t}"
        )
    return delta_pressure / delta_t


@dataclass(frozen=True)
class RollingPressureResult:
    """Vectorized rolling pressure output over a bounded observation window.

    Attributes:
        pressure_index: pressure index at each observation (length =
            len(component_series)).
        acceleration: PAI between each consecutive pair of observations
            (length = len(component_series) - 1). NaN where the
            corresponding delta_t was zero.
        rolling_mean: rolling mean of pressure_index using the trailing
            `window` values; NaN for positions with fewer than `window`
            preceding observations.
        window: the rolling window size used.
    """

    pressure_index: Tuple[float, ...]
    acceleration: Tuple[float, ...]
    rolling_mean: Tuple[float, ...]
    window: int


def rolling_pressure(
    component_series: Sequence[Mapping[str, float]],
    timestamps: Sequence[float],
    weights: Mapping[str, float],
    scales: Optional[Mapping[str, float]] = None,
    window: int = 5,
) -> RollingPressureResult:
    """Compute pressure index, PAI, and a rolling mean over an observation history.

    Args:
        component_series: one raw component mapping per observation,
            each validated the same way as calculate_pressure_index's
            `components` argument.
        timestamps: strictly increasing observation timestamps, aligned
            with component_series (same length).
        weights: component weights, validated once and reused for every
            observation (see calculate_pressure_index).
        scales: optional per-component normalization scales; defaults
            to DEFAULT_COMPONENT_SCALES.
        window: number of trailing pressure_index observations used for
            each rolling mean. Must be a positive integer.

    Returns:
        A RollingPressureResult.

    Raises:
        PressureValidationError: if inputs are misshapen, too short,
            invalid, window is invalid, or timestamps are not strictly
            increasing.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise PressureValidationError(f"window must be a positive int, got {window!r}")
    if len(component_series) != len(timestamps):
        raise PressureValidationError(
            "component_series and timestamps must have equal length"
        )
    if len(component_series) < 1:
        raise PressureValidationError("component_series must contain at least 1 observation")

    weights = _validate_weights(weights)
    scales = _validate_scales(scales if scales is not None else DEFAULT_COMPONENT_SCALES)

    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.ndim != 1:
        raise PressureValidationError("timestamps must be a 1D sequence")
    if not np.all(np.isfinite(ts)):
        raise PressureValidationError("timestamps must be finite (no NaN/Inf)")
    if ts.shape[0] >= 2 and np.any(np.diff(ts) <= 0):
        raise PressureValidationError("timestamps must be strictly increasing")

    keys = sorted(EXPECTED_PRESSURE_WEIGHT_KEYS)
    raw_rows = []
    for components in component_series:
        validated = _validate_components(components)
        raw_rows.append([validated[key] for key in keys])
    raw_matrix = np.array(raw_rows, dtype=np.float64)

    scale_vec = np.array([scales[key] for key in keys], dtype=np.float64)
    weight_vec = np.array([weights[key] for key in keys], dtype=np.float64)

    normalized_matrix = raw_matrix / (raw_matrix + scale_vec)
    pressure_index = normalized_matrix @ weight_vec

    delta_pressure = np.diff(pressure_index)
    delta_t = np.diff(ts)
    acceleration = np.divide(
        delta_pressure,
        delta_t,
        out=np.full_like(delta_pressure, np.nan),
        where=delta_t != 0,
    )

    n = pressure_index.shape[0]
    rolling_mean = np.full(n, np.nan, dtype=np.float64)
    if n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(pressure_index, window)
        rolling_mean[window - 1 :] = windows.mean(axis=1)

    return RollingPressureResult(
        pressure_index=tuple(pressure_index.tolist()),
        acceleration=tuple(acceleration.tolist()),
        rolling_mean=tuple(rolling_mean.tolist()),
        window=window,
    )
