"""Lightweight, deterministic regime-change detection.

ASSUMPTION / EXPLICIT DISCLOSURE: the source Vanguard specification
does not define exact numerical thresholds for what constitutes a
"meaningful" transition in live match dynamics. This module isolates
those thresholds as an explicit, centralized, overridable
configuration dataclass (RegimeThresholds) rather than burying magic
numbers inside the detection logic. It combines existing, already-
authoritative analytical signals (Z_MI, PAI, shot-quality trend) --
it does not introduce any new statistical model, machine learning, or
training step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


class RegimeValidationError(ValueError):
    """Raised when an input to regime-change detection is invalid."""


def _validate_optional_finite(name: str, value: Optional[float]) -> Optional[float]:
    """Validate that value is either None or a finite real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate, or None.

    Returns:
        value, coerced to float, or None.

    Raises:
        RegimeValidationError: if value is present but not numeric,
            NaN, or infinite.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegimeValidationError(f"{name} must be numeric or None, got {type(value)!r}")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise RegimeValidationError(f"{name} must be finite, got {value}")
    return value


@dataclass(frozen=True)
class RegimeThresholds:
    """Centralized, explicit thresholds defining a "material" regime change.

    Attributes:
        z_mi_threshold: absolute Z_MI value at or above which momentum
            is considered to have shifted materially.
        pai_threshold: absolute pressure-acceleration (PAI) value at or
            above which pressure is considered to be accelerating
            materially.
        shot_quality_trend_threshold: absolute shot-quality trend value
            at or above which shot quality is considered to be trending
            materially.
    """

    z_mi_threshold: float = 1.5
    pai_threshold: float = 0.05
    shot_quality_trend_threshold: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
            ("z_mi_threshold", self.z_mi_threshold),
            ("pai_threshold", self.pai_threshold),
            ("shot_quality_trend_threshold", self.shot_quality_trend_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RegimeValidationError(f"{name} must be numeric, got {type(value)!r}")
            if math.isnan(value) or math.isinf(value):
                raise RegimeValidationError(f"{name} must be finite, got {value}")
            if value < 0:
                raise RegimeValidationError(f"{name} must be non-negative, got {value}")


#: Default, documented regime-change thresholds. See RegimeThresholds
#: for the explicit disclosure of these values' assumption status.
DEFAULT_REGIME_THRESHOLDS = RegimeThresholds()


def detect_regime_change(
    z_mi: Optional[float],
    pressure_acceleration: Optional[float],
    shot_quality_trend: Optional[float],
    thresholds: RegimeThresholds = DEFAULT_REGIME_THRESHOLDS,
) -> bool:
    """Determine whether a meaningful regime change has occurred.

    A regime change is signalled if ANY supplied signal's absolute
    value meets or exceeds its configured threshold. A signal that is
    None (not yet available, e.g. insufficient rolling observations)
    contributes nothing -- it neither triggers nor suppresses a regime
    change on its own.

    Args:
        z_mi: current momentum z-score, or None if unavailable.
        pressure_acceleration: current PAI value, or None if unavailable.
        shot_quality_trend: current shot-quality trend value, or None
            if unavailable.
        thresholds: the RegimeThresholds configuration to evaluate
            against. Defaults to DEFAULT_REGIME_THRESHOLDS.

    Returns:
        True if at least one available signal exceeds its threshold,
        False otherwise (including when all three signals are None).

    Raises:
        RegimeValidationError: if any provided signal is not a finite
            real number.
    """
    z_mi = _validate_optional_finite("z_mi", z_mi)
    pressure_acceleration = _validate_optional_finite(
        "pressure_acceleration", pressure_acceleration
    )
    shot_quality_trend = _validate_optional_finite("shot_quality_trend", shot_quality_trend)

    if z_mi is not None and abs(z_mi) >= thresholds.z_mi_threshold:
        return True
    if (
        pressure_acceleration is not None
        and abs(pressure_acceleration) >= thresholds.pai_threshold
    ):
        return True
    if (
        shot_quality_trend is not None
        and abs(shot_quality_trend) >= thresholds.shot_quality_trend_threshold
    ):
        return True
    return False
