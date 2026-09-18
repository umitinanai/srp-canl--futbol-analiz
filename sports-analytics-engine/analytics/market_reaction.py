"""Market-probability and composite-quality analytics: P_fair, MRE, SQS.

This module holds pure mathematical helpers required by later stages
(the future Market/Auditor agents) but does NOT implement a Market
Agent or any agent architecture itself -- it is analytics-layer-only,
consuming and reusing the existing Stage 1.1 contracts wherever
possible rather than duplicating them:

- price validation reuses models.market.validate_price
- SQS weight keys reuse config.settings.EXPECTED_SQS_WEIGHT_KEYS

so this module and the Stage 1.1 contracts can never silently drift
apart.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from config.settings import EXPECTED_SQS_WEIGHT_KEYS
from models.market import InvalidPriceError, validate_price


class MarketReactionValidationError(ValueError):
    """Raised when an input to a market-reaction calculation is invalid."""


def calculate_implied_probabilities(prices: Mapping[str, float]) -> Mapping[str, float]:
    """Compute raw implied probabilities: P_raw_i = 1 / price_i.

    Args:
        prices: mapping of outcome name to decimal odds price. Every
            price must be valid per models.market.validate_price
            (> 1.0, finite, non-NaN).

    Returns:
        A mapping of outcome name to raw implied probability.

    Raises:
        MarketReactionValidationError: if prices is empty.
        InvalidPriceError: if any price is invalid (propagated from
            models.market.validate_price, reusing the Stage 1.1 price
            contract rather than duplicating it).
    """
    if len(prices) == 0:
        raise MarketReactionValidationError("prices must contain at least one outcome")
    raw: dict = {}
    for outcome, price in prices.items():
        validate_price(price)
        raw[outcome] = 1.0 / price
    return raw


def calculate_overround(prices: Mapping[str, float]) -> float:
    """Compute the overround (bookmaker margin): M = sum(P_raw_i).

    Args:
        prices: mapping of outcome name to decimal odds price.

    Returns:
        The sum of raw implied probabilities, typically slightly > 1.0.

    Raises:
        MarketReactionValidationError: if prices is empty.
        InvalidPriceError: if any price is invalid.
    """
    raw = calculate_implied_probabilities(prices)
    return sum(raw.values())


def calculate_fair_probabilities(prices: Mapping[str, float]) -> Mapping[str, float]:
    """Compute vig-free fair probabilities: P_fair_i = P_raw_i / sum(P_raw).

    Args:
        prices: mapping of outcome name to decimal odds price.

    Returns:
        A mapping of outcome name to fair (normalized) probability,
        summing to 1.0 within floating-point tolerance.

    Raises:
        MarketReactionValidationError: if prices is empty or the
            overround is zero (degenerate input).
        InvalidPriceError: if any price is invalid.
    """
    raw = calculate_implied_probabilities(prices)
    overround = sum(raw.values())
    if overround <= 0:
        raise MarketReactionValidationError(
            f"overround must be strictly positive to normalize, got {overround}"
        )
    return {outcome: value / overround for outcome, value in raw.items()}


def calculate_mre(delta_market_probability: float, delta_quant_metric: float) -> Optional[float]:
    """Compute Market Reaction Elasticity: MRE = delta_market_probability / delta_quant_metric.

    Args:
        delta_market_probability: change in market-implied fair
            probability over the interval.
        delta_quant_metric: change in the reference quant metric over
            the same interval.

    Returns:
        The MRE ratio, or None if delta_quant_metric == 0 -- this is
        the explicit, detectable "undefined" representation rather
        than raising or silently dividing by zero, so downstream
        components (e.g. the future Auditor) can flag it as an
        anomaly.

    Raises:
        MarketReactionValidationError: if either input is not a finite
            real number.
    """
    delta_market_probability = _validate_finite("delta_market_probability", delta_market_probability)
    delta_quant_metric = _validate_finite("delta_quant_metric", delta_quant_metric)
    if delta_quant_metric == 0.0:
        return None
    return delta_market_probability / delta_quant_metric


def rolling_mre(
    market_probabilities: Mapping[int, float],
    quant_metric_values: Mapping[int, float],
    min_observations: int = 2,
) -> tuple:
    """Compute MRE across a sequence of aligned observations.

    Args:
        market_probabilities: ordered sequence of market fair
            probabilities (index 0 = earliest). Passed as any indexable
            sequence; typed as Mapping[int, float] here only to
            document that positional alignment with quant_metric_values
            is required.
        quant_metric_values: ordered sequence of quant metric values,
            aligned with market_probabilities (same length).
        min_observations: minimum number of observations required.
            Must be >= 2 (at least one delta is needed).

    Returns:
        A tuple of Optional[float] MRE values, one per consecutive
        pair of observations (length = len(market_probabilities) - 1).
        Each entry is None wherever the corresponding quant-metric
        delta was zero.

    Raises:
        MarketReactionValidationError: if inputs are misshapen, too
            short, non-finite, or min_observations is invalid.
    """
    if isinstance(min_observations, bool) or not isinstance(min_observations, int) or min_observations < 2:
        raise MarketReactionValidationError(
            f"min_observations must be an int >= 2, got {min_observations!r}"
        )
    market_list = list(market_probabilities)
    quant_list = list(quant_metric_values)
    if len(market_list) != len(quant_list):
        raise MarketReactionValidationError(
            "market_probabilities and quant_metric_values must have equal length"
        )
    if len(market_list) < min_observations:
        raise MarketReactionValidationError(
            f"at least {min_observations} observations are required, got {len(market_list)}"
        )

    results = []
    for i in range(1, len(market_list)):
        delta_market = _validate_finite("market_probabilities[i]", market_list[i]) - _validate_finite(
            "market_probabilities[i-1]", market_list[i - 1]
        )
        delta_quant = _validate_finite("quant_metric_values[i]", quant_list[i]) - _validate_finite(
            "quant_metric_values[i-1]", quant_list[i - 1]
        )
        results.append(calculate_mre(delta_market, delta_quant))
    return tuple(results)


def _validate_finite(name: str, value: float) -> float:
    """Validate that value is a finite, non-NaN real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        MarketReactionValidationError: if value is not numeric, NaN, or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MarketReactionValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value):
        raise MarketReactionValidationError(f"{name} is NaN")
    if math.isinf(value):
        raise MarketReactionValidationError(f"{name} is infinite")
    return value


def _validate_sqs_weights(weights: Mapping[str, float]) -> Mapping[str, float]:
    """Validate an SQS weight map against the shared Stage 1.1 key contract.

    Args:
        weights: candidate weight mapping.

    Returns:
        weights, unchanged.

    Raises:
        MarketReactionValidationError: if the key set does not exactly
            match config.settings.EXPECTED_SQS_WEIGHT_KEYS, if any
            weight is invalid, or if the weights do not sum to ~1.0.
    """
    actual_keys = frozenset(weights.keys())
    if actual_keys != EXPECTED_SQS_WEIGHT_KEYS:
        missing = EXPECTED_SQS_WEIGHT_KEYS - actual_keys
        unexpected = actual_keys - EXPECTED_SQS_WEIGHT_KEYS
        raise MarketReactionValidationError(
            f"weights keys must exactly equal {sorted(EXPECTED_SQS_WEIGHT_KEYS)}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    total = 0.0
    for key, value in weights.items():
        value = _validate_finite(f"weights[{key!r}]", value)
        total += value
    if abs(total - 1.0) > 1e-6:
        raise MarketReactionValidationError(f"weights must sum to 1.0, got {total}")
    return weights


def calculate_sqs(
    z_mi: Optional[float],
    z_pai: Optional[float],
    z_sqp: Optional[float],
    z_mre: Optional[float],
    data_quality: Optional[float],
    calibration_confidence: Optional[float],
    weights: Mapping[str, float],
) -> float:
    """Compute the Signal Quality Score (SQS) as the authoritative weighted sum.

    SQS = weights["z_mi"] * z_mi + weights["z_pai"] * z_pai
        + weights["z_sqp"] * z_sqp + weights["z_mre"] * z_mre
        + weights["data_quality"] * data_quality
        + weights["calibration_confidence"] * calibration_confidence

    Args:
        z_mi: momentum z-score component.
        z_pai: pressure-acceleration z-score component.
        z_sqp: shot-quality-proxy z-score component.
        z_mre: market-reaction-elasticity z-score component. May
            legitimately be None (e.g. propagated from calculate_mre()'s
            explicit undefined case) -- this is treated as a genuinely
            missing component, not silently zeroed.
        data_quality: data quality score component, in [0, 1].
        calibration_confidence: calibration confidence component, in [0, 1].
        weights: SQS component weights. Must contain exactly
            config.settings.EXPECTED_SQS_WEIGHT_KEYS and sum to ~1.0
            (see config.settings.Settings.sqs_weights for the
            Stage 1.1-managed default).

    Returns:
        The raw SQS value (a weighted sum of z-scores and quality
        scores; not bounded to [0, 1] since z-scores are unbounded).
        Use normalize_sqs_to_scale() for an optional bounded 0-100
        presentation.

    Raises:
        MarketReactionValidationError: if any required component is
            None (explicit missing-data policy: no silent zero-fill)
            or non-finite, or if weights are invalid.
    """
    weights = _validate_sqs_weights(weights)

    components = {
        "z_mi": z_mi,
        "z_pai": z_pai,
        "z_sqp": z_sqp,
        "z_mre": z_mre,
        "data_quality": data_quality,
        "calibration_confidence": calibration_confidence,
    }
    missing = [key for key, value in components.items() if value is None]
    if missing:
        raise MarketReactionValidationError(
            f"SQS requires all components to be present; missing={sorted(missing)}"
        )

    total = 0.0
    for key, value in components.items():
        value = _validate_finite(f"components[{key!r}]", value)
        total += weights[key] * value
    return total


def normalize_sqs_to_scale(
    raw_sqs: float, scale_min: float = -3.0, scale_max: float = 3.0
) -> float:
    """Map a raw SQS value onto a bounded 0-100 presentation scale.

    ASSUMPTION: the authoritative SQS formula (calculate_sqs) produces
    an unbounded weighted sum of z-scores; the source specification
    does not define a canonical 0-100 mapping. This function is an
    explicit, isolated, documented Stage 2 presentation utility -- it
    is NOT part of the authoritative SQS formula, only a convenience
    view of it, and is trivially replaceable.

    Values are linearly mapped from [scale_min, scale_max] to
    [0, 100], with clipping applied outside that range (so a value at
    or below scale_min maps to 0, and at or above scale_max maps to 100).

    Args:
        raw_sqs: the raw SQS value from calculate_sqs().
        scale_min: raw SQS value mapped to 0 on the output scale.
        scale_max: raw SQS value mapped to 100 on the output scale.
            Must be strictly greater than scale_min.

    Returns:
        A value in [0, 100].

    Raises:
        MarketReactionValidationError: if any input is invalid, or if
            scale_max <= scale_min.
    """
    raw_sqs = _validate_finite("raw_sqs", raw_sqs)
    scale_min = _validate_finite("scale_min", scale_min)
    scale_max = _validate_finite("scale_max", scale_max)
    if scale_max <= scale_min:
        raise MarketReactionValidationError("scale_max must be strictly greater than scale_min")

    clipped = min(max(raw_sqs, scale_min), scale_max)
    return (clipped - scale_min) / (scale_max - scale_min) * 100.0
