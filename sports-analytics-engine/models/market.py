"""Market data contracts: price ticks and derived probability snapshots.

These are pure analytical inputs. Nothing in this module places, models,
or represents a real financial order. Prices are treated strictly as an
analytical signal alongside quant metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class MarketOutcome(str, Enum):
    """Supported 1X2-style market outcomes."""

    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"


class InvalidPriceError(ValueError):
    """Raised when a price tick contains a non-usable price value."""


@dataclass(frozen=True)
class PriceTick:
    """A single normalized market price observation.

    Attributes:
        fixture_id: fixture this price relates to.
        market_type: identifier of the market (e.g. "1X2", "OVER_UNDER_2_5").
        outcome: the specific outcome this price is quoted for.
        price: decimal odds price (must be > 1.0 to be usable).
        timestamp: unix epoch seconds when this price was observed.
    """

    fixture_id: str
    market_type: str
    outcome: str
    price: float
    timestamp: float

    def __post_init__(self) -> None:
        validate_price(self.price)


def validate_price(price: float) -> None:
    """Validate that a decimal odds price is usable for probability math.

    Args:
        price: the decimal odds price to validate.

    Raises:
        InvalidPriceError: if price is None-like, NaN, Inf, zero, or
            not strictly greater than 1.0 (decimal odds floor).
    """
    if price is None:
        raise InvalidPriceError("price is None")
    if isinstance(price, bool):
        raise InvalidPriceError("price must be numeric, not bool")
    if not isinstance(price, (int, float)):
        raise InvalidPriceError(f"price must be numeric, got {type(price)!r}")
    if math.isnan(price):
        raise InvalidPriceError("price is NaN")
    if math.isinf(price):
        raise InvalidPriceError("price is infinite")
    if price <= 1.0:
        raise InvalidPriceError("price must be strictly greater than 1.0")


@dataclass(frozen=True)
class MarketProbability:
    """Derived, vig-free probability snapshot for a market at a point in time.

    Attributes:
        fixture_id: fixture this probability snapshot relates to.
        market_type: identifier of the market these probabilities describe.
        raw_probabilities: outcome -> 1/price, before overround removal.
        fair_probabilities: outcome -> normalized probability summing to 1.
        overround: sum of raw probabilities (the bookmaker margin, M >= 1).
        timestamp: unix epoch seconds this probability snapshot was derived.
    """

    fixture_id: str
    market_type: str
    raw_probabilities: Dict[str, float] = field(default_factory=dict)
    fair_probabilities: Dict[str, float] = field(default_factory=dict)
    overround: float = 0.0
    timestamp: float = 0.0
