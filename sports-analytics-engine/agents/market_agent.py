"""Stage 4 Market Agent: consumes existing market contracts and analytics.

This agent does not implement a real odds provider, does not poll any
remote endpoint, and does not define a provider-specific schema --
Stage 4 has no requirement to invent one (preflight correction B). It
accepts already-available models.market.PriceTick instances (however a
future runtime obtains them is a Stage 5 concern) and reuses, unmodified:

    - models.market.validate_price (via PriceTick.__post_init__)
    - analytics.market_reaction.calculate_implied_probabilities /
      calculate_overround / calculate_fair_probabilities / calculate_mre /
      rolling_mre

It never mutates a QuantResult (frozen dataclass; no dataclasses.replace()
enrichment protocol is introduced here -- preflight correction H) and
never invents a persistence table for MarketProbability (it is a pure,
transient computation; only the underlying PriceTick is persisted,
through the Journal Agent, into the existing market_ticks table).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

from analytics.market_reaction import (
    calculate_fair_probabilities,
    calculate_implied_probabilities,
    calculate_mre,
    calculate_overround,
    rolling_mre,
)
from models.market import MarketProbability, PriceTick

_DEFAULT_HISTORY_MAXLEN = 32


class MarketAgent:
    """Ingests PriceTicks, derives fair probabilities, and computes MRE.

    Owns only a small, bounded, in-memory per-(fixture, market_type,
    outcome) price history -- mirroring data.state_manager's bounded-deque
    convention -- used solely to give rolling_market_reaction() a
    convenient place to read recent prices back from if a caller wants
    them; the MRE math itself always delegates to
    analytics.market_reaction.rolling_mre() over caller-supplied,
    already-aligned sequences (no interpolation/alignment logic is
    invented here for series with different update cadences).
    """

    def __init__(self, journal: Optional[Any] = None, history_maxlen: int = _DEFAULT_HISTORY_MAXLEN) -> None:
        """Initialize the Market Agent.

        Args:
            journal: optional agents.journal_agent.JournalAgent (or any
                object exposing the same async record_price_tick()
                method), duck-typed to avoid a hard compile-time
                dependency.
            history_maxlen: bound on the per-(fixture, market_type,
                outcome) price history retained in memory.
        """
        self._journal = journal
        self._history_maxlen = history_maxlen
        self._price_history: Dict[Tuple[str, str, str], Deque[Tuple[float, float]]] = {}

    def price_history(self, fixture_id: str, market_type: str, outcome: str) -> Tuple[Tuple[float, float], ...]:
        """Return the bounded (timestamp, price) history for one outcome, oldest first."""
        key = (fixture_id, market_type, outcome)
        return tuple(self._price_history.get(key, ()))

    async def record_price_tick(self, tick: PriceTick) -> None:
        """Record a validated PriceTick: bounded in-memory history + persistence.

        PriceTick's own __post_init__ already validates the price
        (models.market.validate_price); an invalid price raises
        models.market.InvalidPriceError before this method is ever
        reached, since construction of the PriceTick itself fails.

        Args:
            tick: the PriceTick to record.
        """
        key = (tick.fixture_id, tick.market_type, tick.outcome)
        history = self._price_history.setdefault(key, deque(maxlen=self._history_maxlen))
        history.append((tick.timestamp, tick.price))

        if self._journal is not None:
            await self._journal.record_price_tick(tick)

    def fair_probabilities(
        self, fixture_id: str, market_type: str, prices: Mapping[str, float], timestamp: float
    ) -> MarketProbability:
        """Derive a MarketProbability snapshot from a set of outcome prices.

        Pure reuse of analytics.market_reaction; no new probability math.

        Args:
            fixture_id: fixture these prices relate to.
            market_type: identifier of the market (e.g. "1X2").
            prices: mapping of outcome name to decimal odds price.
            timestamp: unix epoch seconds this price set was observed.

        Returns:
            A MarketProbability with raw/fair probabilities and overround.

        Raises:
            MarketReactionValidationError: if prices is empty or degenerate.
            InvalidPriceError: if any price is invalid.
        """
        raw = calculate_implied_probabilities(prices)
        fair = calculate_fair_probabilities(prices)
        overround = calculate_overround(prices)
        return MarketProbability(
            fixture_id=fixture_id,
            market_type=market_type,
            raw_probabilities=dict(raw),
            fair_probabilities=dict(fair),
            overround=overround,
            timestamp=timestamp,
        )

    def market_reaction_elasticity(
        self, delta_market_probability: float, delta_quant_metric: float
    ) -> Optional[float]:
        """Thin pass-through to analytics.market_reaction.calculate_mre()."""
        return calculate_mre(delta_market_probability, delta_quant_metric)

    def rolling_market_reaction(
        self,
        market_probabilities: Sequence[float],
        quant_metric_values: Sequence[float],
        min_observations: int = 2,
    ) -> Tuple[Optional[float], ...]:
        """Thin pass-through to analytics.market_reaction.rolling_mre().

        Args:
            market_probabilities: ordered, already-aligned fair-probability
                observations (index 0 = earliest).
            quant_metric_values: ordered, already-aligned quant-metric
                observations, same length as market_probabilities.
            min_observations: forwarded unchanged to rolling_mre().

        Returns:
            The tuple of Optional[float] MRE values rolling_mre() returns.
        """
        return rolling_mre(market_probabilities, quant_metric_values, min_observations=min_observations)
