"""Stage 4 Risk Agent: produces RiskMetrics from repository-backed evidence only.

RiskMetrics is, by its own docstring (models/metrics.py), "historical/
simulation-only risk analytics" -- "No field in this dataclass
represents a real trade, a real capital allocation, or an executable
order." This agent preserves that scope exactly: it performs no bet
sizing, no bankroll allocation, no Kelly staking, and no execution of
any kind.

Field-by-field evidence for what this agent can and cannot populate:

    - model_confidence: passed through from QuantResult.calibration_confidence
      verbatim (no computation). Nothing in data.pipeline currently sets
      that field, so it is None today -- an honest reflection of what
      the pipeline actually produces, not a gap this agent papers over.
    - uncertainty: the only repository-authoritative notion of
      "uncertainty" attached to a Monte Carlo run is its own confidence-
      interval half-widths (analytics.monte_carlo.MonteCarloResult's
      *_ci95 fields, whose own docstring calls them "confidence-interval
      metadata"). When a caller supplies the actual MonteCarloResult
      object the original simulation produced (QuantResult itself does
      not retain it -- only three summary probabilities), this agent
      reports the mean of the three ci95 half-widths as `uncertainty`.
      This is a direct read of an already-computed, already-authoritative
      figure (a trivial arithmetic mean of three existing numbers), not
      a new statistical formula. Without a MonteCarloResult, uncertainty
      is None.
    - volatility, simulated_exposure, simulated_drawdown, risk_score:
      no authoritative formula for any of these exists anywhere in
      analytics/, backtesting/, or elsewhere in the repository (confirmed
      during the Stage 4 preflight). Per "do not invent missing
      mathematics", these are always None.
    - calibration_error: computed via the existing, already-authoritative
      backtesting.metrics.calibration_error() -- but that function
      requires KNOWN outcomes (0/1), which only exist for historical/
      completed fixtures, never for a live in-progress QuantResult. This
      agent therefore only populates it when a caller explicitly supplies
      historical (predicted_probabilities, outcomes) data; otherwise None.
"""

from __future__ import annotations

from typing import Optional, Sequence

from analytics.monte_carlo import MonteCarloResult
from backtesting.metrics import calibration_error as _calibration_error
from models.metrics import QuantResult, RiskMetrics


class RiskAgent:
    """Produces RiskMetrics using only already-existing repository calculations.

    Stateless aside from its optional constructor-injected Journal Agent.
    """

    def __init__(self, journal: Optional[object] = None) -> None:
        """Initialize the Risk Agent.

        Args:
            journal: optional agents.journal_agent.JournalAgent (or any
                object exposing the same async record_risk_metrics()
                method), duck-typed to avoid a hard compile-time
                dependency.
        """
        self._journal = journal

    def evaluate(
        self,
        quant_result: QuantResult,
        monte_carlo_result: Optional[MonteCarloResult] = None,
        historical_predicted_probabilities: Optional[Sequence[float]] = None,
        historical_outcomes: Optional[Sequence[int]] = None,
    ) -> RiskMetrics:
        """Produce a RiskMetrics for one QuantResult.

        Args:
            quant_result: the QuantResult to derive risk analytics for.
            monte_carlo_result: the actual MonteCarloResult object from
                the simulation that produced quant_result's
                monte_carlo_* fields, if the caller happens to have
                retained it (e.g. a backtesting/replay context). Never
                fabricated by this agent if absent.
            historical_predicted_probabilities: known-outcome historical
                predictions, for calibration_error only. Must be
                supplied together with historical_outcomes.
            historical_outcomes: binary (0/1) outcomes aligned with
                historical_predicted_probabilities.

        Returns:
            A RiskMetrics with every field populated only when a
            repository-backed source for it is actually available;
            otherwise left None (see module docstring).
        """
        uncertainty: Optional[float] = None
        if monte_carlo_result is not None:
            uncertainty = (
                monte_carlo_result.home_win_ci95
                + monte_carlo_result.draw_ci95
                + monte_carlo_result.away_win_ci95
            ) / 3.0

        calibration_error_value: Optional[float] = None
        if historical_predicted_probabilities is not None and historical_outcomes is not None:
            calibration_error_value = _calibration_error(
                historical_predicted_probabilities, historical_outcomes
            )

        return RiskMetrics(
            fixture_id=quant_result.fixture_id,
            timestamp=quant_result.timestamp,
            model_confidence=quant_result.calibration_confidence,
            uncertainty=uncertainty,
            volatility=None,
            simulated_exposure=None,
            simulated_drawdown=None,
            risk_score=None,
            calibration_error=calibration_error_value,
        )

    async def persist(self, risk_metrics: RiskMetrics) -> None:
        """Persist a RiskMetrics through the Journal Agent, if one is configured."""
        if self._journal is not None:
            await self._journal.record_risk_metrics(risk_metrics)
