"""Recalculation pipeline: orchestrates Stage 2A analytics over Stage 2B live state.

This module answers exactly one question: "when a recalculation is
warranted (per data.state_manager.should_recalculate), in what order do
the existing Stage 2A analytics functions get called, and how do their
outputs get assembled into a single result?" It introduces NO new
mathematics of its own -- every numeric calculation delegates to an
existing, authoritative analytics.* function. The only exception is the
"dynamic lambda" wiring in compute_dynamic_lambda(), which is a
straightforward application of the already-isolated, already-documented
analytics.bayesian conjugate Gamma-Poisson update (see that module's
docstring for the disclosed assumption) -- no new formula is invented
here either.

Architectural boundary (Stage 2B finalization): every analytics-facing
function in this module (run_lightweight_recalculation,
run_full_recalculation) takes a data.state_manager.AnalyticsSnapshot,
never a mutable LiveMatchState and never a StateManager. A snapshot is
immutable, provider-independent and fully self-contained (including the
bounded mi_history/pressure_history it was built with), so an analytics
calculation can never observe a state mutation happening concurrently
with, or after, the snapshot was taken:

    STATE (mutable) -> AnalyticsSnapshot (immutable) -> analytics.*

process_event() is the single orchestration entry point that wires the
full live flow together end-to-end:

    raw event -> normalize -> StateManager.apply_event -> build_snapshot
        -> should_recalculate -> (lightweight | full) recalculation

This is also the single place where live processing (this module) and
backtest replay (backtesting.engine) are guaranteed to share the same
mathematics: both call analytics.poisson.score_probability_matrix() /
outcome_probabilities() directly, with no separate implementation.

The pipeline's full-recalculation output type is models.metrics.QuantResult,
reusing the Stage 1.1 model contract exactly as intended for later
stages (e.g. a future Quant Agent) rather than introducing a competing
result type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from analytics.bayesian import GammaPoissonPrior, posterior_mean_lambda, update_gamma_poisson
from analytics.monte_carlo import DEFAULT_MONTE_CARLO_SIMULATIONS, run_monte_carlo_simulation
from analytics.momentum import calculate_mi_zscore
from analytics.poisson import calculate_remaining_lambda, outcome_probabilities, score_probability_matrix
from analytics.pressure import calculate_pressure_index
from analytics.xg_proxy import calculate_sqp, calculate_xg_proxy
from data.normalizer import normalize_raw_event
from data.state_manager import (
    DEFAULT_RECALCULATION_POLICY,
    AnalyticsSnapshot,
    RecalculationPolicy,
    StateManager,
    should_recalculate,
)
from models.metrics import QuantResult


class PipelineValidationError(ValueError):
    """Raised when an input to the recalculation pipeline is invalid."""


def compute_team_pressure_components(components: Mapping[str, float]) -> Dict[str, float]:
    """Derive a pressure-components dict with xg_proxy recomputed from shot stats.

    A snapshot's home_components/away_components already contain the
    six EXPECTED_PRESSURE_WEIGHT_KEYS (via TeamFeatures.as_pressure_components()),
    but their "xg_proxy" entry is a raw placeholder (0.0), since
    TeamFeatures.xg_proxy is not populated directly from provider
    events -- it is a derived analytical quantity. This function
    computes the real value via the authoritative
    analytics.xg_proxy.calculate_xg_proxy() from the side's accumulated
    shots_on_target/dangerous_attacks, and returns a new dict with that
    entry substituted (the input mapping, e.g. an AnalyticsSnapshot's
    MappingProxyType component dict, is never mutated).

    Args:
        components: a components mapping containing at least
            "shots_on_target" and "dangerous_attacks" (as produced by
            AnalyticsSnapshot.home_components / away_components).

    Returns:
        A new, plain dict with all six EXPECTED_PRESSURE_WEIGHT_KEYS
        populated and "xg_proxy" recomputed, ready for
        analytics.pressure.calculate_pressure_index().
    """
    updated = dict(components)
    updated["xg_proxy"] = calculate_xg_proxy(
        shots_on_target=components["shots_on_target"],
        dangerous_attacks=components["dangerous_attacks"],
    )
    return updated


def compute_dynamic_lambda(
    prior: GammaPoissonPrior, observed_goals: float, minute: float, regulation_minutes: float
) -> float:
    """Derive a full-match scoring intensity from a prior updated with live evidence.

    This is Stage 2B's "Bayesian / dynamic update" integration point
    (see module docstring): the prior is updated via the standard
    conjugate Gamma-Poisson relationship (analytics.bayesian) using the
    goals actually observed so far and the fraction of the match
    elapsed, then the posterior mean is returned as the current
    best-estimate full-match lambda for analytics.poisson.

    Args:
        prior: the team's current Gamma-Poisson belief about its
            full-match scoring intensity.
        observed_goals: goals scored by this team so far.
        minute: current match minute.
        regulation_minutes: regulation length of the match in minutes.

    Returns:
        The updated (posterior) full-match scoring intensity. If
        minute <= 0 (no time has elapsed yet), the prior's own mean is
        returned unchanged, since there is no elapsed-time evidence yet
        to incorporate.

    Raises:
        PipelineValidationError: if regulation_minutes is not strictly
            positive.
    """
    if regulation_minutes <= 0:
        raise PipelineValidationError("regulation_minutes must be strictly positive")
    if minute <= 0:
        return prior.mean
    elapsed_fraction = minute / regulation_minutes
    posterior = update_gamma_poisson(prior, observed_goals, elapsed_fraction)
    return posterior_mean_lambda(posterior)


def _safe_rate(delta_value: float, delta_t: float) -> Optional[float]:
    """Compute delta_value / delta_t, or None if delta_t is zero.

    Args:
        delta_value: change in the quantity of interest.
        delta_t: elapsed time over the interval.

    Returns:
        The rate, or None if delta_t == 0 (explicit undefined case,
        matching the project's zero-denominator convention rather than
        raising mid-pipeline for a single stale history pair).
    """
    if delta_t == 0:
        return None
    return delta_value / delta_t


def _rate_from_history(history: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Compute a rate-of-change from the last two entries of a bounded history.

    Args:
        history: a sequence of (timestamp, value) tuples, oldest first
            (an AnalyticsSnapshot's immutable mi_history/pressure_history).

    Returns:
        (last_value - previous_value) / (last_t - previous_t), or None
        if fewer than two observations are available or the two most
        recent timestamps are equal.
    """
    if len(history) < 2:
        return None
    (t_prev, v_prev), (t_last, v_last) = history[-2], history[-1]
    return _safe_rate(v_last - v_prev, t_last - t_prev)


def _zscore_from_history(history: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Compute a z-score of the latest value against the full bounded history.

    Args:
        history: a sequence of (timestamp, value) tuples, oldest first.

    Returns:
        calculate_mi_zscore(latest_value, mean, std) over all values
        currently in history, or None if history has fewer than 2
        observations (insufficient data for a meaningful dispersion
        estimate).
    """
    if len(history) < 2:
        return None
    values = np.array([value for _, value in history], dtype=np.float64)
    return calculate_mi_zscore(
        mi_current=float(values[-1]), mi_mean=float(values.mean()), mi_std=float(values.std())
    )


def run_lightweight_recalculation(snapshot: AnalyticsSnapshot) -> Dict[str, float]:
    """Cheaply recompute only the pressure index for both sides, no Monte Carlo.

    This is the "NO -> lightweight state update" branch of the
    recalculation policy: it is safe to call on every event regardless
    of materiality, since it is O(1) and never invokes the 50,000-run
    Monte Carlo engine. Operates entirely on the immutable snapshot --
    it never reads a mutable LiveMatchState or StateManager.

    Args:
        snapshot: an AnalyticsSnapshot, as produced by
            StateManager.build_snapshot().

    Returns:
        A dict with "home_pressure_index" and "away_pressure_index".

    Raises:
        PressureValidationError: propagated from
            analytics.pressure.calculate_pressure_index() if the
            snapshot's component values are themselves invalid (e.g.
            negative, which StateManager.apply_update() already
            prevents from ever being stored).
    """
    from config.settings import Settings

    weights = Settings().pressure_weights
    home_components = compute_team_pressure_components(snapshot.home_components)
    away_components = compute_team_pressure_components(snapshot.away_components)
    return {
        "home_pressure_index": calculate_pressure_index(home_components, weights),
        "away_pressure_index": calculate_pressure_index(away_components, weights),
    }


def run_full_recalculation(
    snapshot: AnalyticsSnapshot,
    home_prior: GammaPoissonPrior,
    away_prior: GammaPoissonPrior,
    regulation_minutes: float,
    pressure_weights: Mapping[str, float],
    monte_carlo_simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
    monte_carlo_seed: Optional[int] = None,
) -> QuantResult:
    """Run the full Stage 2A analytics pipeline over an immutable snapshot.

    Operates entirely on the AnalyticsSnapshot -- it never reads a
    mutable LiveMatchState or StateManager, so this calculation cannot
    be affected by (or interfere with) any state mutation happening
    concurrently with or after the snapshot was taken.

    Call order (fixed, matching the existing Stage 2A public API
    contracts -- no new mathematics is introduced here):

        1. analytics.bayesian: update each side's scoring-intensity
           prior with observed goals/elapsed time -> dynamic full-match
           lambda.
        2. analytics.poisson.calculate_remaining_lambda(): scale to the
           time remaining in the match.
        3. analytics.poisson.score_probability_matrix() /
           outcome_probabilities(): closed-form home/draw/away
           probabilities.
        4. analytics.monte_carlo.run_monte_carlo_simulation(): the
           50,000-run (by default) simulation-based cross-check.
        5. analytics.xg_proxy.calculate_xg_proxy() /
           analytics.pressure.calculate_pressure_index(): per-side
           composite pressure, using the xg_proxy value as one of its
           six input components.
        6. analytics.xg_proxy.calculate_sqp(): shot quality proxy from
           the home side's accumulated stats.
        7. Rolling MI_rate / Z_MI and pressure acceleration (PAI) from
           the snapshot's immutable mi_history / pressure_history, when
           at least two observations are available.

    Args:
        snapshot: an AnalyticsSnapshot, as produced by
            StateManager.build_snapshot().
        home_prior: home side's current Gamma-Poisson scoring-intensity
            belief (see analytics.bayesian).
        away_prior: away side's current Gamma-Poisson scoring-intensity
            belief.
        regulation_minutes: regulation length of the match in minutes
            (from the existing Stage 1.1 LeagueConfig, not hard-coded).
        pressure_weights: pressure component weights (from the existing
            Stage 1.1 Settings.pressure_weights contract).
        monte_carlo_simulations: simulation count; defaults to
            DEFAULT_MONTE_CARLO_SIMULATIONS (50,000) and is never
            silently downgraded below analytics.monte_carlo.MINIMUM_SIMULATIONS.
        monte_carlo_seed: optional deterministic seed for the Monte
            Carlo simulation.

    Returns:
        A populated QuantResult. market_reaction_elasticity,
        fair_probabilities, signal_quality_score, data_quality and
        calibration_confidence are left at their QuantResult defaults
        (None / empty) since this pipeline call has no market price
        input -- callers that do have market data should combine this
        result with analytics.market_reaction functions separately
        rather than this pipeline inventing placeholder values.

    Raises:
        PipelineValidationError: if regulation_minutes is invalid.
        PoissonValidationError, MonteCarloValidationError,
        PressureValidationError, XGProxyValidationError: propagated
            from the underlying analytics calls for genuinely invalid
            inputs.
    """
    lambda_home_full = compute_dynamic_lambda(
        home_prior, snapshot.home_goals, snapshot.minute, regulation_minutes
    )
    lambda_away_full = compute_dynamic_lambda(
        away_prior, snapshot.away_goals, snapshot.minute, regulation_minutes
    )

    lambda_home_remaining = calculate_remaining_lambda(
        lambda_home_full, snapshot.minute, regulation_minutes
    )
    lambda_away_remaining = calculate_remaining_lambda(
        lambda_away_full, snapshot.minute, regulation_minutes
    )

    matrix = score_probability_matrix(lambda_home_remaining, lambda_away_remaining)
    p_home, p_draw, p_away = outcome_probabilities(matrix)

    mc_result = run_monte_carlo_simulation(
        lambda_home_remaining,
        lambda_away_remaining,
        simulations=monte_carlo_simulations,
        seed=monte_carlo_seed,
    )

    home_components = compute_team_pressure_components(snapshot.home_components)
    away_components = compute_team_pressure_components(snapshot.away_components)
    pressure_home = calculate_pressure_index(home_components, pressure_weights)
    pressure_away = calculate_pressure_index(away_components, pressure_weights)
    # Net pressure: positive means the home side currently dominates
    # possession-adjacent territorial pressure. This combination
    # (home - away) is a Stage 2B presentation choice, not a
    # source-specified formula; QuantResult.pressure_index carries a
    # single scalar and this is the most direct, symmetric reduction.
    net_pressure_index = pressure_home - pressure_away

    sqp_home = calculate_sqp(home_components["xg_proxy"], snapshot.home_components["shots"])

    return QuantResult(
        fixture_id=snapshot.fixture_id,
        timestamp=snapshot.timestamp,
        lambda_home=lambda_home_remaining,
        lambda_away=lambda_away_remaining,
        home_win_probability=p_home,
        draw_probability=p_draw,
        away_win_probability=p_away,
        monte_carlo_home_win=mc_result.home_win_probability,
        monte_carlo_draw=mc_result.draw_probability,
        monte_carlo_away_win=mc_result.away_win_probability,
        monte_carlo_simulations=mc_result.simulations,
        mi_rate=_rate_from_history(snapshot.mi_history),
        z_mi=_zscore_from_history(snapshot.mi_history),
        pressure_index=net_pressure_index,
        pressure_acceleration=_rate_from_history(snapshot.pressure_history),
        xg_proxy=home_components["xg_proxy"],
        shot_quality_proxy=sqp_home,
        metadata={
            "model": "srp_full_recalculation_v1",
            "lambda_home_full": str(lambda_home_full),
            "lambda_away_full": str(lambda_away_full),
        },
    )


@dataclass(frozen=True)
class RecalculationOutcome:
    """Result of a single process_event() orchestration call.

    Exactly one of lightweight_result / full_result is populated,
    matching whichever branch should_recalculate() selected (both are
    None if the event was rejected as a duplicate or stale by
    StateManager.apply_event() -- no analytics run at all in that case,
    since no new information was actually applied).

    Attributes:
        snapshot: the AnalyticsSnapshot taken immediately after the
            event was applied (or the unchanged prior snapshot, if the
            event was rejected).
        event_applied: whether the incoming event passed
            StateManager.apply_event()'s duplicate/staleness checks and
            was actually merged into state.
        recalculated: True if a full recalculation was performed
            (should_recalculate() returned True), False otherwise.
        lightweight_result: the run_lightweight_recalculation() output,
            populated iff event_applied and not recalculated.
        full_result: the run_full_recalculation() output, populated iff
            event_applied and recalculated.
    """

    snapshot: AnalyticsSnapshot
    event_applied: bool
    recalculated: bool
    lightweight_result: Optional[Dict[str, float]]
    full_result: Optional[QuantResult]


def process_event(
    manager: StateManager,
    fixture_id: str,
    raw_event: Mapping[str, Any],
    event_timestamp: float,
    home_prior: GammaPoissonPrior,
    away_prior: GammaPoissonPrior,
    regulation_minutes: float,
    pressure_weights: Mapping[str, float],
    event_id: Optional[str] = None,
    received_timestamp: Optional[float] = None,
    policy: RecalculationPolicy = DEFAULT_RECALCULATION_POLICY,
    monte_carlo_simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
    monte_carlo_seed: Optional[int] = None,
) -> RecalculationOutcome:
    """Single authoritative orchestration entry point for the full live flow.

        raw event -> normalize -> StateManager.apply_event -> build_snapshot
            -> should_recalculate -> (lightweight | full) recalculation

    This is the one place in the codebase where the recalculation
    *decision* (should_recalculate()) and the recalculation
    *execution* (run_lightweight_recalculation() /
    run_full_recalculation()) are actually wired together -- neither
    function calls the other; this orchestration function is what
    connects them, so the connection is explicit, deterministic and
    independently testable rather than implied.

    Args:
        manager: the StateManager tracking this (and possibly other)
            fixtures.
        fixture_id: the fixture identifier.
        raw_event: a raw, provider-shaped event dict (in the provider's
            own field naming -- normalized internally via
            data.normalizer.normalize_raw_event()).
        event_timestamp: unix epoch seconds when the event occurred.
        home_prior: home side's current Gamma-Poisson scoring-intensity
            belief, used only if a full recalculation is triggered.
        away_prior: away side's current Gamma-Poisson scoring-intensity
            belief, used only if a full recalculation is triggered.
        regulation_minutes: regulation length of the match in minutes.
        pressure_weights: pressure component weights.
        event_id: optional provider-supplied unique event identifier,
            forwarded to StateManager.apply_event() for deduplication.
        received_timestamp: unix epoch seconds this event was received
            locally. Defaults to the current time.
        policy: the RecalculationPolicy to evaluate should_recalculate()
            against.
        monte_carlo_simulations: simulation count for a full
            recalculation; defaults to DEFAULT_MONTE_CARLO_SIMULATIONS
            (50,000).
        monte_carlo_seed: optional deterministic Monte Carlo seed.

    Returns:
        A RecalculationOutcome describing exactly what happened.
    """
    canonical_fields = normalize_raw_event(raw_event)
    apply_result = manager.apply_event(
        fixture_id, canonical_fields, event_timestamp, event_id, received_timestamp
    )
    snapshot = manager.build_snapshot(fixture_id)

    if not apply_result.applied:
        return RecalculationOutcome(
            snapshot=snapshot, event_applied=False, recalculated=False,
            lightweight_result=None, full_result=None,
        )

    state = apply_result.state
    is_first_observation = state.observation_count == 1
    elapsed_since_recalc: Optional[float] = None
    if state.last_recalculation_minute is not None:
        elapsed_since_recalc = float(state.minute - state.last_recalculation_minute)

    decision = should_recalculate(
        is_first_observation,
        apply_result.score_changed,
        elapsed_since_recalc,
        z_mi=_zscore_from_history(snapshot.mi_history),
        pressure_acceleration=_rate_from_history(snapshot.pressure_history),
        shot_quality_trend=None,  # not tracked as a rolling history in Stage 2B
        policy=policy,
    )

    if not decision:
        lightweight_result = run_lightweight_recalculation(snapshot)
        return RecalculationOutcome(
            snapshot=snapshot, event_applied=True, recalculated=False,
            lightweight_result=lightweight_result, full_result=None,
        )

    full_result = run_full_recalculation(
        snapshot, home_prior, away_prior, regulation_minutes, pressure_weights,
        monte_carlo_simulations=monte_carlo_simulations, monte_carlo_seed=monte_carlo_seed,
    )
    manager.mark_recalculated(fixture_id, state.minute)
    return RecalculationOutcome(
        snapshot=snapshot, event_applied=True, recalculated=True,
        lightweight_result=None, full_result=full_result,
    )
