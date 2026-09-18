"""Monte Carlo simulation engine for match outcome distributions.

Simulates independent Poisson-distributed home/away goal counts using
NumPy's vectorized random number generation (no per-simulation Python
objects, no nested Python loops), and aggregates the results into
outcome probabilities, a bounded total-goals distribution, the primary
GoalEdge additional-goal probability, and simple confidence-interval
metadata.

AUTHORITATIVE REQUIREMENT: both the default AND the minimum-accepted
simulation count are 50,000 (DEFAULT_MONTE_CARLO_SIMULATIONS ==
MINIMUM_SIMULATIONS == 50_000). This supersedes an older, now-obsolete
10,000 default/minimum found in earlier project documents -- 10,000 (or
any value below 50,000) is no longer accepted as either a default or a
configured value anywhere in this project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

#: Standard/default number of simulation rounds. AUTHORITATIVE: 50,000.
DEFAULT_MONTE_CARLO_SIMULATIONS = 50_000

#: Absolute minimum simulation count the engine will ever run. Equal to
#: DEFAULT_MONTE_CARLO_SIMULATIONS per the current authoritative
#: requirement: 10,000 (an older, now-obsolete value from earlier
#: project documents) is no longer accepted as a minimum OR a default
#: anywhere in this module.
MINIMUM_SIMULATIONS = 50_000

#: Upper bound on the total-goals index tracked in the returned
#: distribution histogram. Any simulated total above this is folded
#: into the final bucket, keeping the returned array a small, fixed
#: size regardless of simulation count or scoring intensity, per the
#: "bounded arrays" hardware constraint.
MAX_TOTAL_GOALS_BUCKET = 20


class MonteCarloValidationError(ValueError):
    """Raised when an input to a Monte Carlo simulation is invalid."""


def _validate_finite_number(name: str, value: float) -> float:
    """Validate that value is a finite, non-NaN real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        MonteCarloValidationError: if value is not numeric, NaN, or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MonteCarloValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value):
        raise MonteCarloValidationError(f"{name} is NaN")
    if math.isinf(value):
        raise MonteCarloValidationError(f"{name} is infinite")
    return value


def _validate_lambda(name: str, lam: float) -> float:
    """Validate a scoring intensity: finite and non-negative.

    Args:
        name: name of the parameter, used in error messages.
        lam: the lambda value to validate.

    Returns:
        lam, coerced to float.

    Raises:
        MonteCarloValidationError: if lam is invalid or negative.
    """
    lam = _validate_finite_number(name, lam)
    if lam < 0:
        raise MonteCarloValidationError(f"{name} must be non-negative, got {lam}")
    return lam


def _validate_simulations(simulations: int) -> int:
    """Validate the simulation count against the hard minimum floor.

    Args:
        simulations: requested number of simulation rounds.

    Returns:
        simulations, as an int.

    Raises:
        MonteCarloValidationError: if simulations is not an int, or is
            below MINIMUM_SIMULATIONS.
    """
    if isinstance(simulations, bool) or not isinstance(simulations, int):
        raise MonteCarloValidationError(
            f"simulations must be an int, got {type(simulations)!r}"
        )
    if simulations < MINIMUM_SIMULATIONS:
        raise MonteCarloValidationError(
            f"simulations must be >= {MINIMUM_SIMULATIONS}, got {simulations}"
        )
    return simulations


def _wilson_like_ci95(probability: float, n: int) -> float:
    """Compute a normal-approximation 95% confidence half-width for a proportion.

    Args:
        probability: the observed proportion (e.g. simulated win rate),
            in [0, 1].
        n: the number of trials (simulations) the proportion was
            computed from.

    Returns:
        The 95% confidence interval half-width, i.e. the true
        proportion is approximately probability +/- this value.
    """
    variance = probability * (1.0 - probability) / n
    return 1.96 * math.sqrt(max(variance, 0.0))


@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregated outcome of a Monte Carlo match simulation.

    Only aggregated statistics are retained -- the raw per-simulation
    goal arrays are discarded once these aggregates are computed, so
    result size does not grow with `simulations`.
    """

    lambda_home: float
    lambda_away: float
    simulations: int
    seed: Optional[int]

    home_win_probability: float
    draw_probability: float
    away_win_probability: float

    home_win_ci95: float
    draw_ci95: float
    away_win_ci95: float

    mean_total_goals: float
    total_goals_distribution: Tuple[float, ...]

    #: GoalEdge's primary analytical target: P(at least one additional
    #: goal, by either side, over the simulated remaining time).
    at_least_one_goal_probability: float
    at_least_one_goal_ci95: float

    metadata: dict = field(default_factory=dict)


def run_monte_carlo_simulation(
    lambda_home: float,
    lambda_away: float,
    simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """Run a vectorized Monte Carlo simulation of match goal outcomes.

    Home and away goals are each drawn from independent Poisson
    distributions in a single vectorized call per side (no per-simulation
    Python-level object creation, no nested loops).

    Args:
        lambda_home: home team scoring intensity (finite, non-negative),
            typically the remaining-time intensity.
        lambda_away: away team scoring intensity (finite, non-negative),
            typically the remaining-time intensity.
        simulations: number of simulation rounds. Defaults to
            DEFAULT_MONTE_CARLO_SIMULATIONS (50,000). Must be >=
            MINIMUM_SIMULATIONS (50,000); there is no upper limit.
        seed: optional random seed. The same seed together with the
            same lambda_home/lambda_away/simulations always reproduces
            the exact same result (deterministic reproducibility). A
            local np.random.Generator is used -- global NumPy random
            state is never touched.

    Returns:
        A MonteCarloResult with aggregated outcome probabilities,
        confidence intervals, the additional-goal probability, and a
        bounded total-goals distribution.

    Raises:
        MonteCarloValidationError: if any input is invalid.
    """
    lambda_home = _validate_lambda("lambda_home", lambda_home)
    lambda_away = _validate_lambda("lambda_away", lambda_away)
    simulations = _validate_simulations(simulations)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise MonteCarloValidationError(f"seed must be an int or None, got {type(seed)!r}")

    rng = np.random.default_rng(seed)

    home_goals = rng.poisson(lam=lambda_home, size=simulations)
    away_goals = rng.poisson(lam=lambda_away, size=simulations)

    home_wins = int(np.count_nonzero(home_goals > away_goals))
    away_wins = int(np.count_nonzero(away_goals > home_goals))
    draws = simulations - home_wins - away_wins

    home_win_probability = home_wins / simulations
    draw_probability = draws / simulations
    away_win_probability = away_wins / simulations

    total_goals = home_goals + away_goals
    mean_total_goals = float(total_goals.mean())

    at_least_one_goal_count = int(np.count_nonzero(total_goals > 0))
    at_least_one_goal_probability = at_least_one_goal_count / simulations

    bounded_totals = np.minimum(total_goals, MAX_TOTAL_GOALS_BUCKET)
    counts = np.bincount(bounded_totals, minlength=MAX_TOTAL_GOALS_BUCKET + 1)
    total_goals_distribution = tuple((counts / simulations).tolist())

    return MonteCarloResult(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        simulations=simulations,
        seed=seed,
        home_win_probability=home_win_probability,
        draw_probability=draw_probability,
        away_win_probability=away_win_probability,
        home_win_ci95=_wilson_like_ci95(home_win_probability, simulations),
        draw_ci95=_wilson_like_ci95(draw_probability, simulations),
        away_win_ci95=_wilson_like_ci95(away_win_probability, simulations),
        mean_total_goals=mean_total_goals,
        total_goals_distribution=total_goals_distribution,
        at_least_one_goal_probability=at_least_one_goal_probability,
        at_least_one_goal_ci95=_wilson_like_ci95(at_least_one_goal_probability, simulations),
        metadata={
            "model": "independent_poisson_monte_carlo",
            "max_total_goals_bucket": str(MAX_TOTAL_GOALS_BUCKET),
            "default_simulations": str(DEFAULT_MONTE_CARLO_SIMULATIONS),
        },
    )


def monte_carlo_vs_poisson_sanity_check(
    monte_carlo_result: MonteCarloResult,
    poisson_home_win: float,
    poisson_draw: float,
    poisson_away_win: float,
    tolerance: float = 0.03,
) -> bool:
    """Check that Monte Carlo outcome probabilities agree with the closed-form Poisson model.

    Since both models share the same independent-Poisson assumption,
    their outcome probabilities should agree within Monte Carlo sampling
    error. This is a sanity check, not a proof of correctness of either
    model individually, and is not meant to force artificial agreement.

    Args:
        monte_carlo_result: result from run_monte_carlo_simulation().
        poisson_home_win: closed-form home win probability (e.g. from
            analytics.poisson.home_win_probability()).
        poisson_draw: closed-form draw probability.
        poisson_away_win: closed-form away win probability.
        tolerance: maximum allowed absolute difference per outcome.
            Defaults to 0.03 (3 percentage points), which comfortably
            exceeds the 95% confidence half-width for
            DEFAULT_MONTE_CARLO_SIMULATIONS-scale runs on typical
            football probabilities.

    Returns:
        True if all three outcome probabilities agree within tolerance.

    Raises:
        MonteCarloValidationError: if any probability input is invalid.
    """
    for name, value in (
        ("poisson_home_win", poisson_home_win),
        ("poisson_draw", poisson_draw),
        ("poisson_away_win", poisson_away_win),
    ):
        value = _validate_finite_number(name, value)
        if not (0.0 <= value <= 1.0):
            raise MonteCarloValidationError(f"{name} must be within [0, 1], got {value}")
    tolerance = _validate_finite_number("tolerance", tolerance)
    if tolerance < 0:
        raise MonteCarloValidationError(f"tolerance must be non-negative, got {tolerance}")

    return (
        abs(monte_carlo_result.home_win_probability - poisson_home_win) <= tolerance
        and abs(monte_carlo_result.draw_probability - poisson_draw) <= tolerance
        and abs(monte_carlo_result.away_win_probability - poisson_away_win) <= tolerance
    )


def calculate_model_agreement(
    poisson_home_win: float,
    poisson_draw: float,
    poisson_away_win: float,
    monte_carlo_result: MonteCarloResult,
) -> float:
    """Compute a [0, 1] agreement score between the Poisson and Monte Carlo models.

    agreement = 1 - mean(|poisson_i - monte_carlo_i|) over the three
    mutually exclusive outcomes (home/draw/away). A score of 1.0 means
    perfect agreement; lower scores indicate genuine model disagreement
    and are not artificially forced closer together.

    Args:
        poisson_home_win: closed-form home win probability.
        poisson_draw: closed-form draw probability.
        poisson_away_win: closed-form away win probability.
        monte_carlo_result: result from run_monte_carlo_simulation().

    Returns:
        An agreement score in [0, 1].

    Raises:
        MonteCarloValidationError: if any probability input is invalid.
    """
    for name, value in (
        ("poisson_home_win", poisson_home_win),
        ("poisson_draw", poisson_draw),
        ("poisson_away_win", poisson_away_win),
    ):
        value = _validate_finite_number(name, value)
        if not (0.0 <= value <= 1.0):
            raise MonteCarloValidationError(f"{name} must be within [0, 1], got {value}")

    mean_abs_diff = (
        abs(monte_carlo_result.home_win_probability - poisson_home_win)
        + abs(monte_carlo_result.draw_probability - poisson_draw)
        + abs(monte_carlo_result.away_win_probability - poisson_away_win)
    ) / 3.0
    return max(0.0, 1.0 - mean_abs_diff)
