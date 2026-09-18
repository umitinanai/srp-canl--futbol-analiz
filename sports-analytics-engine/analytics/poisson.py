"""Poisson scoring-intensity model for football score prediction.

Implements the independent-Poisson approach: home and away goals are
modeled as independent Poisson random variables with intensities
(lambdas) derived from scoring rates, and the joint score distribution
is their outer product:

    P(H=h, A=a) = P(H=h) * P(A=a)

All public functions here are pure, deterministic and side-effect free.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

#: Default truncation point for score probability matrices. Chosen so
#: that for realistic football scoring intensities (lambda roughly up
#: to ~6), the truncated Poisson tail beyond this many goals is
#: negligible (< 1e-6), keeping score_probability_matrix's row/column
#: sums close to 1 within the tolerances used by callers and tests.
DEFAULT_MAX_GOALS = 15


class PoissonValidationError(ValueError):
    """Raised when an input to a Poisson calculation is invalid."""


def _validate_finite_number(name: str, value: float) -> float:
    """Validate that value is a finite, non-NaN real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        PoissonValidationError: if value is not numeric, is NaN, or is
            infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoissonValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value):
        raise PoissonValidationError(f"{name} is NaN")
    if math.isinf(value):
        raise PoissonValidationError(f"{name} is infinite")
    return value


def _validate_lambda(name: str, lam: float) -> float:
    """Validate a Poisson intensity (lambda): finite and non-negative.

    Args:
        name: name of the parameter, used in error messages.
        lam: the lambda value to validate.

    Returns:
        lam, coerced to float.

    Raises:
        PoissonValidationError: if lam is invalid (see
            _validate_finite_number) or negative.
    """
    lam = _validate_finite_number(name, lam)
    if lam < 0:
        raise PoissonValidationError(f"{name} must be non-negative, got {lam}")
    return lam


def _validate_non_negative_int(name: str, value: int) -> int:
    """Validate that value is a non-negative integer.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value as an int.

    Raises:
        PoissonValidationError: if value is not an integer or is negative.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise PoissonValidationError(f"{name} must be an int, got {type(value)!r}")
    if value < 0:
        raise PoissonValidationError(f"{name} must be non-negative, got {value}")
    return value


def poisson_pmf(k: int, lam: float) -> float:
    """Compute the Poisson probability mass function P(X = k).

    Args:
        k: number of goals (non-negative integer).
        lam: Poisson intensity (non-negative, finite).

    Returns:
        P(X = k) for X ~ Poisson(lam).

    Raises:
        PoissonValidationError: if k or lam is invalid.
    """
    k = _validate_non_negative_int("k", k)
    lam = _validate_lambda("lam", lam)

    if lam == 0.0:
        return 1.0 if k == 0 else 0.0

    log_pmf = -lam + k * math.log(lam) - math.lgamma(k + 1)
    return math.exp(log_pmf)


def poisson_probability(k: int, lam: float) -> float:
    """Alias of poisson_pmf(), matching the master specification's naming.

    Args:
        k: number of goals (non-negative integer).
        lam: Poisson intensity (non-negative, finite).

    Returns:
        P(X = k) for X ~ Poisson(lam).

    Raises:
        PoissonValidationError: if k or lam is invalid.
    """
    return poisson_pmf(k, lam)


def _poisson_pmf_vector(lam: float, max_goals: int) -> np.ndarray:
    """Compute P(X=0), P(X=1), ..., P(X=max_goals) for X ~ Poisson(lam).

    Uses the stable recurrence P(k) = P(k-1) * lam / k, computed via a
    single vectorized cumulative product, avoiding both large
    intermediate factorials and any Python-level loop over goal counts.

    Args:
        lam: Poisson intensity (non-negative, finite). Assumed already
            validated by the caller.
        max_goals: highest goal count included (inclusive). Assumed
            already validated as a non-negative int by the caller.

    Returns:
        A numpy array of length max_goals + 1 with pmf[k] = P(X=k).
    """
    pmf = np.empty(max_goals + 1, dtype=np.float64)
    pmf[0] = math.exp(-lam)
    if max_goals >= 1:
        k = np.arange(1, max_goals + 1, dtype=np.float64)
        ratios = lam / k
        pmf[1:] = pmf[0] * np.cumprod(ratios)
    return pmf
    """Compute P(X=0), P(X=1), ..., P(X=max_goals) for X ~ Poisson(lam).

    Uses the stable recurrence P(k) = P(k-1) * lam / k, computed via a
    single vectorized cumulative product, avoiding both large
    intermediate factorials and any Python-level loop over goal counts.

    Args:
        lam: Poisson intensity (non-negative, finite). Assumed already
            validated by the caller.
        max_goals: highest goal count included (inclusive). Assumed
            already validated as a non-negative int by the caller.

    Returns:
        A numpy array of length max_goals + 1 with pmf[k] = P(X=k).
    """
    pmf = np.empty(max_goals + 1, dtype=np.float64)
    pmf[0] = math.exp(-lam)
    if max_goals >= 1:
        k = np.arange(1, max_goals + 1, dtype=np.float64)
        ratios = lam / k
        pmf[1:] = pmf[0] * np.cumprod(ratios)
    return pmf


def calculate_remaining_lambda(
    lambda_full_match: float,
    elapsed_minutes: float,
    regulation_minutes: float = 90.0,
) -> float:
    """Scale a full-match scoring intensity down to the remaining time.

    lambda_remaining = lambda_full_match * remaining_minutes / regulation_minutes

    If elapsed_minutes >= regulation_minutes (including exactly at full
    time, or beyond due to stoppage time), remaining_minutes is clamped
    to 0 and this function deterministically returns 0.0 -- there is no
    more match time left in which further goals (under this model) can
    occur.

    Args:
        lambda_full_match: full 90-minute (or full-regulation) scoring
            intensity. Must be finite and non-negative.
        elapsed_minutes: minutes of the match already played. Must be
            finite and non-negative.
        regulation_minutes: total regulation length of the match in
            minutes. Must be finite and strictly positive.

    Returns:
        The remaining-time scoring intensity, always >= 0.

    Raises:
        PoissonValidationError: if any input is invalid, including
            negative elapsed_minutes or non-positive regulation_minutes.
    """
    lambda_full_match = _validate_lambda("lambda_full_match", lambda_full_match)
    elapsed_minutes = _validate_finite_number("elapsed_minutes", elapsed_minutes)
    if elapsed_minutes < 0:
        raise PoissonValidationError(
            f"elapsed_minutes must be non-negative, got {elapsed_minutes}"
        )
    regulation_minutes = _validate_finite_number("regulation_minutes", regulation_minutes)
    if regulation_minutes <= 0:
        raise PoissonValidationError(
            f"regulation_minutes must be strictly positive, got {regulation_minutes}"
        )

    remaining_minutes = regulation_minutes - elapsed_minutes
    if remaining_minutes <= 0:
        return 0.0

    return lambda_full_match * remaining_minutes / regulation_minutes


def score_probability_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """Build the joint score probability matrix under independent Poisson goals.

    matrix[h, a] = P(home_goals = h) * P(away_goals = a)

    Args:
        lambda_home: home team scoring intensity (finite, non-negative).
        lambda_away: away team scoring intensity (finite, non-negative).
        max_goals: highest goal count (per team) included in the
            truncated matrix. Must be a non-negative int.

    Returns:
        A (max_goals + 1) x (max_goals + 1) numpy array. Rows index
        home goals, columns index away goals. The matrix sums to
        approximately 1.0 (exactly 1.0 in the limit max_goals -> inf).

    Raises:
        PoissonValidationError: if any input is invalid.
    """
    lambda_home = _validate_lambda("lambda_home", lambda_home)
    lambda_away = _validate_lambda("lambda_away", lambda_away)
    max_goals = _validate_non_negative_int("max_goals", max_goals)

    home_pmf = _poisson_pmf_vector(lambda_home, max_goals)
    away_pmf = _poisson_pmf_vector(lambda_away, max_goals)
    return np.outer(home_pmf, away_pmf)


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Validate that matrix is a finite, non-negative, square 2D array.

    Args:
        matrix: candidate score probability matrix.

    Returns:
        matrix, unchanged.

    Raises:
        PoissonValidationError: if matrix is not a valid probability matrix.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise PoissonValidationError("matrix must be a 2D numpy array")
    if matrix.shape[0] != matrix.shape[1]:
        raise PoissonValidationError("matrix must be square (home x away)")
    if not np.all(np.isfinite(matrix)):
        raise PoissonValidationError("matrix contains NaN or infinite values")
    if np.any(matrix < 0):
        raise PoissonValidationError("matrix contains negative probabilities")
    return matrix


def home_win_probability(matrix: np.ndarray) -> float:
    """Compute P(home_goals > away_goals) from a score probability matrix.

    Args:
        matrix: a score probability matrix as produced by
            score_probability_matrix().

    Returns:
        The probability of a home win, in [0, 1].

    Raises:
        PoissonValidationError: if matrix is invalid.
    """
    matrix = _validate_matrix(matrix)
    return float(np.tril(matrix, k=-1).sum())


def draw_probability(matrix: np.ndarray) -> float:
    """Compute P(home_goals == away_goals) from a score probability matrix.

    Args:
        matrix: a score probability matrix as produced by
            score_probability_matrix().

    Returns:
        The probability of a draw, in [0, 1].

    Raises:
        PoissonValidationError: if matrix is invalid.
    """
    matrix = _validate_matrix(matrix)
    return float(np.trace(matrix))


def away_win_probability(matrix: np.ndarray) -> float:
    """Compute P(away_goals > home_goals) from a score probability matrix.

    Args:
        matrix: a score probability matrix as produced by
            score_probability_matrix().

    Returns:
        The probability of an away win, in [0, 1].

    Raises:
        PoissonValidationError: if matrix is invalid.
    """
    matrix = _validate_matrix(matrix)
    return float(np.triu(matrix, k=1).sum())


def total_goals_distribution(matrix: np.ndarray) -> np.ndarray:
    """Compute the distribution of total goals (home + away) from a matrix.

    Args:
        matrix: a score probability matrix as produced by
            score_probability_matrix().

    Returns:
        A 1D numpy array of length (2 * max_goals + 1) where index t
        holds P(home_goals + away_goals == t).

    Raises:
        PoissonValidationError: if matrix is invalid.
    """
    matrix = _validate_matrix(matrix)
    n = matrix.shape[0]
    flipped = np.fliplr(matrix)
    totals = np.array(
        [np.trace(flipped, offset=(n - 1) - t) for t in range(2 * n - 1)],
        dtype=np.float64,
    )
    return totals


def _validate_total_goals_distribution(distribution: np.ndarray) -> np.ndarray:
    """Validate a total-goals distribution array.

    Args:
        distribution: candidate distribution array.

    Returns:
        distribution, unchanged.

    Raises:
        PoissonValidationError: if distribution is invalid.
    """
    if not isinstance(distribution, np.ndarray) or distribution.ndim != 1:
        raise PoissonValidationError("distribution must be a 1D numpy array")
    if not np.all(np.isfinite(distribution)):
        raise PoissonValidationError("distribution contains NaN or infinite values")
    if np.any(distribution < 0):
        raise PoissonValidationError("distribution contains negative probabilities")
    return distribution


def over_probability(distribution: np.ndarray, threshold: float) -> float:
    """Compute P(total_goals > threshold) from a total-goals distribution.

    Args:
        distribution: a total-goals distribution as produced by
            total_goals_distribution().
        threshold: the goals line (e.g. 2.5 for an "Over 2.5" market).
            Must be finite.

    Returns:
        The probability that total goals strictly exceed threshold.

    Raises:
        PoissonValidationError: if distribution or threshold is invalid.
    """
    distribution = _validate_total_goals_distribution(distribution)
    threshold = _validate_finite_number("threshold", threshold)
    goal_counts = np.arange(distribution.shape[0], dtype=np.float64)
    mask = goal_counts > threshold
    return float(distribution[mask].sum())


def under_probability(distribution: np.ndarray, threshold: float) -> float:
    """Compute P(total_goals < threshold) from a total-goals distribution.

    Args:
        distribution: a total-goals distribution as produced by
            total_goals_distribution().
        threshold: the goals line (e.g. 2.5 for an "Under 2.5" market).
            Must be finite.

    Returns:
        The probability that total goals are strictly below threshold.

    Raises:
        PoissonValidationError: if distribution or threshold is invalid.
    """
    distribution = _validate_total_goals_distribution(distribution)
    threshold = _validate_finite_number("threshold", threshold)
    goal_counts = np.arange(distribution.shape[0], dtype=np.float64)
    mask = goal_counts < threshold
    return float(distribution[mask].sum())


def outcome_probabilities(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Convenience helper: compute (home, draw, away) probabilities together.

    Args:
        matrix: a score probability matrix as produced by
            score_probability_matrix().

    Returns:
        A (home_win_probability, draw_probability, away_win_probability)
        tuple.

    Raises:
        PoissonValidationError: if matrix is invalid.
    """
    matrix = _validate_matrix(matrix)
    return (
        home_win_probability(matrix),
        draw_probability(matrix),
        away_win_probability(matrix),
    )


def probability_at_least_one_additional_goal(
    lambda_home_remaining: float, lambda_away_remaining: float
) -> float:
    """Compute the probability of at least one further goal from this point on.

    This is GoalEdge's primary analytical target: given remaining-time
    scoring intensities for both sides, the total additional goals
    (home + away) over the rest of the match is itself Poisson with
    intensity lambda_home_remaining + lambda_away_remaining (sum of
    independent Poisson variables is Poisson). The probability of zero
    additional goals is P(X=0) = exp(-lambda_total), so:

        P(at least one goal) = 1 - exp(-(lambda_home + lambda_away))

    Args:
        lambda_home_remaining: home team's remaining-time scoring
            intensity, e.g. from calculate_remaining_lambda(). Finite,
            non-negative.
        lambda_away_remaining: away team's remaining-time scoring
            intensity. Finite, non-negative.

    Returns:
        The probability, in [0, 1], of at least one more goal (by
        either side) before the end of regulation. Returns exactly 0.0
        when both remaining lambdas are 0.0 (e.g. at full time).

    Raises:
        PoissonValidationError: if either input is invalid.
    """
    lambda_home_remaining = _validate_lambda("lambda_home_remaining", lambda_home_remaining)
    lambda_away_remaining = _validate_lambda("lambda_away_remaining", lambda_away_remaining)
    lambda_total = lambda_home_remaining + lambda_away_remaining
    return 1.0 - math.exp(-lambda_total)
