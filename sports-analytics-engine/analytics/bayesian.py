"""Bayesian update interface for scoring-intensity model parameters.

ASSUMPTION / EXPLICIT DISCLOSURE: the source Vanguard specification
does not define a specific prior/likelihood/posterior formulation for
updating scoring intensity from live evidence. No such formula exists
in the project's authoritative documents. Rather than inventing an ad
hoc heuristic and presenting it as an established Vanguard formula,
this module implements the standard, textbook conjugate Gamma-Poisson
Bayesian update:

    Prior:      lambda ~ Gamma(shape=alpha, rate=beta)
    Likelihood: observed_goals ~ Poisson(lambda * observed_time_fraction)
    Posterior:  lambda ~ Gamma(shape=alpha + observed_goals,
                                rate=beta + observed_time_fraction)

This is the well-known conjugate prior relationship for a Poisson rate
parameter (not a Vanguard-specific invention), isolated behind a small,
clearly-named, replaceable interface as required. `observed_time_fraction`
is expressed in units of "full matches" (e.g. 0.5 for 45 minutes of a
90-minute match), so that the posterior mean remains directly
interpretable as a full-match-equivalent scoring intensity, consistent
with the rest of the Poisson engine's lambda convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class BayesianValidationError(ValueError):
    """Raised when an input to a Bayesian update is invalid."""


def _validate_finite_positive(name: str, value: float) -> float:
    """Validate that value is a finite, strictly positive real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        BayesianValidationError: if value is invalid or non-positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BayesianValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise BayesianValidationError(f"{name} must be finite, got {value}")
    if value <= 0:
        raise BayesianValidationError(f"{name} must be strictly positive, got {value}")
    return value


def _validate_finite_non_negative(name: str, value: float) -> float:
    """Validate that value is a finite, non-negative real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        BayesianValidationError: if value is invalid or negative.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BayesianValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise BayesianValidationError(f"{name} must be finite, got {value}")
    if value < 0:
        raise BayesianValidationError(f"{name} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class GammaPoissonPrior:
    """A Gamma(shape, rate) prior/posterior over a Poisson rate parameter.

    Attributes:
        shape: Gamma shape parameter (alpha). Strictly positive.
        rate: Gamma rate parameter (beta), in units of "full matches".
            Strictly positive.
    """

    shape: float
    rate: float

    def __post_init__(self) -> None:
        _validate_finite_positive("shape", self.shape)
        _validate_finite_positive("rate", self.rate)

    @property
    def mean(self) -> float:
        """Return the posterior/prior mean scoring intensity: shape / rate."""
        return self.shape / self.rate


def update_gamma_poisson(
    prior: GammaPoissonPrior, observed_goals: float, observed_time_fraction: float
) -> GammaPoissonPrior:
    """Apply one conjugate Gamma-Poisson Bayesian update.

    posterior.shape = prior.shape + observed_goals
    posterior.rate  = prior.rate  + observed_time_fraction

    Args:
        prior: the current Gamma(shape, rate) belief about the scoring
            intensity.
        observed_goals: number of goals observed during the interval
            being incorporated. Non-negative, finite (need not be an
            integer -- fractional "expected goal" evidence is
            permitted by the conjugate update mathematics, though
            typical usage passes integer goal counts).
        observed_time_fraction: the interval's duration expressed as a
            fraction of a full match (e.g. 15 minutes of a 90-minute
            match = 15/90 = 0.1666...). Strictly positive.

    Returns:
        A new GammaPoissonPrior representing the updated posterior.

    Raises:
        BayesianValidationError: if any input is invalid.
    """
    observed_goals = _validate_finite_non_negative("observed_goals", observed_goals)
    observed_time_fraction = _validate_finite_positive(
        "observed_time_fraction", observed_time_fraction
    )
    return GammaPoissonPrior(
        shape=prior.shape + observed_goals,
        rate=prior.rate + observed_time_fraction,
    )


def posterior_mean_lambda(posterior: GammaPoissonPrior) -> float:
    """Return the full-match-equivalent scoring intensity implied by a posterior.

    Args:
        posterior: a GammaPoissonPrior, typically produced by
            update_gamma_poisson().

    Returns:
        The posterior mean, in the same "goals per full match" units
        used by analytics.poisson's lambda_full_match convention.
    """
    return posterior.mean
