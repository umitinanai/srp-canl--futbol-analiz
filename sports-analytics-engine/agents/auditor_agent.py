"""Stage 4 Auditor Agent: independently re-verifies a QuantResult.

"Independent" means independently INVOKING the same already-validated
analytical functions the original computation used, from the same
immutable inputs (an AnalyticsSnapshot plus the priors/config that
produced the QuantResult being audited) -- never re-deriving a
competing formula. Concretely, this agent calls
data.pipeline.run_full_recalculation() (the same authoritative
recomputation entry point Stage 2B itself uses) a second time and
compares its output against the QuantResult under audit, plus a small
set of self-consistency checks on the audited QuantResult's own fields
that require no recomputation at all.

Deterministic, evidence-backed minimum check set (each check is added
to `checks_passed` ONLY when the fields it needs are actually present
on the inputs -- a check that cannot be reconstructed from available
data is never faked):

    - fixture_id_matches: quant_result.fixture_id == snapshot.fixture_id.
    - probabilities_sum_to_one: quant_result's own closed-form
      home/draw/away win probabilities sum to ~1.0 (a documented
      property of analytics.poisson.score_probability_matrix()).
    - monte_carlo_sums_to_one: quant_result's own Monte Carlo
      home/draw/away win probabilities sum to ~1.0.
    - monte_carlo_agrees_with_poisson: quant_result's own Monte Carlo
      probabilities agree with its own closed-form probabilities within
      a tolerance. This mirrors analytics.monte_carlo.
      monte_carlo_vs_poisson_sanity_check()'s own comparison and default
      3% tolerance; that function itself cannot be called here because
      it requires a full MonteCarloResult object, which QuantResult does
      not retain (only three summary floats) -- constructing a synthetic
      MonteCarloResult to satisfy the signature would be fabrication, so
      the same comparison is applied directly to the floats that do
      exist, without inventing a new formula.
    - lambda_matches_recomputation / poisson_probabilities_match_recomputation:
      compares the DETERMINISTIC (non-stochastic) fields of an
      independently-recomputed QuantResult (via run_full_recalculation(),
      called with the same snapshot/priors/config) against the audited
      one. These fields involve no randomness, so they are expected to
      match exactly given identical inputs.

If NONE of the above checks could be evaluated (e.g. the audited
QuantResult carries no probabilities at all), the verdict is
AuditVerdict.UNDEFINED -- the existing contract's own explicit "I could
not determine this" value -- rather than a fabricated VALID/INVALID.

Safety interaction (preflight correction G): an INVALID verdict may
place the fixture into the EXISTING safety.quarantine.Quarantine using
the EXISTING QuarantineReason.ANALYTICS_FAILURE member (already used by
safety.orchestrator.Orchestrator for exactly this class of problem --
an analytics result failing validation). No new quarantine reason, no
kill-switch activation rule, and no invented "N invalid audits" or
rolling-rate policy is introduced anywhere in this module.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Mapping, Optional

from analytics.bayesian import GammaPoissonPrior
from analytics.monte_carlo import DEFAULT_MONTE_CARLO_SIMULATIONS
from data.pipeline import run_full_recalculation
from data.state_manager import AnalyticsSnapshot
from models.events import JournalEvent, JournalEventType
from models.metrics import AuditResult, AuditVerdict, QuantResult
from safety.quarantine import Quarantine, QuarantineReason

_DEFAULT_TOLERANCE = 1e-6
_DEFAULT_MONTE_CARLO_TOLERANCE = 0.03
_DEFAULT_CACHE_MAXLEN = 256


@dataclass(frozen=True)
class _ProbabilityTriple:
    home: float
    draw: float
    away: float


def _extract_probability_triple(
    home: Optional[float], draw: Optional[float], away: Optional[float]
) -> Optional[_ProbabilityTriple]:
    if home is None or draw is None or away is None:
        return None
    return _ProbabilityTriple(home=home, draw=draw, away=away)


class AuditorAgent:
    """Independently re-verifies QuantResults using only existing analytics calls.

    Owns a small, bounded audit cache keyed by payload_checksum (reusing
    the existing checksum concept from models/match.py's Snapshot
    docstring -- "Section 22 audit cache") so an identical payload is
    never re-audited from scratch.
    """

    def __init__(
        self,
        journal: Optional[object] = None,
        quarantine: Optional[Quarantine] = None,
        tolerance: float = _DEFAULT_TOLERANCE,
        monte_carlo_tolerance: float = _DEFAULT_MONTE_CARLO_TOLERANCE,
        cache_maxlen: int = _DEFAULT_CACHE_MAXLEN,
    ) -> None:
        """Initialize the Auditor Agent.

        Args:
            journal: optional agents.journal_agent.JournalAgent (or any
                object exposing the same async record_audit_result()/
                record_journal_event() methods), duck-typed.
            quarantine: optional EXISTING safety.quarantine.Quarantine
                instance. When provided, an INVALID verdict quarantines
                the fixture via its existing quarantine_fixture() API
                using the existing ANALYTICS_FAILURE reason. Never
                constructs its own Quarantine.
            tolerance: absolute tolerance for deterministic
                (non-stochastic) field comparisons.
            monte_carlo_tolerance: absolute tolerance for the
                Monte-Carlo-vs-Poisson self-consistency check, matching
                analytics.monte_carlo.monte_carlo_vs_poisson_sanity_check()'s
                own default.
            cache_maxlen: bound on the number of distinct
                payload_checksums retained in the audit cache.
        """
        self._journal = journal
        self._quarantine = quarantine
        self._tolerance = tolerance
        self._mc_tolerance = monte_carlo_tolerance
        self._cache: Dict[str, AuditResult] = {}
        self._cache_order: Deque[str] = deque()
        self._cache_maxlen = cache_maxlen

    def cached_result(self, payload_checksum: str) -> Optional[AuditResult]:
        """Return a previously-cached AuditResult for payload_checksum, if any."""
        return self._cache.get(payload_checksum)

    async def audit(
        self,
        quant_result: QuantResult,
        snapshot: AnalyticsSnapshot,
        home_prior: GammaPoissonPrior,
        away_prior: GammaPoissonPrior,
        regulation_minutes: float,
        pressure_weights: Mapping[str, float],
        payload_checksum: str = "",
        monte_carlo_simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
        monte_carlo_seed: Optional[int] = None,
    ) -> AuditResult:
        """Independently audit one QuantResult.

        Args:
            quant_result: the QuantResult to audit. Never mutated.
            snapshot: the AnalyticsSnapshot quant_result was derived
                from (immutable; the same boundary object
                data.pipeline itself consumes).
            home_prior: the SAME home-side GammaPoissonPrior used to
                produce quant_result, required for a faithful
                independent recomputation.
            away_prior: the SAME away-side GammaPoissonPrior used to
                produce quant_result.
            regulation_minutes: the SAME value used to produce quant_result.
            pressure_weights: the SAME weights used to produce quant_result.
            payload_checksum: optional cache key (e.g. a
                models.match.Snapshot.payload_checksum) enabling the
                audit cache. Audits are not cached when empty.
            monte_carlo_simulations: forwarded to the independent
                recomputation's Monte Carlo pass.
            monte_carlo_seed: forwarded to the independent
                recomputation's Monte Carlo pass. Note that unless this
                matches the seed originally used, the two Monte Carlo
                samples are independent draws -- exactly why Monte Carlo
                fields are checked via tolerance-based self-consistency
                rather than exact recomputation equality (see module
                docstring).

        Returns:
            An AuditResult. Never raises for a "the data disagrees"
            outcome -- that is AuditVerdict.INVALID, not an exception.
        """
        if payload_checksum:
            cached = self._cache.get(payload_checksum)
            if cached is not None:
                return cached

        checks: Dict[str, bool] = {}
        reasons: Dict[str, str] = {}

        checks["fixture_id_matches"] = quant_result.fixture_id == snapshot.fixture_id
        if not checks["fixture_id_matches"]:
            reasons["fixture_id_matches"] = (
                f"quant_result.fixture_id={quant_result.fixture_id!r} != "
                f"snapshot.fixture_id={snapshot.fixture_id!r}"
            )

        poisson_triple = _extract_probability_triple(
            quant_result.home_win_probability, quant_result.draw_probability, quant_result.away_win_probability
        )
        if poisson_triple is not None:
            total = poisson_triple.home + poisson_triple.draw + poisson_triple.away
            checks["probabilities_sum_to_one"] = math.isclose(total, 1.0, abs_tol=1e-6)
            if not checks["probabilities_sum_to_one"]:
                reasons["probabilities_sum_to_one"] = f"home+draw+away={total}"

        monte_carlo_triple = _extract_probability_triple(
            quant_result.monte_carlo_home_win, quant_result.monte_carlo_draw, quant_result.monte_carlo_away_win
        )
        if monte_carlo_triple is not None:
            mc_total = monte_carlo_triple.home + monte_carlo_triple.draw + monte_carlo_triple.away
            checks["monte_carlo_sums_to_one"] = math.isclose(mc_total, 1.0, abs_tol=1e-6)
            if not checks["monte_carlo_sums_to_one"]:
                reasons["monte_carlo_sums_to_one"] = f"mc home+draw+away={mc_total}"

            if poisson_triple is not None:
                checks["monte_carlo_agrees_with_poisson"] = (
                    abs(monte_carlo_triple.home - poisson_triple.home) <= self._mc_tolerance
                    and abs(monte_carlo_triple.draw - poisson_triple.draw) <= self._mc_tolerance
                    and abs(monte_carlo_triple.away - poisson_triple.away) <= self._mc_tolerance
                )
                if not checks["monte_carlo_agrees_with_poisson"]:
                    reasons["monte_carlo_agrees_with_poisson"] = (
                        "Monte Carlo outcome probabilities diverge from the closed-form "
                        f"Poisson probabilities beyond tolerance={self._mc_tolerance}"
                    )

        if quant_result.lambda_home is not None and quant_result.lambda_away is not None:
            recomputed = run_full_recalculation(
                snapshot, home_prior, away_prior, regulation_minutes, pressure_weights,
                monte_carlo_simulations=monte_carlo_simulations, monte_carlo_seed=monte_carlo_seed,
            )
            checks["lambda_matches_recomputation"] = math.isclose(
                recomputed.lambda_home, quant_result.lambda_home, abs_tol=self._tolerance
            ) and math.isclose(recomputed.lambda_away, quant_result.lambda_away, abs_tol=self._tolerance)
            if not checks["lambda_matches_recomputation"]:
                reasons["lambda_matches_recomputation"] = (
                    f"recomputed=({recomputed.lambda_home}, {recomputed.lambda_away}) "
                    f"audited=({quant_result.lambda_home}, {quant_result.lambda_away})"
                )

            if poisson_triple is not None:
                checks["poisson_probabilities_match_recomputation"] = (
                    math.isclose(recomputed.home_win_probability, poisson_triple.home, abs_tol=self._tolerance)
                    and math.isclose(recomputed.draw_probability, poisson_triple.draw, abs_tol=self._tolerance)
                    and math.isclose(recomputed.away_win_probability, poisson_triple.away, abs_tol=self._tolerance)
                )
                if not checks["poisson_probabilities_match_recomputation"]:
                    reasons["poisson_probabilities_match_recomputation"] = (
                        "independently recomputed closed-form probabilities diverge from the audited QuantResult"
                    )

        # fixture_id_matches is a structural check available whenever both
        # objects exist at all; it alone is not sufficient evidence to
        # call a result VALID (a QuantResult carrying no probabilities/
        # lambdas has nothing substantive verified about it, even though
        # its fixture_id happens to match). At least one of the
        # substantive, data-dependent checks above must have been
        # evaluable for a VALID/INVALID verdict; otherwise UNDEFINED.
        substantive_checks = {key: value for key, value in checks.items() if key != "fixture_id_matches"}
        if not substantive_checks:
            verdict = AuditVerdict.UNDEFINED
        elif all(checks.values()):
            verdict = AuditVerdict.VALID
        else:
            verdict = AuditVerdict.INVALID

        result = AuditResult(
            fixture_id=quant_result.fixture_id,
            timestamp=quant_result.timestamp,
            verdict=verdict,
            checks_passed=checks,
            reasons=reasons,
            payload_checksum=payload_checksum,
        )

        if payload_checksum:
            if len(self._cache_order) >= self._cache_maxlen:
                evicted = self._cache_order.popleft()
                self._cache.pop(evicted, None)
            self._cache[payload_checksum] = result
            self._cache_order.append(payload_checksum)

        if self._journal is not None:
            await self._journal.record_audit_result(result)
            await self._journal.record_journal_event(
                JournalEvent(
                    event_type=JournalEventType.AUDIT_RESULT,
                    fixture_id=result.fixture_id,
                    timestamp=result.timestamp,
                    details={"verdict": result.verdict.value},
                )
            )

        if verdict == AuditVerdict.INVALID and self._quarantine is not None:
            await self._quarantine.quarantine_fixture(
                quant_result.fixture_id,
                QuarantineReason.ANALYTICS_FAILURE,
                "audit verdict INVALID: " + "; ".join(f"{k}={v}" for k, v in reasons.items()),
            )

        return result
