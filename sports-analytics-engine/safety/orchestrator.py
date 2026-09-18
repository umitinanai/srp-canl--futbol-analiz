"""Minimal system orchestrator coordinating Stage 2B under Stage 3 control.

This module owns NO analytics mathematics and duplicates NO Stage 2B
logic: every event is dispatched to the real, unmodified
data.pipeline.process_event(). Its only responsibilities are:

    - lifecycle sequencing (start/stop) around provider iteration
    - checking the kill switch before dispatching each new event
    - checking (and skipping) quarantined fixtures
    - classifying exceptions raised by process_event() into either a
      fixture-scoped quarantine decision or a system-scoped lifecycle
      failure, recording an ErrorRecord either way, and never silently
      swallowing an exception

Dependency direction: this module imports from data.*, analytics.*, and
models.* (lower layers). Nothing in data/, analytics/, backtesting/, or
models/ imports from safety/ -- Stage 3 depends on the lower layers,
never the reverse.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Deque, Mapping, Optional

from analytics.bayesian import GammaPoissonPrior
from analytics.monte_carlo import DEFAULT_MONTE_CARLO_SIMULATIONS, MonteCarloValidationError
from analytics.poisson import PoissonValidationError
from analytics.pressure import PressureValidationError
from analytics.xg_proxy import XGProxyValidationError
from data.normalizer import NormalizationError
from data.pipeline import PipelineValidationError, RecalculationOutcome, process_event
from data.provider import LiveDataProvider, ProviderConnectionState, ProviderError
from data.state_manager import (
    DEFAULT_RECALCULATION_POLICY,
    RecalculationPolicy,
    StateManager,
    StateManagerValidationError,
)
from safety.kill_switch import KillSwitch, KillSwitchActivatedError
from safety.lifecycle import LifecycleState, SystemLifecycle
from safety.observability import ErrorRecord, SystemStatus
from safety.quarantine import Quarantine, QuarantineReason

#: Bounded retention for the orchestrator's own diagnostic error
#: history (distinct from, and much smaller than, any quarantine or
#: analytics collection). Only the single most recent error is used
#: for SystemStatus.last_error; this bounded history exists purely for
#: additional operational diagnosis without growing without limit.
DEFAULT_ERROR_HISTORY_MAXLEN = 64


class OrchestratorNotRunningError(RuntimeError):
    """Raised when processing is attempted while the orchestrator is not RUNNING."""


class Orchestrator:
    """Coordinates provider ingestion into Stage 2B's process_event(), under control.

    Owns a SystemLifecycle, a KillSwitch, and a Quarantine -- each a
    small, independent, concurrency-safe component. The orchestrator
    itself adds no additional global lock: per-fixture concurrency is
    exactly what StateManager (used internally by process_event())
    already provides, and is not narrowed here.
    """

    def __init__(
        self,
        manager: StateManager,
        regulation_minutes: float,
        pressure_weights: Mapping[str, float],
        policy: RecalculationPolicy = DEFAULT_RECALCULATION_POLICY,
        monte_carlo_simulations: int = DEFAULT_MONTE_CARLO_SIMULATIONS,
        monte_carlo_seed: Optional[int] = None,
        quarantine_max_records: int = 256,
        error_history_maxlen: int = DEFAULT_ERROR_HISTORY_MAXLEN,
    ) -> None:
        """Initialize the orchestrator around an existing StateManager.

        Args:
            manager: the StateManager that process_event() will operate
                on. Not owned exclusively -- the orchestrator never
                reaches into its internals, only calls process_event()
                with it.
            regulation_minutes: forwarded to process_event() on every call.
            pressure_weights: forwarded to process_event() on every call.
            policy: the RecalculationPolicy forwarded to process_event().
            monte_carlo_simulations: forwarded to process_event(); never
                below analytics.monte_carlo.MINIMUM_SIMULATIONS (50,000),
                since process_event() -> run_full_recalculation() ->
                run_monte_carlo_simulation() enforces that floor itself.
            monte_carlo_seed: optional deterministic Monte Carlo seed,
                forwarded to process_event().
            quarantine_max_records: bound forwarded to Quarantine().
            error_history_maxlen: bound on the orchestrator's own
                diagnostic error history.
        """
        self.lifecycle = SystemLifecycle()
        self.kill_switch = KillSwitch()
        self.quarantine = Quarantine(max_records=quarantine_max_records)

        self._manager = manager
        self._regulation_minutes = regulation_minutes
        self._pressure_weights = pressure_weights
        self._policy = policy
        self._monte_carlo_simulations = monte_carlo_simulations
        self._monte_carlo_seed = monte_carlo_seed

        self._last_error: Optional[ErrorRecord] = None
        self._error_history: Deque[ErrorRecord] = deque(maxlen=error_history_maxlen)
        self._provider_state: Optional[ProviderConnectionState] = None

    @property
    def last_error(self) -> Optional[ErrorRecord]:
        """Return the most recently recorded ErrorRecord, or None."""
        return self._last_error

    def error_history(self) -> tuple:
        """Return a bounded, oldest-first tuple of recently recorded errors."""
        return tuple(self._error_history)

    async def _record_error(
        self, error_type: str, message: str, source: str, fixture_id: Optional[str] = None
    ) -> ErrorRecord:
        """Record a failure for observability, without raising.

        Args:
            error_type: short machine-readable category.
            message: human-readable description.
            source: originating layer/component.
            fixture_id: affected fixture, if fixture-scoped.

        Returns:
            The ErrorRecord that was stored.
        """
        record = ErrorRecord(
            error_type=error_type, message=message, timestamp=time.time(),
            source=source, fixture_id=fixture_id,
        )
        self._last_error = record
        self._error_history.append(record)
        return record

    async def start(self) -> None:
        """Transition CREATED -> STARTING -> RUNNING.

        Raises:
            LifecycleTransitionError: if the orchestrator is not in a
                state from which starting is valid (e.g. already
                RUNNING or STOPPED).
        """
        try:
            await self.lifecycle.transition_to(LifecycleState.STARTING)
            await self.lifecycle.transition_to(LifecycleState.RUNNING)
        except Exception as exc:
            await self._record_error("startup_failure", str(exc), source="lifecycle")
            raise

    async def stop(self) -> None:
        """Idempotently stop the orchestrator (delegates to SystemLifecycle.stop())."""
        await self.lifecycle.stop()

    async def process(
        self,
        fixture_id: str,
        raw_event: Mapping[str, Any],
        event_timestamp: float,
        home_prior: GammaPoissonPrior,
        away_prior: GammaPoissonPrior,
        event_id: Optional[str] = None,
        received_timestamp: Optional[float] = None,
    ) -> Optional[RecalculationOutcome]:
        """Dispatch one raw event to the real Stage 2B process_event(), under control.

        Control checks (in order): kill switch, lifecycle RUNNING,
        fixture quarantine. If any of these blocks processing, this
        method either raises (kill switch / not running -- caller
        errors) or returns None (fixture quarantined -- deterministic,
        silent skip of an already-isolated fixture, not a new failure).

        Args:
            fixture_id: the fixture identifier.
            raw_event: a raw, provider-shaped event dict.
            event_timestamp: unix epoch seconds when the event occurred.
            home_prior: home side's Gamma-Poisson prior, forwarded to
                process_event() unchanged.
            away_prior: away side's Gamma-Poisson prior, forwarded to
                process_event() unchanged.
            event_id: optional provider-supplied event identifier.
            received_timestamp: unix epoch seconds this event was
                received locally.

        Returns:
            The RecalculationOutcome from the real process_event() call,
            or None if the fixture is currently quarantined (processing
            deliberately skipped, not attempted).

        Raises:
            KillSwitchActivatedError: if the kill switch has been activated.
            OrchestratorNotRunningError: if the lifecycle is not RUNNING.
            asyncio.CancelledError: always propagated, never converted
                into a quarantine decision or an ordinary error record.
        """
        self.kill_switch.check()
        if self.lifecycle.state != LifecycleState.RUNNING:
            raise OrchestratorNotRunningError(
                f"Orchestrator is not RUNNING (state={self.lifecycle.state.value})"
            )
        if self.quarantine.is_quarantined(fixture_id):
            return None

        try:
            return process_event(
                self._manager, fixture_id, raw_event, event_timestamp,
                home_prior, away_prior, self._regulation_minutes, self._pressure_weights,
                event_id=event_id, received_timestamp=received_timestamp,
                policy=self._policy, monte_carlo_simulations=self._monte_carlo_simulations,
                monte_carlo_seed=self._monte_carlo_seed,
            )
        except asyncio.CancelledError:
            # Never converted into a quarantine decision or error record
            # as if it were an ordinary failure -- cancellation is not a
            # fixture-processing defect.
            raise
        except NormalizationError as exc:
            await self._record_error("malformed_event", str(exc), source="normalizer", fixture_id=fixture_id)
            await self.quarantine.quarantine_fixture(fixture_id, QuarantineReason.MALFORMED_EVENT, str(exc))
            return None
        except StateManagerValidationError as exc:
            await self._record_error("state_processing_failure", str(exc), source="state", fixture_id=fixture_id)
            await self.quarantine.quarantine_fixture(
                fixture_id, QuarantineReason.STATE_PROCESSING_FAILURE, str(exc)
            )
            return None
        except (
            PoissonValidationError, MonteCarloValidationError,
            PressureValidationError, XGProxyValidationError, PipelineValidationError,
        ) as exc:
            await self._record_error("analytics_failure", str(exc), source="analytics", fixture_id=fixture_id)
            await self.quarantine.quarantine_fixture(fixture_id, QuarantineReason.ANALYTICS_FAILURE, str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - deliberately broad: see docstring
            # An unclassified failure does not, on its own, justify an
            # invented quarantine rule (see safety.quarantine module
            # docstring) -- it is recorded for observability and
            # re-raised rather than silently absorbed as a fixture
            # isolation decision or a false success.
            await self._record_error("unexpected_exception", str(exc), source="orchestrator", fixture_id=fixture_id)
            raise

    async def run_provider(
        self,
        provider: LiveDataProvider,
        fixture_id: str,
        home_prior: GammaPoissonPrior,
        away_prior: GammaPoissonPrior,
    ) -> None:
        """Consume a provider's event stream for one fixture through process().

        This is the real Provider -> Stage 3 -> process_event()
        integration point. A provider-level failure (ProviderError) is
        a system/provider-scoped failure: it is recorded and the
        lifecycle is marked FAILED, but it is explicitly NOT turned
        into a fixture quarantine decision (see safety.quarantine
        module docstring / QuarantineReason).

        Args:
            provider: the LiveDataProvider to stream events from.
            fixture_id: the fixture identifier to pass through to
                provider.stream_events() and process().
            home_prior: forwarded to process() on every event.
            away_prior: forwarded to process() on every event.

        Raises:
            ProviderError: propagated after being recorded and after
                marking the lifecycle FAILED.
            asyncio.CancelledError: always propagated.
        """
        try:
            async for raw_event in provider.stream_events(fixture_id):
                self._provider_state = provider.connection_state
                if self.kill_switch.is_activated:
                    break
                await self.process(fixture_id, raw_event, time.time(), home_prior, away_prior)
                self._provider_state = provider.connection_state
        except asyncio.CancelledError:
            raise
        except ProviderError as exc:
            await self._record_error("provider_failure", str(exc), source="provider", fixture_id=fixture_id)
            self._provider_state = provider.connection_state
            await self.lifecycle.mark_failed("provider_failure", str(exc))
            raise

    def status(self) -> SystemStatus:
        """Return a typed, deterministic snapshot of the current control surface."""
        return SystemStatus(
            lifecycle_state=self.lifecycle.state,
            provider_state=self._provider_state,
            kill_switch_state=self.kill_switch.state,
            quarantined_fixture_count=len(self.quarantine),
            last_error=self._last_error,
        )
