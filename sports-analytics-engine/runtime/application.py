"""Stage 5 application orchestrator: `ApplicationRuntime`.

Composes the already-existing, already-tested Stage 1-4 components into
a runnable multi-fixture application. Contains no analytics math, no
SQL, no normalization, and no safety-policy invention -- every
responsibility here is construction, sequencing, and routing of
components that already exist.

GLOBAL VS FIXTURE-LOCAL DESIGN (the central architectural decision of
this module): `safety.orchestrator.Orchestrator` owns its OWN
`SystemLifecycle`, `KillSwitch`, and `Quarantine` internally (they are
constructed inside `Orchestrator.__init__`, not injectable). If this
runtime shared ONE `Orchestrator` instance across every registered
fixture, a single fixture's provider connection exhausting its
reconnect budget would call that ONE shared `lifecycle.mark_failed()`,
which would then make `Orchestrator.process()` raise
`OrchestratorNotRunningError` for EVERY other fixture too -- an
ordinary, isolated provider hiccup on one match would silently stop
processing for every unrelated match sharing the runtime. That is
explicitly disallowed by this task.

The fix is NOT to duplicate the KillSwitch per fixture (that would make
the "global emergency stop" not actually global) and NOT to invent a
new safety subsystem. Instead: this runtime constructs ONE
`safety.orchestrator.Orchestrator` (with its own `StateManager`,
`SystemLifecycle`, `Quarantine`, and an internal, deliberately-UNUSED
local `KillSwitch`) PER REGISTERED FIXTURE, giving fixture-local
lifecycle/quarantine isolation "for free" from existing Stage 3 code.
Separately, this runtime owns exactly ONE `safety.kill_switch.KillSwitch`
of its own -- `_global_kill_switch` -- checked (via the exact same
`KillSwitch.check()` / `KillSwitchActivatedError` API Stage 3 already
defines) at the top of every method that would start new work, BEFORE
any fixture-local Orchestrator is ever reached. Activating
`_global_kill_switch` therefore blocks new processing for every
fixture identically, matching the existing, tested "kill switch is a
global emergency stop" semantics -- while an ordinary provider/
malformed/state/analytics failure on one fixture only ever touches
that ONE fixture's own (never-shared) Orchestrator instance.

This is a minimal composition/adapter boundary, not a Stage 3 rewrite:
no line of `safety/*.py` changes, and both `KillSwitch` and
`Orchestrator` are used exactly as their existing public APIs define.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Set, Tuple

from agents.auditor_agent import AuditorAgent
from agents.data_agent import DataAgent
from agents.journal_agent import JournalAgent
from agents.market_agent import MarketAgent
from agents.quant_agent import QuantAgent
from agents.risk_agent import RiskAgent
from analytics.bayesian import GammaPoissonPrior
from config.leagues import get_league
from config.settings import Settings
from data.normalizer import build_snapshot
from data.pipeline import RecalculationOutcome
from data.provider import LiveDataProvider, ProviderError
from data.state_manager import StateManager
from models.market import PriceTick
from safety.kill_switch import KillSwitch, KillSwitchActivatedError
from safety.lifecycle import LifecycleTransitionError
from safety.orchestrator import Orchestrator, OrchestratorNotRunningError
from storage.database import Database
from storage.db_writer import DBWriter, DBWriterFailedError, DBWriterNotAcceptingError, DBWriterState

#: Conservative, non-authoritative default Gamma-Poisson prior, used
#: ONLY when a caller does not supply one at fixture registration time.
#: This is a Stage 5 CALIBRATION PARAMETER, not a statistically-derived
#: or repository-authoritative value: analytics/bayesian.py declares no
#: default anywhere (GammaPoissonPrior always requires an explicit
#: shape/rate), and no other Stage 1-4 code declares an authoritative
#: starting belief for a fixture that has not yet been observed.
#: shape=1.0, rate=1.0 implies a prior mean of 1.0 goal/match -- the
#: simplest neutral choice for a strictly-positive Gamma prior. It is
#: NOT claimed to be statistically optimal, and callers should supply
#: their own values via register_fixture(home_prior=..., away_prior=...)
#: whenever a better-informed starting belief is available.
DEFAULT_PRIOR_SHAPE = 1.0
DEFAULT_PRIOR_RATE = 1.0


class FixtureNotRegisteredError(RuntimeError):
    """Raised when a fixture-scoped call is made before register_fixture()."""


class FixtureIdentityMismatchError(RuntimeError):
    """Raised when register_fixture() is called again for an already-registered
    fixture_id with a different league_id, home_team, or away_team than its
    original registration. These are treated as immutable identity fields;
    only phase/minute/home_goals/away_goals may be refreshed on repeat calls.
    """


class FixtureAlreadyHasActiveProviderError(RuntimeError):
    """Raised when start_provider_task() is called for a fixture that already has one running."""


class ProviderAlreadyAttachedError(RuntimeError):
    """Raised when the same LiveDataProvider instance is attached to more than one fixture."""


class ApplicationShuttingDownError(RuntimeError):
    """Raised when new work is requested after shutdown has been requested."""


@dataclass
class FixtureContext:
    """Per-fixture runtime state: everything that must NOT leak between fixtures.

    Owns its own StateManager + safety.Orchestrator (hence its own
    SystemLifecycle/Quarantine/local-unused-KillSwitch -- see module
    docstring), its own DataAgent/AuditorAgent, and its own Bayesian
    prior pair. Nothing here is shared with any other FixtureContext.
    """

    fixture_id: str
    league_id: str
    home_team: str
    away_team: str
    regulation_minutes: float
    state_manager: StateManager
    orchestrator: Orchestrator
    data_agent: DataAgent
    auditor_agent: AuditorAgent
    home_prior: GammaPoissonPrior
    away_prior: GammaPoissonPrior
    registered_at: float
    task: "Optional[asyncio.Task[None]]" = None
    last_provider_error: Optional[str] = None
    last_persisted_quarantine_timestamp: Optional[float] = None


@dataclass(frozen=True)
class FixtureStatus:
    """Immutable, read-only presentation of one fixture's current state."""

    fixture_id: str
    league_id: str
    home_team: str
    away_team: str
    lifecycle_state: str
    quarantined: bool
    provider_running: bool
    last_provider_error: Optional[str]


@dataclass(frozen=True)
class RuntimeStatus:
    """Immutable, read-only snapshot of the whole application runtime.

    Built fresh on every call from already-existing in-memory state --
    not a second observability system, just an aggregation.
    """

    started: bool
    shutting_down: bool
    terminal_error: Optional[str]
    global_kill_switch_state: str
    dbwriter_state: str
    dbwriter_pending: int
    dbwriter_last_error: Optional[str]
    registered_fixture_count: int
    fixtures: Tuple[FixtureStatus, ...]


class ApplicationRuntime:
    """Stage 5 application orchestrator: composition only, no domain math.

    Construction does not perform any I/O; call `start()` to connect
    the database, start the shared DBWriter, and construct the shared
    Journal/Quant/Market/Risk agents. Fixtures are registered
    afterwards via `register_fixture()`.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._database = Database(db_path=settings.db_path, busy_timeout_ms=settings.db_busy_timeout_ms)
        self._writer: Optional[DBWriter] = None
        self._journal: Optional[JournalAgent] = None
        self._quant_agent: Optional[QuantAgent] = None
        self._market_agent: Optional[MarketAgent] = None
        self._risk_agent: Optional[RiskAgent] = None

        #: The one, single, truly-global emergency stop for this runtime.
        #: See module docstring for why this is separate from any
        #: per-fixture Orchestrator's own (deliberately unused) KillSwitch.
        self._global_kill_switch = KillSwitch()

        self._fixtures: Dict[str, FixtureContext] = {}
        self._registration_lock = asyncio.Lock()
        self._attached_provider_ids: Set[int] = set()

        self._started = False
        self._stopped = False
        self._shutting_down = False
        self._shutdown_event = asyncio.Event()
        self._terminal_error: Optional[str] = None
        self.shutdown_task_exceptions: List[BaseException] = []

    # ------------------------------------------------------------------
    # Properties (read-only observability surface)
    # ------------------------------------------------------------------

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    @property
    def terminal_error(self) -> Optional[str]:
        return self._terminal_error

    @property
    def global_kill_switch(self) -> KillSwitch:
        """Expose the global KillSwitch for an authorized operator caller (e.g. main.py).

        The dashboard must never call `.activate()` on this -- it is
        read-only by requirement; this property exists for legitimate
        non-dashboard callers only.
        """
        return self._global_kill_switch

    def fixture_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._fixtures.keys()))

    def get_fixture(self, fixture_id: str) -> Optional[FixtureContext]:
        return self._fixtures.get(fixture_id)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect the database, start the shared DBWriter, construct shared agents.

        Idempotent: calling start() twice is a no-op.
        """
        if self._started:
            return
        await self._database.connect()
        await self._database.init_schema()

        self._writer = DBWriter(
            self._database,
            queue_maxsize=self._settings.db_queue_maxsize,
            batch_size=self._settings.db_batch_size,
        )
        await self._writer.start()

        self._journal = JournalAgent(self._writer)
        self._quant_agent = QuantAgent(self._journal)
        self._market_agent = MarketAgent(self._journal)
        self._risk_agent = RiskAgent(self._journal)

        self._started = True

    # ------------------------------------------------------------------
    # Fixture registration
    # ------------------------------------------------------------------

    async def register_fixture(
        self,
        fixture_id: str,
        league_id: str,
        home_team: str,
        away_team: str,
        home_prior: Optional[GammaPoissonPrior] = None,
        away_prior: Optional[GammaPoissonPrior] = None,
        phase: str = "PHASE_1_SLEEP",
        minute: int = 0,
        home_goals: int = 0,
        away_goals: int = 0,
    ) -> FixtureContext:
        """Register a fixture, constructing its per-fixture runtime state.

        Registration metadata (league_id/home_team/away_team) is
        entirely caller-supplied -- this runtime does not discover
        fixtures or invent metadata (no such source exists in the
        repository; see the Stage 5 preflight). `league_id` must
        already be a registered league in config.leagues (raises
        KeyError otherwise, matching that module's own existing,
        deliberate "no silent defaulting" policy).

        Repeated registration for an already-known fixture_id with the
        SAME immutable identity (league_id, home_team, away_team) is
        safe and idempotent: it re-issues the `matches` upsert
        (preserving created_at, per JournalAgent.record_match()'s
        existing semantics) to refresh mutable fields (phase/minute/
        goals) without rebuilding that fixture's live StateManager/
        Orchestrator/priors -- doing so would discard in-progress live
        state. Repeated registration that changes league_id, home_team,
        or away_team is REJECTED with FixtureIdentityMismatchError
        before touching the database at all: silently updating the
        `matches` row while an already-running FixtureContext keeps
        using the OLD league-derived regulation_minutes/Orchestrator
        configuration would split the in-memory and persisted views of
        the same fixture_id. No such "change a fixture's identity"
        operation exists anywhere in this repository's evidence, so
        rejecting it is the smallest safe policy.

        ATOMICITY: for a genuinely new fixture_id, the parent `matches`
        WriteRequest is awaited to successful enqueue on the shared
        DBWriter BEFORE this fixture becomes visible (published into
        `self._fixtures`, and therefore reachable from
        `_require_fixture()`, `fixture_ids()`, `ingest_event()`,
        `submit_price_tick()`, or `start_provider_task()`). Because the
        shared DBWriter is a single FIFO queue whose `_flush()` groups
        and executes requests in the order they were enqueued (see
        storage/db_writer.py), an enqueue that has already returned is
        guaranteed to be flushed no later than -- and, since it was
        enqueued first, strictly before -- any later child WriteRequest
        for the same fixture_id. No second synchronization mechanism is
        introduced; this relies entirely on FIFO ordering that already
        exists. If the parent enqueue fails, any fixture-local resources
        already constructed (its Orchestrator, in particular) are
        stopped and NOTHING is published -- a failed registration can
        never leave a half-visible fixture.

        Concurrent calls for the SAME new fixture_id are serialized by
        an internal lock so exactly one FixtureContext is ever created.

        Raises:
            ApplicationShuttingDownError: if shutdown has been requested.
            KeyError: if league_id is not a registered league.
            FixtureIdentityMismatchError: if fixture_id is already
                registered with a different league_id/home_team/away_team.
            DBWriterNotAcceptingError, DBWriterFailedError: propagated
                (and treated as a terminal-failure trigger) if the
                shared DBWriter cannot accept the registration write.
        """
        if self._shutting_down:
            raise ApplicationShuttingDownError("application runtime is shutting down; no new registrations accepted")
        league = get_league(league_id)

        async with self._registration_lock:
            existing = self._fixtures.get(fixture_id)
            if existing is not None:
                if (existing.league_id, existing.home_team, existing.away_team) != (league_id, home_team, away_team):
                    raise FixtureIdentityMismatchError(
                        f"fixture_id {fixture_id!r} is already registered with "
                        f"league_id={existing.league_id!r}, home_team={existing.home_team!r}, "
                        f"away_team={existing.away_team!r}; cannot re-register with "
                        f"league_id={league_id!r}, home_team={home_team!r}, away_team={away_team!r}"
                    )
                async with self._dbwriter_failure_guard():
                    await self._journal.record_match(
                        fixture_id, league_id, home_team, away_team, phase, minute, home_goals, away_goals
                    )
                return existing

            manager = StateManager()
            orchestrator = Orchestrator(
                manager=manager,
                regulation_minutes=league.regulation_minutes,
                pressure_weights=self._settings.pressure_weights,
                monte_carlo_simulations=self._settings.monte_carlo_simulations,
                monte_carlo_seed=self._settings.monte_carlo_seed,
            )
            await orchestrator.start()

            try:
                data_agent = DataAgent(
                    orchestrator,
                    journal=self._journal,
                    data_age_green_seconds=self._settings.data_age_green_seconds,
                    data_age_yellow_seconds=self._settings.data_age_yellow_seconds,
                )
                auditor_agent = AuditorAgent(journal=self._journal, quarantine=orchestrator.quarantine)

                context = FixtureContext(
                    fixture_id=fixture_id,
                    league_id=league_id,
                    home_team=home_team,
                    away_team=away_team,
                    regulation_minutes=league.regulation_minutes,
                    state_manager=manager,
                    orchestrator=orchestrator,
                    data_agent=data_agent,
                    auditor_agent=auditor_agent,
                    home_prior=home_prior if home_prior is not None else GammaPoissonPrior(DEFAULT_PRIOR_SHAPE, DEFAULT_PRIOR_RATE),
                    away_prior=away_prior if away_prior is not None else GammaPoissonPrior(DEFAULT_PRIOR_SHAPE, DEFAULT_PRIOR_RATE),
                    registered_at=time.time(),
                )

                # Parent write MUST be successfully enqueued before this
                # fixture becomes visible to anything else -- see the
                # ATOMICITY note above. Not yet published to
                # self._fixtures at this point.
                async with self._dbwriter_failure_guard():
                    await self._journal.record_match(
                        fixture_id, league_id, home_team, away_team, phase, minute, home_goals, away_goals
                    )
            except BaseException:
                # Registration did not complete: never publish a
                # half-created fixture, and never leak the Orchestrator
                # we already started. The original exception (including
                # a DBWriter-failure escalation already triggered by
                # _dbwriter_failure_guard, or a CancelledError) is never
                # swallowed.
                await orchestrator.stop()
                raise

            self._fixtures[fixture_id] = context
            return context

    def _require_fixture(self, fixture_id: str) -> FixtureContext:
        context = self._fixtures.get(fixture_id)
        if context is None:
            raise FixtureNotRegisteredError(
                f"fixture_id {fixture_id!r} is not registered; call register_fixture() first"
            )
        return context

    # ------------------------------------------------------------------
    # Per-event live flow
    # ------------------------------------------------------------------

    async def ingest_event(
        self,
        fixture_id: str,
        raw_event: Mapping[str, Any],
        event_timestamp: float,
        event_id: Optional[str] = None,
        received_timestamp: Optional[float] = None,
    ) -> Optional[RecalculationOutcome]:
        """Ingest one raw event for a registered fixture through the full Stage 4 flow.

        Order: global KillSwitch check -> fixture lookup (rejects an
        unregistered fixture BEFORE any DB write is attempted) ->
        DataAgent.ingest_event() (the existing safety-gated live path,
        unmodified) -> downstream Quant/Risk/Auditor routing ->
        quarantine-transition bridge.

        Every exception DataAgent.ingest_event() can raise
        (KillSwitchActivatedError from that fixture's own -- never
        activated -- local switch, OrchestratorNotRunningError,
        asyncio.CancelledError, or any unclassified exception Stage 3
        chooses to re-raise) propagates unchanged: this method adds no
        new exception handling around it, only around the
        DBWriter-specific failure signal (see _dbwriter_failure_guard).
        """
        if self._shutting_down:
            raise ApplicationShuttingDownError("application runtime is shutting down; no new work accepted")
        self._global_kill_switch.check()
        context = self._require_fixture(fixture_id)

        async with self._dbwriter_failure_guard():
            outcome = await context.data_agent.ingest_event(
                fixture_id,
                raw_event,
                event_timestamp,
                context.home_prior,
                context.away_prior,
                event_id=event_id,
                received_timestamp=received_timestamp,
            )
            payload_checksum = build_snapshot(fixture_id, raw_event, event_timestamp, received_timestamp).payload_checksum
            await self._route_outcome(context, outcome, payload_checksum)
            await self._bridge_quarantine(context)

        return outcome

    async def _route_outcome(
        self, context: FixtureContext, outcome: Optional[RecalculationOutcome], payload_checksum: str
    ) -> None:
        """Route a RecalculationOutcome to Quant, then (for a full result) Risk and Auditor.

        No agent here calls another agent directly (Stage 4 invariant);
        this method is the ONLY place that sequences them.
        """
        if outcome is None:
            return

        await self._quant_agent.handle_outcome(outcome)

        if outcome.recalculated and outcome.full_result is not None:
            risk_metrics = self._risk_agent.evaluate(outcome.full_result)
            await self._risk_agent.persist(risk_metrics)

            await context.auditor_agent.audit(
                outcome.full_result,
                outcome.snapshot,
                context.home_prior,
                context.away_prior,
                context.regulation_minutes,
                self._settings.pressure_weights,
                payload_checksum=payload_checksum,
                monte_carlo_simulations=self._settings.monte_carlo_simulations,
                monte_carlo_seed=self._settings.monte_carlo_seed,
            )

    async def _bridge_quarantine(self, context: FixtureContext) -> None:
        """Persist a NEWLY-made quarantine decision through the Journal, exactly once.

        Bridges the gap identified in the Stage 5 preflight: neither
        safety.Orchestrator nor AuditorAgent ever calls
        JournalAgent.record_quarantine_record() themselves (by design --
        Quarantine is the in-memory source of truth for runtime
        decisions; nothing there is modified). This method only
        OBSERVES that store and forwards a fact, comparing the record's
        own timestamp against the last one already persisted for this
        fixture so a repeated, unchanged quarantine record (e.g. from
        skipped subsequent events for an already-quarantined fixture)
        is never re-persisted as a duplicate transition.
        """
        record = context.orchestrator.quarantine.get(context.fixture_id)
        if record is None:
            return
        if context.last_persisted_quarantine_timestamp == record.timestamp:
            return
        await self._journal.record_quarantine_record(record)
        context.last_persisted_quarantine_timestamp = record.timestamp

    async def submit_price_tick(self, tick: PriceTick) -> None:
        """Submit an already-valid, externally-supplied PriceTick for a registered fixture.

        MarketAgent remains input-driven (no market provider exists or
        is invented here, per Stage 4/5 scope): this is simply the
        application-level boundary a caller with real price data uses.
        """
        if self._shutting_down:
            raise ApplicationShuttingDownError("application runtime is shutting down; no new work accepted")
        self._global_kill_switch.check()
        self._require_fixture(tick.fixture_id)

        async with self._dbwriter_failure_guard():
            await self._market_agent.record_price_tick(tick)

    # ------------------------------------------------------------------
    # Provider task ownership
    # ------------------------------------------------------------------

    def start_provider_task(self, fixture_id: str, provider: LiveDataProvider) -> "asyncio.Task[None]":
        """Start a background task streaming `provider` into `ingest_event()` for one fixture.

        One provider instance per fixture, always: LiveDataProvider
        stores its connection_state as a single mutable instance
        attribute (see data/provider.py), so sharing one instance
        across concurrent fixtures would race that field. This method
        refuses to attach the same provider object twice.

        This runtime deliberately does NOT call
        DataAgent.run_provider()/Orchestrator.run_provider() for this
        purpose: those delegate straight to Orchestrator.process()
        without giving this runtime a chance to run its own per-event
        routing (Quant/Risk/Auditor/quarantine-bridge) or checksum
        correlation. Iterating `provider.stream_events()` here and
        calling THIS runtime's own `ingest_event()` per raw event is
        the documented, intended alternative (see agents/data_agent.py's
        own run_provider() docstring) -- it reuses the provider's
        existing async-iterator contract without duplicating any
        normalization/state/analytics logic itself.
        """
        if self._shutting_down:
            raise ApplicationShuttingDownError("application runtime is shutting down; no new provider tasks accepted")
        context = self._require_fixture(fixture_id)
        if context.task is not None and not context.task.done():
            raise FixtureAlreadyHasActiveProviderError(f"fixture_id {fixture_id!r} already has an active provider task")
        if id(provider) in self._attached_provider_ids:
            raise ProviderAlreadyAttachedError(
                "the same LiveDataProvider instance cannot be attached to more than one fixture "
                "(its connection_state is a single shared mutable attribute)"
            )

        self._attached_provider_ids.add(id(provider))
        task = asyncio.create_task(self._run_provider_loop(fixture_id, provider))
        context.task = task
        return task

    async def _run_provider_loop(self, fixture_id: str, provider: LiveDataProvider) -> None:
        context = self._fixtures[fixture_id]
        try:
            async for raw_event in provider.stream_events(fixture_id):
                if self._shutting_down:
                    break
                try:
                    await self.ingest_event(fixture_id, raw_event, time.time())
                except KillSwitchActivatedError:
                    break
                except OrchestratorNotRunningError:
                    break
        except asyncio.CancelledError:
            raise
        except ProviderError as exc:
            # Fixture-local, by construction: only THIS fixture's own
            # (never-shared) Orchestrator/lifecycle is touched. Every
            # other fixture's Orchestrator is a completely separate
            # object and is entirely unaffected.
            context.last_provider_error = str(exc)
            try:
                await context.orchestrator.lifecycle.mark_failed("provider_failure", str(exc))
            except LifecycleTransitionError:
                pass  # benign race: this fixture's lifecycle already left RUNNING (e.g. shutdown)
        finally:
            # Release this id() once the task is done, so a later,
            # genuinely different provider object cannot be mistaken for
            # a reuse attempt merely because CPython recycled the same
            # memory address after the earlier provider was garbage
            # collected.
            self._attached_provider_ids.discard(id(provider))

    # ------------------------------------------------------------------
    # DBWriter failure detection / escalation
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _dbwriter_failure_guard(self) -> AsyncIterator[None]:
        """Detect a DBWriter failure surfacing from a JournalAgent call and escalate.

        Never swallows the exception -- "do not claim writes succeeded"
        -- it is always re-raised after requesting a global shutdown.
        """
        try:
            yield
        except (DBWriterNotAcceptingError, DBWriterFailedError) as exc:
            self.request_shutdown(f"DBWriter failure: {exc}", is_failure=True)
            raise

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def request_shutdown(self, reason: str = "", *, is_failure: bool = False) -> None:
        """Signal that shutdown should begin. Safe to call multiple times/reasons.

        Does not itself perform teardown -- see `shutdown()`. Only the
        first failure reason is retained, matching KillSwitch's own
        "first activation wins" convention.
        """
        self._shutting_down = True
        if is_failure and self._terminal_error is None:
            self._terminal_error = reason or "unspecified failure"
        self._shutdown_event.set()

    async def wait_for_shutdown(self, poll_interval: float = 1.0) -> None:
        """Block until shutdown is requested, ALSO proactively detecting a terminal DBWriter FAILED.

        DBWriter can transition to FAILED asynchronously, inside its
        own background consumer loop, with no callback mechanism and no
        guarantee any caller will make another enqueue()-ing call soon
        enough to reactively discover it (_dbwriter_failure_guard alone
        only catches it on the NEXT such call). This bounded poll loop
        (default 1s) is the smallest addition that also catches the
        "system went idle right when persistence died" case, without
        adding unrequested per-event concurrency.
        """
        while not self._shutdown_event.is_set():
            if self._writer is not None and self._writer.state == DBWriterState.FAILED:
                self.request_shutdown(f"DBWriter reached FAILED: {self._writer.last_error}", is_failure=True)
                break
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                continue

    async def shutdown(self) -> None:
        """Perform the full graceful teardown sequence. Idempotent.

        Order: stop accepting new work (already true once
        request_shutdown() was called by ANY path) -> cancel and await
        every fixture provider task -> stop each fixture's own
        Orchestrator lifecycle -> stop the shared DBWriter (drains
        whatever is already queued) -> close the shared Database.
        Cancelling tasks strictly BEFORE stopping the writer prevents a
        `DBWriterNotAcceptingError` from ever escaping a task that
        hadn't finished unwinding yet; stopping the writer strictly
        before closing the database prevents a drained-but-uncommitted
        write from losing its connection.
        """
        if self._stopped:
            return
        self._shutting_down = True
        self._shutdown_event.set()

        tasks = [context.task for context in self._fixtures.values() if context.task is not None and not context.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    self.shutdown_task_exceptions.append(result)

        for context in self._fixtures.values():
            await context.orchestrator.stop()

        if self._writer is not None:
            await self._writer.stop()
        await self._database.close()

        self._stopped = True

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def build_status_snapshot(self) -> RuntimeStatus:
        """Build an immutable RuntimeStatus from current in-memory state.

        Pure aggregation of existing state -- not a second metrics
        system. Safe to call synchronously from the same thread/loop
        this runtime lives on.
        """
        fixtures = []
        for context in self._fixtures.values():
            record = context.orchestrator.quarantine.get(context.fixture_id)
            fixtures.append(
                FixtureStatus(
                    fixture_id=context.fixture_id,
                    league_id=context.league_id,
                    home_team=context.home_team,
                    away_team=context.away_team,
                    lifecycle_state=context.orchestrator.lifecycle.state.value,
                    quarantined=record is not None,
                    provider_running=context.task is not None and not context.task.done(),
                    last_provider_error=context.last_provider_error,
                )
            )

        writer_state = self._writer.state.value if self._writer is not None else DBWriterState.CREATED.value
        return RuntimeStatus(
            started=self._started,
            shutting_down=self._shutting_down,
            terminal_error=self._terminal_error,
            global_kill_switch_state=self._global_kill_switch.state.value,
            dbwriter_state=writer_state,
            dbwriter_pending=self._writer.pending if self._writer is not None else 0,
            dbwriter_last_error=self._writer.last_error if self._writer is not None else None,
            registered_fixture_count=len(self._fixtures),
            fixtures=tuple(fixtures),
        )

    async def async_status_snapshot(self) -> RuntimeStatus:
        """Coroutine wrapper around build_status_snapshot(), for cross-thread bridging.

        Used by the dashboard via asyncio.run_coroutine_threadsafe() so
        the snapshot is always built ON this runtime's own event loop,
        never raced against from another thread.
        """
        return self.build_status_snapshot()
