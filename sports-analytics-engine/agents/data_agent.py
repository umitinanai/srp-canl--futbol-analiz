"""Stage 4 Data Agent: adapter/coordinator above the existing provider +
safety-gated live processing path.

This module owns NO analytics mathematics, NO live-flow orchestration,
and NO safety policy. It reuses, unmodified:

    - data.normalizer.build_snapshot() / data_age_status() to obtain a
      checksummed models.match.Snapshot and its freshness classification
      for the ONE purpose those functions exist for but that
      data.pipeline.process_event() itself never calls them for:
      producing a persistable record of raw ingestion. (Confirmed during
      the Stage 4 preflight: data.pipeline.process_event() only calls
      data.normalizer.normalize_raw_event(), never build_snapshot() --
      so no models.match.Snapshot is ever produced by the live flow
      today. Calling the existing build_snapshot() here is reuse, not a
      new implementation.)
    - safety.orchestrator.Orchestrator.process() for the ENTIRE
      kill-switch / lifecycle / quarantine-gated live flow. This agent
      never calls data.pipeline.process_event() directly and never
      touches data.state_manager.StateManager directly.

Bayesian prior ownership: data.pipeline.process_event() requires
caller-supplied home_prior/away_prior on every call. This agent does
NOT own, store, or evolve them -- they are passed through as explicit
parameters on every call, exactly mirroring Orchestrator.process()'s
own signature. Assigning hidden prior ownership to this agent was
explicitly not authorized (Stage 4 preflight correction E); the
smallest-architecture choice is a pure pass-through.

Fixture discovery / phase-based polling scheduling are explicitly out
of scope for this agent (Stage 4 preflight correction D): MatchPhase
and Settings.phase*_interval_seconds exist but are not consumed by any
runtime code, and wiring a scheduler would require application-level
orchestration that belongs to Stage 5, not Stage 4's building blocks.

KNOWN SCHEMA BOUNDARY: storage/schema.sql declares `snapshots.fixture_id`
as a FOREIGN KEY REFERENCES matches(fixture_id), enforced live
(Database.connect() turns on PRAGMA foreign_keys=ON). ingest_event()'s
Snapshot persistence therefore requires a `matches` row for the
fixture to already exist. This agent intentionally does NOT create one
-- it has no team/league metadata available anywhere on the live
provider -> normalizer -> pipeline path to create an honest one, and
fabricating placeholder business data (fake team/league names) merely
to satisfy the constraint was rejected as worse than the alternative.
Whichever component owns real fixture/match registration (Stage 5, or
a test's own setup) is responsible for calling
agents.journal_agent.JournalAgent.record_match() first. Against a real
schema-enforced Database with no such row present, the enqueued
snapshot write will fail with a SQLite IntegrityError and the shared
DBWriter will retry-then-FAIL it -- this propagates like any other
unhandled failure (see storage/db_writer.py); it is not swallowed here.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from analytics.bayesian import GammaPoissonPrior
from data.normalizer import build_snapshot, data_age_status
from data.pipeline import RecalculationOutcome
from data.provider import LiveDataProvider
from models.events import JournalEvent, JournalEventType
from safety.orchestrator import Orchestrator


class DataAgent:
    """Coordinates one raw event's journey through ingestion + the safety-gated pipeline.

    Holds no state of its own beyond its constructor-injected
    collaborators (an Orchestrator, and optionally a JournalAgent) --
    all per-fixture/per-event data (priors, timestamps) is passed
    explicitly on each call, not cached here.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        journal: Optional[Any] = None,
        data_age_green_seconds: float = 5.0,
        data_age_yellow_seconds: float = 10.0,
    ) -> None:
        """Initialize the Data Agent.

        Args:
            orchestrator: the Stage 3 safety-gated Orchestrator this
                agent dispatches every event through. Never bypassed.
            journal: optional agents.journal_agent.JournalAgent (or any
                object exposing the same async record_* methods) used
                to persist raw snapshot receipt facts. Duck-typed (not
                imported by name) to avoid a hard compile-time
                dependency; callers may omit it entirely for a
                Data Agent that only forwards to the safety layer.
            data_age_green_seconds: forwarded to data.normalizer.data_age_status().
            data_age_yellow_seconds: forwarded to data.normalizer.data_age_status().
        """
        self._orchestrator = orchestrator
        self._journal = journal
        self._data_age_green_seconds = data_age_green_seconds
        self._data_age_yellow_seconds = data_age_yellow_seconds

    async def ingest_event(
        self,
        fixture_id: str,
        raw_event: Mapping[str, Any],
        event_timestamp: float,
        home_prior: GammaPoissonPrior,
        away_prior: GammaPoissonPrior,
        event_id: Optional[str] = None,
        received_timestamp: Optional[float] = None,
    ) -> Optional[RecalculationOutcome]:
        """Ingest one raw provider event: record its receipt, then run it through Orchestrator.process().

        Ordering is deliberate: the raw Snapshot is built and persisted
        BEFORE the safety-gated call, so a durable record of "this data
        was received" exists independently of whatever the control
        plane subsequently decides to do with it (kill-switch
        rejection, quarantine skip, or normal processing). This keeps
        "recording" and "decision" separated, per Stage 4's
        observability requirement.

        Any exception raised by Orchestrator.process() (KillSwitchActivatedError,
        OrchestratorNotRunningError, asyncio.CancelledError, or any
        unclassified exception it chooses to re-raise) propagates
        unchanged -- this method adds no new exception handling on top
        of the existing Stage 3 semantics.

        Args:
            fixture_id: the fixture identifier.
            raw_event: a raw, provider-shaped event dict.
            event_timestamp: unix epoch seconds when the event occurred.
            home_prior: home side's current Gamma-Poisson prior,
                forwarded to Orchestrator.process() unchanged. Owned by
                the caller, not this agent.
            away_prior: away side's current Gamma-Poisson prior, owned
                by the caller.
            event_id: optional provider-supplied unique event identifier.
            received_timestamp: unix epoch seconds this event was
                received locally. Defaults to the current time; the same
                value is used both for the persisted Snapshot and for
                the Orchestrator.process() call, so the two stay
                consistent.

        Returns:
            Whatever Orchestrator.process() returns: a RecalculationOutcome,
            or None if the fixture is currently quarantined.

        Note:
            If `journal` is a real JournalAgent backed by a schema-
            enforced Database, persisting the Snapshot requires a
            `matches` row for `fixture_id` to already exist (see the
            module docstring's KNOWN SCHEMA BOUNDARY note) -- this
            method does not create one.
        """
        received_timestamp = time.time() if received_timestamp is None else received_timestamp

        snapshot = build_snapshot(fixture_id, raw_event, event_timestamp, received_timestamp)
        age_status = data_age_status(
            snapshot, self._data_age_green_seconds, self._data_age_yellow_seconds
        )
        if self._journal is not None:
            await self._journal.record_snapshot(snapshot, age_status)

        outcome = await self._orchestrator.process(
            fixture_id,
            raw_event,
            event_timestamp,
            home_prior,
            away_prior,
            event_id=event_id,
            received_timestamp=received_timestamp,
        )

        if self._journal is not None:
            event_type = (
                JournalEventType.SNAPSHOT_RECEIVED if outcome is not None
                else JournalEventType.SNAPSHOT_REJECTED
            )
            await self._journal.record_journal_event(
                JournalEvent(
                    event_type=event_type,
                    fixture_id=fixture_id,
                    timestamp=received_timestamp,
                    details={"data_age_status": age_status.value},
                )
            )

        return outcome

    async def run_provider(
        self,
        provider: LiveDataProvider,
        fixture_id: str,
        home_prior: GammaPoissonPrior,
        away_prior: GammaPoissonPrior,
    ) -> None:
        """Thin, unmodified delegation to Orchestrator.run_provider().

        This is pure reuse (zero new logic): it exists so DataAgent can
        serve as a complete adapter over a LiveDataProvider's stream
        without callers needing to reach past it into the injected
        Orchestrator directly. Unlike ingest_event(), this path does NOT
        persist a Snapshot per event -- Orchestrator.run_provider()'s own
        loop calls its own process() internally and does not hand control
        back to this agent per event, so duplicating its loop here (to
        splice in snapshot persistence) was rejected as unnecessary
        risk to already-tested provider/kill-switch/cancellation
        handling. Callers that need per-event snapshot persistence
        should iterate `provider.stream_events(fixture_id)` themselves
        and call ingest_event() for each raw event instead.
        """
        await self._orchestrator.run_provider(provider, fixture_id, home_prior, away_prior)
