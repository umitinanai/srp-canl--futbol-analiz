"""Stage 4 Quant Agent: consumes an already-computed RecalculationOutcome.

This agent performs NO analytics computation itself. Every number in a
QuantResult was already produced by data.pipeline.run_full_recalculation()
(invoked, under safety control, by safety.orchestrator.Orchestrator.process()
-- which is in turn called by agents.data_agent.DataAgent). The Quant
Agent's entire job is to decide what happens to that already-computed
result: persist a full result, or record the fact of a lightweight-only
update without inventing a persistence shape the schema doesn't support.

Lightweight-result decision (evidence-backed): storage/schema.sql's
analytics_results table has no columns for a lightweight-only result
(it only has slots for a full QuantResult's fields: lambda_home,
monte_carlo_*, etc.). Rather than force lightweight_result's
{"home_pressure_index", "away_pressure_index"} dict into that
full-result-shaped table (which would either leave most columns NULL
for a reason unrelated to genuine missing data, or require a schema
change -- both rejected by the "no new contract/schema change unless
proven necessary" default), a lightweight recalculation is instead
recorded as a JournalEventType.MODEL_CALCULATION fact via the Journal
Agent's generic journal_events path, carrying the lightweight values in
its `details` dict. This is schema-safe and uses an existing event
type/table rather than inventing either.
"""

from __future__ import annotations

from typing import Any, Optional

from data.pipeline import RecalculationOutcome
from models.events import JournalEvent, JournalEventType


class QuantAgent:
    """Persists full QuantResults and journals lightweight-recalculation facts.

    Holds no state; every call is a pure function of the
    RecalculationOutcome handed to it, aside from the optional
    constructor-injected Journal Agent used for persistence.
    """

    def __init__(self, journal: Optional[Any] = None) -> None:
        """Initialize the Quant Agent.

        Args:
            journal: optional agents.journal_agent.JournalAgent (or any
                object exposing the same async record_* methods),
                duck-typed to avoid a hard compile-time dependency.
        """
        self._journal = journal

    async def handle_outcome(self, outcome: RecalculationOutcome) -> None:
        """Route one RecalculationOutcome to the correct persistence/journal path.

        Never calls data.pipeline.run_full_recalculation() or any
        analytics.* function -- it only reads fields already present on
        `outcome`. Never mutates outcome.full_result (QuantResult is a
        frozen dataclass; this method never constructs a new instance
        from it either).

        Args:
            outcome: a RecalculationOutcome as returned by
                safety.orchestrator.Orchestrator.process() (via
                agents.data_agent.DataAgent.ingest_event()).
        """
        if not outcome.event_applied:
            # Duplicate or stale event: StateManager.apply_event() made
            # no change and no analytics ran. Nothing new to record.
            return

        if self._journal is None:
            return

        if outcome.recalculated and outcome.full_result is not None:
            await self._journal.record_quant_result(outcome.full_result)
            await self._journal.record_journal_event(
                JournalEvent(
                    event_type=JournalEventType.MODEL_CALCULATION,
                    fixture_id=outcome.full_result.fixture_id,
                    timestamp=outcome.full_result.timestamp,
                    details={
                        "recalculated": "true",
                        "monte_carlo_simulations": str(outcome.full_result.monte_carlo_simulations),
                    },
                )
            )
        elif outcome.lightweight_result is not None:
            details = {"recalculated": "false"}
            details.update({key: str(value) for key, value in outcome.lightweight_result.items()})
            await self._journal.record_journal_event(
                JournalEvent(
                    event_type=JournalEventType.MODEL_CALCULATION,
                    fixture_id=outcome.snapshot.fixture_id,
                    timestamp=outcome.snapshot.timestamp,
                    details=details,
                )
            )
