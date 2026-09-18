"""Stage 4 persistence boundary: the sole translator from domain
contracts/events into storage.db_writer.WriteRequest objects.

Repository evidence (models/metrics.py module docstring) describes the
intended chain "Quant Agent -> Auditor Agent -> Journal Agent -> DB
writer". This module is that final hop: every other Stage 4 agent
hands JournalAgent an already-validated, already-immutable domain
object, and JournalAgent alone knows how to turn it into a
storage.db_writer.WriteRequest matching storage/schema.sql exactly.

JournalAgent never constructs its own Database/DBWriter/WriterToken --
it is handed one already-running, shared DBWriter instance (dependency
injection, no global state) and only ever calls its public
enqueue()/enqueue_nowait() API, exactly like storage/db_writer.py's own
module docstring requires of every writer in the system. It also never
makes a safety/product decision (kill-switch activation, quarantine
placement, audit verdicts): it only records decisions/facts that some
other component already made.

Serialization notes:
    - JSON columns use models.match.canonical_json() (sorted keys, "?"
      no ad hoc json.dumps) so persisted JSON is deterministic and
      matches the checksum-style canonicalization already used
      elsewhere in the repository.
    - Enum values are persisted via their `.value` (or, for `str, Enum`
      members embedded inside a details dict, serialize natively via
      json.dumps since they are themselves str instances) -- never via
      str(enum_member), which would produce "ClassName.MEMBER" instead
      of the bare value.
    - snapshots.identity_checksum carries a real UNIQUE constraint
      (storage/schema.sql). A duplicate snapshot is an EXPECTED,
      semantically valid occurrence (see models/match.py's Snapshot
      docstring), not a database error -- so record_snapshot() uses
      "INSERT OR IGNORE" rather than a plain INSERT. A plain INSERT
      would raise an IntegrityError on every legitimate duplicate,
      which DBWriter's generic retry-then-FAIL policy would treat as a
      transient failure and eventually escalate to DBWriterState.FAILED,
      incorrectly taking down the single shared writer for an entirely
      expected condition.
"""

from __future__ import annotations

import time
from typing import Optional

from models.events import AnomalyEvent, JournalEvent, JournalEventType, KillSwitchEvent, QuarantineEvent
from models.market import PriceTick
from models.match import Snapshot, canonical_json
from models.match import DataAgeStatus
from models.metrics import AuditResult, QuantResult, RiskMetrics
from safety.quarantine import QuarantineRecord
from storage.db_writer import DBWriter, WriteRequest

_MATCH_UPSERT_QUERY = """
    INSERT INTO matches
        (fixture_id, league_id, home_team, away_team, phase, minute,
         home_goals, away_goals, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(fixture_id) DO UPDATE SET
        league_id=excluded.league_id,
        home_team=excluded.home_team,
        away_team=excluded.away_team,
        phase=excluded.phase,
        minute=excluded.minute,
        home_goals=excluded.home_goals,
        away_goals=excluded.away_goals,
        updated_at=excluded.updated_at
"""

_SNAPSHOT_INSERT_QUERY = """
    INSERT OR IGNORE INTO snapshots
        (fixture_id, event_timestamp, received_timestamp, data_age,
         data_age_status, payload_checksum, identity_checksum, payload_json,
         created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_ANALYTICS_RESULT_INSERT_QUERY = """
    INSERT INTO analytics_results
        (fixture_id, timestamp, lambda_home, lambda_away,
         home_win_probability, draw_probability, away_win_probability,
         monte_carlo_home_win, monte_carlo_draw, monte_carlo_away_win,
         monte_carlo_simulations, mi_rate, z_mi, pressure_index,
         pressure_acceleration, xg_proxy, shot_quality_proxy,
         market_reaction_elasticity, signal_quality_score, data_quality,
         calibration_confidence, metadata_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_MARKET_TICK_INSERT_QUERY = """
    INSERT INTO market_ticks
        (fixture_id, market_type, outcome, price, timestamp, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
"""

_AUDIT_EVENT_INSERT_QUERY = """
    INSERT INTO audit_events
        (fixture_id, timestamp, verdict, checks_json, reasons_json,
         payload_checksum, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_JOURNAL_EVENT_INSERT_QUERY = """
    INSERT INTO journal_events
        (event_type, fixture_id, timestamp, details_json, created_at)
    VALUES (?, ?, ?, ?, ?)
"""

_QUARANTINE_EVENT_INSERT_QUERY = """
    INSERT INTO quarantine_events
        (fixture_id, reason, timestamp, details_json, created_at)
    VALUES (?, ?, ?, ?, ?)
"""

_SHADOW_ANALYSIS_INSERT_QUERY = """
    INSERT INTO shadow_analysis
        (fixture_id, timestamp, model_confidence, uncertainty, volatility,
         simulated_exposure, simulated_drawdown, risk_score,
         calibration_error, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class JournalAgent:
    """Stage 4's sole owner of translation from domain contracts to WriteRequests.

    Every method here does exactly one thing: build a WriteRequest that
    matches storage/schema.sql exactly, and enqueue it on the single
    shared DBWriter this instance was constructed with. No method here
    decides anything -- callers (Data/Quant/Market/Risk/Auditor agents,
    or safety.* components) have already decided what happened; this
    class only records it.

    DBWriter failures (DBWriterNotAcceptingError, DBWriterQueueFullError)
    are never caught here -- they propagate to the caller unchanged, per
    "Journal/DBWriter failure must be observable and propagated".
    """

    def __init__(self, writer: DBWriter) -> None:
        """Initialize the Journal Agent around an already-constructed DBWriter.

        Args:
            writer: the single shared DBWriter instance for the process.
                JournalAgent never constructs its own DBWriter and never
                requests a WriterToken -- it only calls enqueue() on the
                instance handed to it.
        """
        self._writer = writer

    @property
    def pending(self) -> int:
        """Return the number of write requests currently queued (observability)."""
        return self._writer.pending

    async def record_match(
        self,
        fixture_id: str,
        league_id: str,
        home_team: str,
        away_team: str,
        phase: str,
        minute: int,
        home_goals: int,
        away_goals: int,
        now: Optional[float] = None,
    ) -> None:
        """Upsert a match row, called only when a caller already has full match metadata.

        Stage 4's live-event agents (Data Agent) do not receive team
        names / league identifiers from the existing provider ->
        normalizer -> pipeline path (models.match.Team/MatchState, which
        do carry that data, are not wired into the live pipeline -- see
        data.state_manager.LiveMatchState, which has no such fields).
        This method exists so a caller that DOES have this data (e.g. a
        fixture-registration step, out of Stage 4's scope) can still
        populate `matches` through the single Journal/DBWriter boundary;
        it is not invoked automatically by any other Stage 4 agent.
        """
        now = time.time() if now is None else now
        await self._writer.enqueue(
            WriteRequest(
                table="matches",
                query=_MATCH_UPSERT_QUERY,
                params=(
                    fixture_id, league_id, home_team, away_team, phase,
                    minute, home_goals, away_goals, now, now,
                ),
            )
        )

    async def record_snapshot(self, snapshot: Snapshot, data_age_status: DataAgeStatus, now: Optional[float] = None) -> None:
        """Persist a checksummed Snapshot (see module docstring re: OR IGNORE).

        storage/schema.sql declares `snapshots.fixture_id` as a
        `FOREIGN KEY REFERENCES matches (fixture_id)`, enforced live
        (Database.connect() turns on `PRAGMA foreign_keys=ON`). This
        method does not create or fabricate that parent row -- a
        `matches` row for `snapshot.fixture_id` must already exist
        (via record_match(), called by whichever component owns real
        match metadata) or the enqueued write will fail with a SQLite
        IntegrityError and the DBWriter will retry-then-FAIL it.
        """
        now = time.time() if now is None else now
        await self._writer.enqueue(
            WriteRequest(
                table="snapshots",
                query=_SNAPSHOT_INSERT_QUERY,
                params=(
                    snapshot.fixture_id,
                    snapshot.event_timestamp,
                    snapshot.received_timestamp,
                    snapshot.data_age,
                    data_age_status.value,
                    snapshot.payload_checksum,
                    snapshot.identity_checksum,
                    canonical_json(snapshot.payload),
                    now,
                ),
            )
        )

    async def record_quant_result(self, result: QuantResult, now: Optional[float] = None) -> None:
        """Persist a full QuantResult to analytics_results.

        `fair_probabilities` has no dedicated analytics_results column
        (schema evidence); it is folded into metadata_json alongside the
        existing `metadata` dict rather than being silently dropped or
        provoking a schema change (see Stage 4 preflight, Missing-
        Contract analysis item on MarketProbability/fair_probabilities).

        `analytics_results.fixture_id` carries the same FOREIGN KEY to
        `matches` as `snapshots` does (see record_snapshot()) -- a
        `matches` row for `result.fixture_id` must already exist.
        """
        now = time.time() if now is None else now
        metadata_json = canonical_json(
            {"metadata": result.metadata, "fair_probabilities": result.fair_probabilities}
        )
        await self._writer.enqueue(
            WriteRequest(
                table="analytics_results",
                query=_ANALYTICS_RESULT_INSERT_QUERY,
                params=(
                    result.fixture_id,
                    result.timestamp,
                    result.lambda_home,
                    result.lambda_away,
                    result.home_win_probability,
                    result.draw_probability,
                    result.away_win_probability,
                    result.monte_carlo_home_win,
                    result.monte_carlo_draw,
                    result.monte_carlo_away_win,
                    result.monte_carlo_simulations,
                    result.mi_rate,
                    result.z_mi,
                    result.pressure_index,
                    result.pressure_acceleration,
                    result.xg_proxy,
                    result.shot_quality_proxy,
                    result.market_reaction_elasticity,
                    result.signal_quality_score,
                    result.data_quality,
                    result.calibration_confidence,
                    metadata_json,
                    now,
                ),
            )
        )

    async def record_price_tick(self, tick: PriceTick, now: Optional[float] = None) -> None:
        """Persist a PriceTick to market_ticks.

        `market_ticks.fixture_id` carries the same FOREIGN KEY to
        `matches` as `snapshots` does (see record_snapshot()) -- a
        `matches` row for `tick.fixture_id` must already exist.
        """
        now = time.time() if now is None else now
        await self._writer.enqueue(
            WriteRequest(
                table="market_ticks",
                query=_MARKET_TICK_INSERT_QUERY,
                params=(tick.fixture_id, tick.market_type, tick.outcome, tick.price, tick.timestamp, now),
            )
        )

    async def record_risk_metrics(self, risk: RiskMetrics, now: Optional[float] = None) -> None:
        """Persist RiskMetrics to shadow_analysis. Unset fields persist as SQL NULL.

        `shadow_analysis.fixture_id` carries the same FOREIGN KEY to
        `matches` as `snapshots` does (see record_snapshot()) -- a
        `matches` row for `risk.fixture_id` must already exist.
        """
        now = time.time() if now is None else now
        await self._writer.enqueue(
            WriteRequest(
                table="shadow_analysis",
                query=_SHADOW_ANALYSIS_INSERT_QUERY,
                params=(
                    risk.fixture_id,
                    risk.timestamp,
                    risk.model_confidence,
                    risk.uncertainty,
                    risk.volatility,
                    risk.simulated_exposure,
                    risk.simulated_drawdown,
                    risk.risk_score,
                    risk.calibration_error,
                    now,
                ),
            )
        )

    async def record_audit_result(self, result: AuditResult, now: Optional[float] = None) -> None:
        """Persist an AuditResult to audit_events."""
        now = time.time() if now is None else now
        await self._writer.enqueue(
            WriteRequest(
                table="audit_events",
                query=_AUDIT_EVENT_INSERT_QUERY,
                params=(
                    result.fixture_id,
                    result.timestamp,
                    result.verdict.value,
                    canonical_json(result.checks_passed),
                    canonical_json(result.reasons),
                    result.payload_checksum,
                    now,
                ),
            )
        )

    async def record_journal_event(self, event: JournalEvent, now: Optional[float] = None) -> None:
        """Persist a JournalEvent to journal_events."""
        now = time.time() if now is None else now
        await self._writer.enqueue(
            WriteRequest(
                table="journal_events",
                query=_JOURNAL_EVENT_INSERT_QUERY,
                params=(event.event_type.value, event.fixture_id, event.timestamp, canonical_json(event.details), now),
            )
        )

    async def record_anomaly_event(self, event: AnomalyEvent, now: Optional[float] = None) -> None:
        """Persist an AnomalyEvent through journal_events.

        storage/schema.sql has no dedicated anomaly table (confirmed
        during preflight); per the "no new contract unless proven
        necessary" default, an AnomalyEvent is recorded as a
        journal_events row using the existing JournalEventType.ANOMALY
        member rather than inventing a new table.
        """
        now = time.time() if now is None else now
        details = {"anomaly_type": event.anomaly_type, "details": event.details}
        await self._writer.enqueue(
            WriteRequest(
                table="journal_events",
                query=_JOURNAL_EVENT_INSERT_QUERY,
                params=(JournalEventType.ANOMALY.value, event.fixture_id, event.timestamp, canonical_json(details), now),
            )
        )

    async def record_kill_switch_event(self, event: KillSwitchEvent, now: Optional[float] = None) -> None:
        """Persist a KillSwitchEvent through journal_events (no dedicated table exists)."""
        now = time.time() if now is None else now
        details = {
            "previous_state": event.previous_state.value,
            "new_state": event.new_state.value,
            "reason": event.reason,
        }
        await self._writer.enqueue(
            WriteRequest(
                table="journal_events",
                query=_JOURNAL_EVENT_INSERT_QUERY,
                params=(JournalEventType.KILL_SWITCH.value, None, event.timestamp, canonical_json(details), now),
            )
        )

    async def record_quarantine_event(self, event: QuarantineEvent, now: Optional[float] = None) -> None:
        """Persist a models.events.QuarantineEvent to quarantine_events."""
        now = time.time() if now is None else now
        await self._writer.enqueue(
            WriteRequest(
                table="quarantine_events",
                query=_QUARANTINE_EVENT_INSERT_QUERY,
                params=(event.fixture_id, event.reason.value, event.timestamp, canonical_json(event.details), now),
            )
        )

    async def record_quarantine_record(self, record: QuarantineRecord, now: Optional[float] = None) -> None:
        """Persist a safety.quarantine.QuarantineRecord (the REAL runtime decision object).

        safety.quarantine.QuarantineReason and models.events.QuarantineReason
        are two distinct, non-overlapping enums (confirmed during
        preflight -- no production code ever produces the latter). This
        method persists whichever reason enum the caller actually has by
        storing its `.value` directly; no cross-enum mapping is invented.
        """
        now = time.time() if now is None else now
        await self._writer.enqueue(
            WriteRequest(
                table="quarantine_events",
                query=_QUARANTINE_EVENT_INSERT_QUERY,
                params=(record.fixture_id, record.reason.value, record.timestamp, canonical_json({"detail": record.detail}), now),
            )
        )
