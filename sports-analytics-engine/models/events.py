"""Event contracts flowing through the Journal Agent and Safety layer.

All persistence for these events must go through the single DB writer
queue defined in storage/db_writer.py; this module only defines the
event shapes themselves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class JournalEventType(str, Enum):
    """Enumeration of all journalable event kinds in the system."""

    SNAPSHOT_RECEIVED = "snapshot_received"
    SNAPSHOT_REJECTED = "snapshot_rejected"
    DATA_AGE_VIOLATION = "data_age_violation"
    MODEL_CALCULATION = "model_calculation"
    ANOMALY = "anomaly"
    QUARANTINE = "quarantine"
    KILL_SWITCH = "kill_switch"
    WARNING = "warning"
    RECOVERY = "recovery"
    PHASE_TRANSITION = "phase_transition"
    AUDIT_RESULT = "audit_result"


class SystemRunState(str, Enum):
    """Global operational state of the engine, controlled by the kill switch."""

    RUNNING = "RUNNING"
    SAFE_MODE = "SAFE_MODE"
    STOPPED = "STOPPED"


class QuarantineReason(str, Enum):
    """Reasons a snapshot or fixture may be quarantined."""

    STALE = "stale"
    NEGATIVE_AGE = "negative_age"
    MALFORMED = "malformed"
    IMPOSSIBLE_SCORE = "impossible_score"
    INVALID_PROBABILITY = "invalid_probability"
    NAN_VALUE = "nan_value"
    INF_VALUE = "inf_value"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    PROVIDER_INCONSISTENCY = "provider_inconsistency"


@dataclass(frozen=True)
class JournalEvent:
    """A single journalable event, ready to be enqueued to the DB writer."""

    event_type: JournalEventType
    fixture_id: Optional[str]
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuarantineEvent:
    """Record of a snapshot or fixture being placed into quarantine."""

    fixture_id: str
    reason: QuarantineReason
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KillSwitchEvent:
    """Record of a global run-state transition triggered by the kill switch."""

    previous_state: SystemRunState
    new_state: SystemRunState
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class AnomalyEvent:
    """Record of an anomaly detected by the Auditor Agent or Safety layer."""

    fixture_id: Optional[str]
    anomaly_type: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
