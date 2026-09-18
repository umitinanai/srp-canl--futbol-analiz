"""Match state, snapshot and data-age related data contracts.

This module defines the core in-memory and wire representations for a
single football (soccer) fixture as it moves through the Vanguard Live
Quant Engine pipeline: raw provider snapshots, normalized match state,
and the bounded rolling history structures attached to each match.

No analytics, no I/O and no agent logic lives here. This module only
defines *data* and small, pure validation helpers on that data.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Optional


class MatchPhase(str, Enum):
    """Lifecycle phase of a tracked fixture.

    PHASE_1_SLEEP:   low-frequency polling, match not yet interesting.
    PHASE_2_TRACKING: medium-frequency polling, building baselines.
    PHASE_3_HUNT:     high-frequency polling, deep analytical snapshot.
    QUARANTINED:      match data has been flagged unsafe and is parked.
    INVALID:          match state is structurally invalid.
    COMPLETED:        fixture has finished and no further polling occurs.
    """

    PHASE_1_SLEEP = "PHASE_1_SLEEP"
    PHASE_2_TRACKING = "PHASE_2_TRACKING"
    PHASE_3_HUNT = "PHASE_3_HUNT"
    QUARANTINED = "QUARANTINED"
    INVALID = "INVALID"
    COMPLETED = "COMPLETED"


class DataAgeStatus(str, Enum):
    """Freshness classification of a snapshot based on its data_age."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    INVALID = "INVALID"


class DataAgeValidationError(ValueError):
    """Raised when a snapshot's timestamps are structurally invalid."""


def classify_data_age(
    data_age_seconds: float,
    green_threshold_seconds: float,
    yellow_threshold_seconds: float,
) -> DataAgeStatus:
    """Classify a data age value into GREEN / YELLOW / INVALID.

    Args:
        data_age_seconds: received_timestamp - event_timestamp, in seconds.
        green_threshold_seconds: inclusive upper bound for GREEN.
        yellow_threshold_seconds: inclusive upper bound for YELLOW.

    Returns:
        The DataAgeStatus classification.

    Raises:
        DataAgeValidationError: if data_age_seconds is negative, NaN
            or otherwise not a finite real number.
    """
    if data_age_seconds != data_age_seconds:  # NaN check without math import
        raise DataAgeValidationError("data_age_seconds is NaN")
    if data_age_seconds in (float("inf"), float("-inf")):
        raise DataAgeValidationError("data_age_seconds is infinite")
    if data_age_seconds < 0:
        raise DataAgeValidationError("data_age_seconds is negative")

    if data_age_seconds <= green_threshold_seconds:
        return DataAgeStatus.GREEN
    if data_age_seconds <= yellow_threshold_seconds:
        return DataAgeStatus.YELLOW
    return DataAgeStatus.INVALID


@dataclass(frozen=True)
class Team:
    """A single team participating in a fixture."""

    team_id: str
    name: str
    is_home: bool


@dataclass(frozen=True)
class Score:
    """Current score of a fixture."""

    home_goals: int
    away_goals: int

    def __post_init__(self) -> None:
        if self.home_goals < 0 or self.away_goals < 0:
            raise ValueError("Score components cannot be negative")


@dataclass(frozen=True)
class Snapshot:
    """A single normalized data point received from a provider.

    Two distinct SHA-256 digests are tracked, each with a different
    purpose:

    payload_checksum:
        Hash of the canonical payload content ONLY. Two snapshots with
        the same payload_checksum carry byte-identical content, even if
        they belong to different fixtures or different moments in time.
        This is a pure content-equality signal, used by the audit cache
        described in Section 22 (avoid re-auditing identical content).
        It is deliberately NOT unique and NOT used for duplicate
        prevention on its own, since the same content can legitimately
        recur (e.g. a 0-0 scoreline snapshot looks identical across many
        early-match polls for different fixtures).

    identity_checksum:
        Hash of ``f"{fixture_id}|{event_timestamp}|{payload_checksum}"``.
        This uniquely identifies *this specific snapshot occurrence* --
        i.e. "fixture X reported this exact content at this exact event
        time". This is the field enforced UNIQUE at the database level
        (storage/schema.sql) to satisfy the "no duplicate snapshots"
        requirement, and is what MatchState uses for in-memory dedup.

    Attributes:
        fixture_id: unique identifier of the fixture.
        event_timestamp: unix epoch seconds when the event actually occurred
            (as reported by the provider).
        received_timestamp: unix epoch seconds when this snapshot was
            received/normalized locally.
        payload: normalized provider payload (score, minute, stats, ...).
        payload_checksum: SHA-256 hex digest of the canonical payload
            content only. See class docstring.
        identity_checksum: SHA-256 hex digest identifying this specific
            snapshot occurrence. See class docstring.
    """

    fixture_id: str
    event_timestamp: float
    received_timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    payload_checksum: str = ""
    identity_checksum: str = ""

    @property
    def data_age(self) -> float:
        """Return received_timestamp - event_timestamp in seconds."""
        return self.received_timestamp - self.event_timestamp

    def with_checksum(self) -> "Snapshot":
        """Return a copy of this snapshot with both checksums computed and set.

        Returns:
            A new Snapshot with payload_checksum set to the hash of the
            canonical payload, and identity_checksum set to the hash of
            fixture_id + event_timestamp + payload_checksum.
        """
        canonical = canonical_json(self.payload)
        payload_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        identity_source = f"{self.fixture_id}|{self.event_timestamp}|{payload_digest}"
        identity_digest = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()
        return Snapshot(
            fixture_id=self.fixture_id,
            event_timestamp=self.event_timestamp,
            received_timestamp=self.received_timestamp,
            payload=self.payload,
            payload_checksum=payload_digest,
            identity_checksum=identity_digest,
        )


def canonical_json(payload: Dict[str, Any]) -> str:
    """Produce a deterministic canonical JSON string for hashing.

    Keys are sorted and separators are minimal so that semantically
    identical payloads always hash to the same SHA-256 digest.

    Args:
        payload: an arbitrary JSON-serializable dictionary.

    Returns:
        A canonical JSON string representation of payload.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def new_bounded_deque(maxlen: int) -> Deque[Any]:
    """Create a bounded deque, guarding against non-positive maxlen.

    Args:
        maxlen: maximum number of elements retained. Must be >= 1.

    Returns:
        An empty deque bounded to maxlen elements.

    Raises:
        ValueError: if maxlen is not a positive integer.
    """
    if maxlen < 1:
        raise ValueError("maxlen must be a positive integer")
    return deque(maxlen=maxlen)


@dataclass
class MatchState:
    """Mutable, bounded in-memory state tracked for a single fixture.

    All history-bearing fields are bounded deques so that RAM usage stays
    constant regardless of how long a fixture is tracked, in accordance
    with the 2GB RAM / 2 vCPU hardware target.
    """

    fixture_id: str
    league_id: str
    home_team: Team
    away_team: Team
    score: Score = field(default_factory=lambda: Score(0, 0))
    minute: int = 0
    phase: MatchPhase = MatchPhase.PHASE_1_SLEEP
    last_poll_timestamp: float = field(default_factory=time.time)
    phase_change_timestamp: float = field(default_factory=time.time)

    price_history: Deque[Any] = field(default_factory=lambda: new_bounded_deque(16))
    margin_history: Deque[Any] = field(default_factory=lambda: new_bounded_deque(16))
    timeline_1_15: Deque[Any] = field(default_factory=lambda: new_bounded_deque(16))
    ht_snap: Optional[Dict[str, Any]] = None

    rolling_stats: Dict[str, Any] = field(default_factory=dict)
    baselines: Dict[str, Any] = field(default_factory=dict)
    red_cards: Dict[str, int] = field(default_factory=lambda: {"home": 0, "away": 0})

    quarantine_counter: int = 0
    seen_identity_checksums: Deque[str] = field(default_factory=lambda: new_bounded_deque(64))

    def transition_phase(self, new_phase: MatchPhase, now: Optional[float] = None) -> None:
        """Transition this match to a new phase, updating timestamps.

        Args:
            new_phase: the phase to transition into.
            now: optional injected timestamp (for deterministic testing).
                Defaults to time.time().
        """
        self.phase = new_phase
        self.phase_change_timestamp = now if now is not None else time.time()

    def register_snapshot_identity(self, identity_checksum: str) -> bool:
        """Register a snapshot's identity_checksum for in-memory dedup.

        This is a fast, bounded, best-effort in-memory check used before
        a snapshot is even sent downstream. It is a complement to, not a
        replacement for, the authoritative UNIQUE constraint on
        snapshots.identity_checksum enforced at the database level
        (see storage/schema.sql), since this deque only retains the most
        recent entries.

        Args:
            identity_checksum: the Snapshot.identity_checksum to register.

        Returns:
            True if this identity_checksum was new and has been recorded,
            False if it was already present (i.e. a duplicate snapshot).
        """
        if identity_checksum in self.seen_identity_checksums:
            return False
        self.seen_identity_checksums.append(identity_checksum)
        return True

    def increment_quarantine(self) -> int:
        """Increment and return the quarantine counter for this match."""
        self.quarantine_counter += 1
        return self.quarantine_counter
