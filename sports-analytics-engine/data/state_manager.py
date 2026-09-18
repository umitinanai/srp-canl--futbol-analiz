"""In-memory live match state, recalculation policy, and match screening.

This is the live hot path's source of truth. It intentionally does not
touch SQLite: Stage 1.1's Database/DBWriter remain available for
persistence/audit elsewhere in the system, but the live per-event
update path here only ever touches bounded in-memory structures (dict,
dataclasses, bounded deque), per the "live data is not a database hot
path" requirement.

Multiple simultaneous matches are supported via a plain dict keyed by
fixture_id; each match's state and rolling history are fully
independent of every other match's.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Deque, Dict, Mapping, Optional, Tuple
from collections import deque

from analytics.regime import DEFAULT_REGIME_THRESHOLDS, RegimeThresholds, detect_regime_change
from models.match import canonical_json

#: The six canonical pressure-component names (mirrors
#: config.settings.EXPECTED_PRESSURE_WEIGHT_KEYS) tracked per team side.
_PRESSURE_COMPONENT_NAMES: Tuple[str, ...] = (
    "shots",
    "shots_on_target",
    "dangerous_attacks",
    "corners",
    "possession_changes",
    "xg_proxy",
)

#: Default bound on rolling per-match history (pressure/momentum
#: observation deques). Keeps memory usage constant regardless of match
#: duration, matching the Stage 1.1 bounded-history convention.
DEFAULT_HISTORY_MAXLEN = 32

#: Bound on the number of recent event fingerprints/ids retained per
#: fixture for duplicate detection (see apply_event()). Deliberately a
#: separate, larger constant from DEFAULT_HISTORY_MAXLEN: fingerprints
#: are small strings (cheap to retain many of) and duplicate delivery
#: in real live feeds is typically back-to-back or within a short
#: retry window, so a generous window meaningfully reduces the
#: (documented, accepted) risk of a stale duplicate being re-accepted
#: after its fingerprint has aged out -- without resorting to an
#: unbounded set or a database-backed dedup store.
#:
#: IMPORTANT, DOCUMENTED LIMITATION: this is a fixed-size FIFO window,
#: not an unbounded/persistent dedup store. If a provider re-sends an
#: event whose fingerprint has already been evicted (i.e. more than
#: DEDUP_FINGERPRINT_WINDOW distinct events have been applied to this
#: fixture since), it MAY be re-accepted as if new. In practice this
#: residual risk is substantially mitigated by the independent
#: out-of-order/staleness check in apply_event(): a re-sent event whose
#: event_timestamp is older than the fixture's last applied
#: event_timestamp is rejected as stale regardless of whether its
#: fingerprint is still in the dedup window. The only unprotected case
#: is a resend that is both fingerprint-evicted AND not older than the
#: most recently applied event's timestamp -- an increasingly narrow
#: and unlikely combination as DEDUP_FINGERPRINT_WINDOW grows. Building
#: a database-backed or otherwise unbounded dedup store to close this
#: last gap is explicitly out of scope (see module docstring).
DEDUP_FINGERPRINT_WINDOW = 256


class StateManagerValidationError(ValueError):
    """Raised when an input to the state manager is invalid."""


def _validate_finite_non_negative(name: str, value: float) -> float:
    """Validate that value is a finite, non-negative real number.

    Args:
        name: name of the parameter, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        StateManagerValidationError: if value is invalid or negative.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateManagerValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise StateManagerValidationError(f"{name} must be finite, got {value}")
    if value < 0:
        raise StateManagerValidationError(f"{name} must be non-negative, got {value}")
    return value


@dataclass
class TeamFeatures:
    """Accumulated raw pressure-input statistics for one team side."""

    shots: float = 0.0
    shots_on_target: float = 0.0
    dangerous_attacks: float = 0.0
    corners: float = 0.0
    possession_changes: float = 0.0
    xg_proxy: float = 0.0

    def as_pressure_components(self) -> Dict[str, float]:
        """Return this team's stats as a components dict for calculate_pressure_index()."""
        return {name: getattr(self, name) for name in _PRESSURE_COMPONENT_NAMES}


@dataclass
class LiveMatchState:
    """Canonical, bounded in-memory state for one live fixture.

    Attributes:
        fixture_id: canonical fixture identifier.
        minute: current match minute.
        home_goals: current home team goal count.
        away_goals: current away team goal count.
        home_features: accumulated home-side pressure-input statistics.
        away_features: accumulated away-side pressure-input statistics.
        last_updated: unix epoch seconds of the most recent update.
        last_recalculation_minute: match minute at which the expensive
            model was last recalculated, or None if never.
        observation_count: total number of updates applied to this state.
        pressure_history: bounded (timestamp, pressure_index) history
            for the home side, used for rolling/regime calculations.
        mi_history: bounded (timestamp, momentum_indicator) history.
        last_event_timestamp: event_timestamp (not received_timestamp)
            of the most recent event actually applied via apply_event(),
            used to detect and reject out-of-order (stale) events.
        seen_event_fingerprints: bounded (maxlen=DEDUP_FINGERPRINT_WINDOW)
            set (as a deque) of identifiers for events already applied,
            used for duplicate detection by apply_event(). See
            DEDUP_FINGERPRINT_WINDOW and apply_event() for how
            fingerprints are derived when a provider does not supply an
            explicit event id, and for the documented bounded-window
            limitation.
    """

    fixture_id: str
    minute: int = 0
    home_goals: int = 0
    away_goals: int = 0
    home_features: TeamFeatures = field(default_factory=TeamFeatures)
    away_features: TeamFeatures = field(default_factory=TeamFeatures)
    last_updated: float = field(default_factory=time.time)
    last_recalculation_minute: Optional[int] = None
    observation_count: int = 0
    pressure_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_MAXLEN)
    )
    mi_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_MAXLEN)
    )
    last_event_timestamp: Optional[float] = None
    seen_event_fingerprints: Deque[str] = field(
        default_factory=lambda: deque(maxlen=DEDUP_FINGERPRINT_WINDOW)
    )

    def score_tuple(self) -> Tuple[int, int]:
        """Return the current (home_goals, away_goals) tuple."""
        return (self.home_goals, self.away_goals)


@dataclass(frozen=True)
class AnalyticsSnapshot:
    """An immutable, provider-independent snapshot ready for analytics consumption.

    This is the sole interface between the live state layer and the
    analytics layer: analytics functions must never read a mutable
    LiveMatchState directly, only a snapshot taken from one at a single
    point in time. Component mappings are wrapped in MappingProxyType so
    a caller cannot mutate them and silently corrupt a value that has
    already been handed to an analytics calculation.

    mi_history/pressure_history are captured as plain tuples (not the
    live deques) for the same reason: a snapshot must be a fully
    self-contained, immutable, serializable value that cannot change
    out from under an in-flight analytics calculation, and cannot be
    used to reach back into the mutable LiveMatchState it was derived
    from.
    """

    fixture_id: str
    minute: int
    home_goals: int
    away_goals: int
    home_components: Mapping[str, float]
    away_components: Mapping[str, float]
    timestamp: float
    mi_history: Tuple[Tuple[float, float], ...] = ()
    pressure_history: Tuple[Tuple[float, float], ...] = ()


@dataclass(frozen=True)
class ApplyEventResult:
    """Outcome of StateManager.apply_event()/apply_event_async().

    Attributes:
        state: the fixture's LiveMatchState after this call (unchanged
            if the event was a duplicate or stale).
        score_changed: True if this event changed home_goals and/or
            away_goals. Always False when applied is False.
        applied: True if the event's fields were actually merged into
            state. False if the event was rejected as a duplicate or as
            stale (out-of-order).
        is_duplicate: True if this event's identity/fingerprint had
            already been seen for this fixture.
        is_stale: True if this event's event_timestamp was older than
            the most recently applied event's event_timestamp for this
            fixture (out-of-order delivery).
    """

    state: LiveMatchState
    score_changed: bool
    applied: bool
    is_duplicate: bool
    is_stale: bool


class StateManager:
    """Owns and mutates LiveMatchState for every currently-tracked fixture.

    A plain dict keyed by fixture_id keeps matches fully independent of
    one another; nothing here bounds the *number* of concurrently
    tracked matches (that is a Stage 3+/orchestrator concern), but every
    per-match rolling history is individually bounded.
    """

    def __init__(self) -> None:
        self._matches: Dict[str, LiveMatchState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, fixture_id: str) -> asyncio.Lock:
        """Return (creating if necessary) the per-fixture lock for fixture_id.

        Locks are per-fixture rather than global, so concurrent updates
        to different fixtures never block one another.

        Args:
            fixture_id: the fixture identifier.

        Returns:
            The asyncio.Lock guarding mutations to this fixture's state.
        """
        lock = self._locks.get(fixture_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[fixture_id] = lock
        return lock

    def get(self, fixture_id: str) -> Optional[LiveMatchState]:
        """Retrieve the current state for a fixture, if tracked.

        Args:
            fixture_id: the fixture identifier.

        Returns:
            The LiveMatchState, or None if the fixture is not tracked.
        """
        return self._matches.get(fixture_id)

    def remove(self, fixture_id: str) -> bool:
        """Remove a fixture's state entirely (e.g. once a match completes).

        Args:
            fixture_id: the fixture identifier.

        Returns:
            True if a state was present and removed, False otherwise.
        """
        return self._matches.pop(fixture_id, None) is not None

    def fixture_ids(self) -> Tuple[str, ...]:
        """Return a sorted tuple of all currently-tracked fixture IDs."""
        return tuple(sorted(self._matches.keys()))

    def apply_update(
        self,
        fixture_id: str,
        canonical_fields: Mapping[str, float],
        timestamp: Optional[float] = None,
    ) -> Tuple[LiveMatchState, bool]:
        """Merge a canonical partial update onto a fixture's live state.

        Creates a new LiveMatchState on first use of a fixture_id.
        Fields absent from canonical_fields retain their previous
        values -- this is the explicit merge policy that lets partial
        live events (which rarely carry every statistic) accumulate
        correctly over time.

        Args:
            fixture_id: the fixture identifier.
            canonical_fields: a dict as produced by
                data.normalizer.normalize_raw_event() -- canonical keys
                only, any subset.
            timestamp: unix epoch seconds this update was applied.
                Defaults to the current time.

        Returns:
            A (state, score_changed) tuple: the updated LiveMatchState,
            and whether this update changed home_goals and/or
            away_goals relative to the prior state.

        Raises:
            StateManagerValidationError: if any provided field value is
                invalid.
        """
        if timestamp is None:
            timestamp = time.time()
        timestamp = _validate_finite_non_negative("timestamp", timestamp)

        state = self._matches.get(fixture_id)
        is_first_observation = state is None
        if state is None:
            state = LiveMatchState(fixture_id=fixture_id)
            self._matches[fixture_id] = state

        previous_score = state.score_tuple()

        if "minute" in canonical_fields:
            state.minute = int(_validate_finite_non_negative("minute", canonical_fields["minute"]))
        if "home_goals" in canonical_fields:
            state.home_goals = int(
                _validate_finite_non_negative("home_goals", canonical_fields["home_goals"])
            )
        if "away_goals" in canonical_fields:
            state.away_goals = int(
                _validate_finite_non_negative("away_goals", canonical_fields["away_goals"])
            )

        _apply_side_features(state.home_features, "home_", canonical_fields)
        _apply_side_features(state.away_features, "away_", canonical_fields)

        state.last_updated = timestamp
        state.observation_count += 1

        score_changed = (not is_first_observation) and (state.score_tuple() != previous_score)
        return state, score_changed

    def record_pressure_observation(
        self, fixture_id: str, timestamp: float, pressure_index: float
    ) -> None:
        """Append a (timestamp, pressure_index) point to a fixture's bounded history.

        Args:
            fixture_id: the fixture identifier. Must already be tracked.
            timestamp: unix epoch seconds of the observation.
            pressure_index: the pressure index value at this timestamp.

        Raises:
            StateManagerValidationError: if fixture_id is not tracked.
        """
        state = self._require_state(fixture_id)
        state.pressure_history.append((timestamp, pressure_index))

    def record_mi_observation(self, fixture_id: str, timestamp: float, mi_value: float) -> None:
        """Append a (timestamp, momentum_indicator) point to a fixture's bounded history.

        Args:
            fixture_id: the fixture identifier. Must already be tracked.
            timestamp: unix epoch seconds of the observation.
            mi_value: the momentum indicator value at this timestamp.

        Raises:
            StateManagerValidationError: if fixture_id is not tracked.
        """
        state = self._require_state(fixture_id)
        state.mi_history.append((timestamp, mi_value))

    def mark_recalculated(self, fixture_id: str, minute: int) -> None:
        """Record that the expensive model was just recalculated for a fixture.

        Args:
            fixture_id: the fixture identifier. Must already be tracked.
            minute: the match minute at which recalculation occurred.

        Raises:
            StateManagerValidationError: if fixture_id is not tracked.
        """
        state = self._require_state(fixture_id)
        state.last_recalculation_minute = minute

    def apply_event(
        self,
        fixture_id: str,
        canonical_fields: Mapping[str, float],
        event_timestamp: float,
        event_id: Optional[str] = None,
        received_timestamp: Optional[float] = None,
    ) -> ApplyEventResult:
        """Apply a live event with deduplication and out-of-order rejection.

        This is the recommended Stage 2B entry point for live provider
        events (apply_update() remains available unchanged for direct,
        order/dedup-agnostic use, e.g. backtesting replay of
        already-deduplicated, already-ordered historical rows).

        Deduplication: if event_id is given, it is used as the event's
        identity directly. If event_id is None, a deterministic
        fingerprint is derived from
        (fixture_id, event_timestamp, canonical_fields) via
        models.match.canonical_json + SHA-256, reusing the exact same
        canonicalization helper the Stage 1.1 Snapshot checksum uses.
        Either way, an identity already present in the fixture's bounded
        seen_event_fingerprints deque is rejected as a duplicate without
        mutating state.

        Out-of-order rejection: if this fixture has already applied an
        event with a newer event_timestamp, this event is rejected as
        stale without mutating state. This does not reorder or buffer
        stale events -- it simply prevents them from overwriting newer
        state, per the "lightweight, deterministic" design goal (no
        full event-sourcing system).

        Args:
            fixture_id: the fixture identifier.
            canonical_fields: a dict as produced by
                data.normalizer.normalize_raw_event().
            event_timestamp: unix epoch seconds when the event actually
                occurred (provider-reported), used for both
                out-of-order detection and duplicate fingerprinting.
            event_id: optional provider-supplied unique event identifier.
            received_timestamp: unix epoch seconds this event was
                received locally. Defaults to the current time. Used as
                the state's last_updated timestamp (see apply_update()).

        Returns:
            An ApplyEventResult describing what happened.

        Raises:
            StateManagerValidationError: if event_timestamp or any
                provided field value is invalid.
        """
        event_timestamp = _validate_finite_non_negative("event_timestamp", event_timestamp)

        state = self._matches.get(fixture_id)
        if state is None:
            state = LiveMatchState(fixture_id=fixture_id)
            self._matches[fixture_id] = state

        fingerprint = event_id if event_id is not None else _fingerprint_event(
            fixture_id, event_timestamp, canonical_fields
        )
        if fingerprint in state.seen_event_fingerprints:
            return ApplyEventResult(
                state=state, score_changed=False, applied=False,
                is_duplicate=True, is_stale=False,
            )

        if state.last_event_timestamp is not None and event_timestamp < state.last_event_timestamp:
            return ApplyEventResult(
                state=state, score_changed=False, applied=False,
                is_duplicate=False, is_stale=True,
            )

        state.seen_event_fingerprints.append(fingerprint)
        updated_state, score_changed = self.apply_update(
            fixture_id, canonical_fields, timestamp=received_timestamp
        )
        updated_state.last_event_timestamp = event_timestamp

        return ApplyEventResult(
            state=updated_state, score_changed=score_changed, applied=True,
            is_duplicate=False, is_stale=False,
        )

    async def apply_event_async(
        self,
        fixture_id: str,
        canonical_fields: Mapping[str, float],
        event_timestamp: float,
        event_id: Optional[str] = None,
        received_timestamp: Optional[float] = None,
    ) -> ApplyEventResult:
        """Concurrency-safe wrapper around apply_event() using a per-fixture lock.

        Safe to call concurrently (e.g. from multiple asyncio tasks
        handling the same or different fixtures): calls for the SAME
        fixture_id are serialized against one another, while calls for
        DIFFERENT fixtures never block each other (see _get_lock()).

        Args:
            fixture_id: the fixture identifier.
            canonical_fields: a dict as produced by
                data.normalizer.normalize_raw_event().
            event_timestamp: unix epoch seconds when the event occurred.
            event_id: optional provider-supplied unique event identifier.
            received_timestamp: unix epoch seconds this event was
                received locally. Defaults to the current time.

        Returns:
            An ApplyEventResult describing what happened.
        """
        async with self._get_lock(fixture_id):
            return self.apply_event(
                fixture_id, canonical_fields, event_timestamp, event_id, received_timestamp
            )

    def build_snapshot(self, fixture_id: str) -> AnalyticsSnapshot:
        """Build an immutable AnalyticsSnapshot from a fixture's current live state.

        Args:
            fixture_id: the fixture identifier. Must already be tracked.

        Returns:
            An AnalyticsSnapshot capturing this fixture's state at this
            moment. Later mutations to the fixture's LiveMatchState do
            not affect a snapshot already taken.

        Raises:
            StateManagerValidationError: if fixture_id is not tracked.
        """
        state = self._require_state(fixture_id)
        return AnalyticsSnapshot(
            fixture_id=state.fixture_id,
            minute=state.minute,
            home_goals=state.home_goals,
            away_goals=state.away_goals,
            home_components=MappingProxyType(dict(state.home_features.as_pressure_components())),
            away_components=MappingProxyType(dict(state.away_features.as_pressure_components())),
            timestamp=state.last_updated,
            mi_history=tuple(state.mi_history),
            pressure_history=tuple(state.pressure_history),
        )

    def _require_state(self, fixture_id: str) -> LiveMatchState:
        """Return the tracked state for fixture_id, or raise if untracked.

        Args:
            fixture_id: the fixture identifier.

        Returns:
            The LiveMatchState.

        Raises:
            StateManagerValidationError: if fixture_id is not tracked.
        """
        state = self._matches.get(fixture_id)
        if state is None:
            raise StateManagerValidationError(f"fixture_id {fixture_id!r} is not tracked")
        return state


def _fingerprint_event(
    fixture_id: str, event_timestamp: float, canonical_fields: Mapping[str, float]
) -> str:
    """Derive a deterministic identity fingerprint for an event lacking an explicit id.

    Reuses models.match.canonical_json for deterministic key ordering
    (the same canonicalization Stage 1.1's Snapshot checksum uses)
    before hashing, so semantically identical field dicts always
    fingerprint identically regardless of key insertion order.

    Args:
        fixture_id: the fixture identifier.
        event_timestamp: the event's reported event_timestamp.
        canonical_fields: the event's canonical field dict.

    Returns:
        A SHA-256 hex digest uniquely identifying this event occurrence.
    """
    payload = {"fixture_id": fixture_id, "event_timestamp": event_timestamp, **canonical_fields}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _apply_side_features(
    features: TeamFeatures, prefix: str, canonical_fields: Mapping[str, float]
) -> None:
    """Merge canonical shot/pressure-stat fields for one side onto TeamFeatures.

    Args:
        features: the TeamFeatures instance to mutate (home or away).
        prefix: "home_" or "away_".
        canonical_fields: canonical fields dict, as produced by
            data.normalizer.normalize_raw_event().
    """
    mapping = {
        f"{prefix}shots": "shots",
        f"{prefix}shots_on_target": "shots_on_target",
        f"{prefix}dangerous_attacks": "dangerous_attacks",
        f"{prefix}corners": "corners",
        f"{prefix}possession_changes": "possession_changes",
    }
    for canonical_key, attr_name in mapping.items():
        if canonical_key in canonical_fields:
            setattr(features, attr_name, canonical_fields[canonical_key])


@dataclass(frozen=True)
class RecalculationPolicy:
    """Centralized, explicit configuration for the recalculation policy.

    Attributes:
        elapsed_minutes_threshold: force a recalculation if at least
            this many match minutes have passed since the last
            recalculation, even with no other material signal.
        regime_thresholds: thresholds passed through to
            analytics.regime.detect_regime_change() for momentum/
            pressure/shot-quality-trend based triggers.
    """

    elapsed_minutes_threshold: float = 5.0
    regime_thresholds: RegimeThresholds = DEFAULT_REGIME_THRESHOLDS

    def __post_init__(self) -> None:
        if (
            isinstance(self.elapsed_minutes_threshold, bool)
            or not isinstance(self.elapsed_minutes_threshold, (int, float))
            or math.isnan(self.elapsed_minutes_threshold)
            or math.isinf(self.elapsed_minutes_threshold)
            or self.elapsed_minutes_threshold <= 0
        ):
            raise StateManagerValidationError(
                "elapsed_minutes_threshold must be a finite, strictly positive number"
            )


#: Default, documented recalculation policy. See RecalculationPolicy
#: for the explicit configuration this policy depends on.
DEFAULT_RECALCULATION_POLICY = RecalculationPolicy()


def should_recalculate(
    is_first_observation: bool,
    score_changed: bool,
    elapsed_minutes_since_last_recalc: Optional[float],
    z_mi: Optional[float] = None,
    pressure_acceleration: Optional[float] = None,
    shot_quality_trend: Optional[float] = None,
    policy: RecalculationPolicy = DEFAULT_RECALCULATION_POLICY,
) -> bool:
    """Decide whether the expensive analytical model should be recalculated.

    This is a pure, deterministic function -- it does not read or
    mutate any StateManager/LiveMatchState directly, so it is trivially
    unit-testable and reusable from any calling context (live pipeline,
    backtesting replay, etc.).

    Truth table (first matching rule wins):
        1. is_first_observation           -> True
        2. score_changed                  -> True
        3. elapsed_minutes_since_last_recalc
           is not None and
           >= policy.elapsed_minutes_threshold -> True
        4. otherwise, delegate to analytics.regime.detect_regime_change()
           using z_mi / pressure_acceleration / shot_quality_trend

    Args:
        is_first_observation: True if this is the first update ever
            observed for the fixture.
        score_changed: True if this update changed home_goals and/or
            away_goals.
        elapsed_minutes_since_last_recalc: match minutes elapsed since
            the last recalculation, or None if unknown/not yet
            recalculated.
        z_mi: current momentum z-score, or None if unavailable.
        pressure_acceleration: current PAI value, or None if unavailable.
        shot_quality_trend: current shot-quality trend value, or None
            if unavailable.
        policy: the RecalculationPolicy configuration to evaluate against.

    Returns:
        True if a full analytical recalculation is warranted, False
        otherwise (e.g. for a non-material update to an identical, or
        near-identical, state).
    """
    if is_first_observation:
        return True
    if score_changed:
        return True
    if (
        elapsed_minutes_since_last_recalc is not None
        and elapsed_minutes_since_last_recalc >= policy.elapsed_minutes_threshold
    ):
        return True
    return detect_regime_change(
        z_mi, pressure_acceleration, shot_quality_trend, policy.regime_thresholds
    )


def filter_material_matches(recalculation_flags: Mapping[str, bool]) -> Tuple[str, ...]:
    """Select the fixture IDs that qualify for deep (expensive) analysis.

    This performs screening only -- it does not rank or compare matches
    against one another. Zero, one, or many fixtures may qualify.

    Args:
        recalculation_flags: a mapping of fixture_id to the boolean
            result of should_recalculate() for that fixture's latest
            update.

    Returns:
        A sorted tuple of fixture IDs whose flag was True.
    """
    return tuple(sorted(fixture_id for fixture_id, flag in recalculation_flags.items() if flag))
