"""Normalize provider-specific raw events into canonical fields.

Provider-specific field names must stop here. Everything downstream of
normalize_raw_event() consumes only the canonical key set defined by
CANONICAL_FIELD_ALIASES, regardless of which provider produced the
original event. This module also builds the canonical Stage 1.1
Snapshot object (reusing models.match.Snapshot and
models.match.classify_data_age directly, rather than duplicating that
contract).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Tuple

from models.match import DataAgeStatus, Snapshot, classify_data_age

#: Canonical key -> accepted raw provider field name aliases. A raw
#: event only needs to contain SOME of these keys (partial live updates
#: are normal and expected); normalize_raw_event() returns only the
#: canonical keys that were actually found, so the caller (the state
#: manager) can merge partial updates onto existing state rather than
#: requiring every field on every single event.
CANONICAL_FIELD_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "minute": ("minute", "time_min", "clock_minute", "match_minute"),
    "home_goals": ("home_goals", "score_home", "homeScore", "home_score"),
    "away_goals": ("away_goals", "score_away", "awayScore", "away_score"),
    "home_shots": ("home_shots", "shots_home", "homeShots"),
    "away_shots": ("away_shots", "shots_away", "awayShots"),
    "home_shots_on_target": (
        "home_shots_on_target", "sot_home", "homeShotsOnTarget", "home_sot",
    ),
    "away_shots_on_target": (
        "away_shots_on_target", "sot_away", "awayShotsOnTarget", "away_sot",
    ),
    "home_dangerous_attacks": (
        "home_dangerous_attacks", "dangerous_attacks_home", "homeDangerousAttacks",
    ),
    "away_dangerous_attacks": (
        "away_dangerous_attacks", "dangerous_attacks_away", "awayDangerousAttacks",
    ),
    "home_corners": ("home_corners", "corners_home", "homeCorners"),
    "away_corners": ("away_corners", "corners_away", "awayCorners"),
    "home_possession_changes": (
        "home_possession_changes", "possession_changes_home", "homePossessionChanges",
    ),
    "away_possession_changes": (
        "away_possession_changes", "possession_changes_away", "awayPossessionChanges",
    ),
}

#: Canonical keys that represent a non-negative count/stat and must
#: therefore reject negative values during normalization.
_NON_NEGATIVE_KEYS = frozenset(CANONICAL_FIELD_ALIASES.keys()) - {"minute"}


class NormalizationError(ValueError):
    """Raised when a raw provider event cannot be safely normalized."""


def normalize_raw_event(raw_event: Mapping[str, Any]) -> Dict[str, float]:
    """Extract and validate whichever canonical fields are present in a raw event.

    Only canonical keys actually found (via any of their configured
    aliases) in raw_event are included in the result -- this function
    never invents a default value for an absent field. Malformed values
    (non-numeric, NaN, Inf, or negative where a count is expected) are
    rejected outright rather than silently entering the pipeline.

    Args:
        raw_event: a raw, provider-shaped event dict.

    Returns:
        A dict containing only the canonical keys that were present and
        valid in raw_event.

    Raises:
        NormalizationError: if raw_event is not a mapping, or if any
            present field's value is invalid (non-numeric, NaN, Inf, or
            an out-of-domain negative count).
    """
    if not isinstance(raw_event, Mapping):
        raise NormalizationError(f"raw_event must be a mapping, got {type(raw_event)!r}")

    normalized: Dict[str, float] = {}
    for canonical_key, aliases in CANONICAL_FIELD_ALIASES.items():
        raw_value = _first_present(raw_event, aliases)
        if raw_value is None:
            continue
        value = _validate_numeric(canonical_key, raw_value)
        if canonical_key in _NON_NEGATIVE_KEYS and value < 0:
            raise NormalizationError(f"{canonical_key} must be non-negative, got {value}")
        normalized[canonical_key] = value

    return normalized


def _first_present(raw_event: Mapping[str, Any], aliases: Tuple[str, ...]) -> Any:
    """Return the first alias's value found in raw_event, or None.

    Args:
        raw_event: the raw event dict.
        aliases: candidate field names to look for, in priority order.

    Returns:
        The value of the first alias present (and not None) in
        raw_event, or None if no alias was found.
    """
    for alias in aliases:
        if alias in raw_event and raw_event[alias] is not None:
            return raw_event[alias]
    return None


def _validate_numeric(name: str, value: Any) -> float:
    """Validate that value is a finite, non-NaN real number and coerce to float.

    Args:
        name: canonical field name, used in error messages.
        value: the raw value to validate.

    Returns:
        value as a float.

    Raises:
        NormalizationError: if value is not numeric, is NaN, or is infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value):
        raise NormalizationError(f"{name} is NaN")
    if math.isinf(value):
        raise NormalizationError(f"{name} is infinite")
    return value


def build_snapshot(
    fixture_id: str,
    raw_event: Mapping[str, Any],
    event_timestamp: float,
    received_timestamp: Any = None,
) -> Snapshot:
    """Build a canonical, checksummed Stage 1.1 Snapshot from a raw provider event.

    Reuses models.match.Snapshot directly (including its identity/
    payload checksum split) rather than defining a competing snapshot
    representation.

    Args:
        fixture_id: canonical fixture identifier.
        raw_event: the raw, provider-shaped event dict (stored as the
            Snapshot's payload verbatim, unnormalized -- normalization
            for analytics consumption happens separately via
            normalize_raw_event()).
        event_timestamp: unix epoch seconds when the event occurred.
        received_timestamp: unix epoch seconds when the event was
            received locally. Defaults to the current time if not
            provided.

    Returns:
        A Snapshot with both payload_checksum and identity_checksum set.

    Raises:
        NormalizationError: if raw_event is not a mapping.
    """
    if not isinstance(raw_event, Mapping):
        raise NormalizationError(f"raw_event must be a mapping, got {type(raw_event)!r}")
    if received_timestamp is None:
        received_timestamp = time.time()

    snapshot = Snapshot(
        fixture_id=fixture_id,
        event_timestamp=event_timestamp,
        received_timestamp=received_timestamp,
        payload=dict(raw_event),
    )
    return snapshot.with_checksum()


def data_age_status(
    snapshot: Snapshot,
    green_threshold_seconds: float = 5.0,
    yellow_threshold_seconds: float = 10.0,
) -> DataAgeStatus:
    """Classify a Snapshot's freshness, reusing the Stage 1.1 data-age contract.

    Args:
        snapshot: the Snapshot to classify.
        green_threshold_seconds: inclusive upper bound for GREEN.
        yellow_threshold_seconds: inclusive upper bound for YELLOW.

    Returns:
        The DataAgeStatus classification (GREEN, YELLOW, or INVALID).
        Note: classify_data_age() itself raises on a negative data_age
        rather than returning INVALID; callers that need Stage 3's
        exact "negative => INVALID" behavior should catch
        DataAgeValidationError themselves -- Stage 2 intentionally does
        not re-implement Stage 3's quarantine/kill-switch handling here.
    """
    return classify_data_age(
        snapshot.data_age, green_threshold_seconds, yellow_threshold_seconds
    )
