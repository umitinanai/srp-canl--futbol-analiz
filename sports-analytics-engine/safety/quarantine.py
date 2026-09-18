"""Minimal, deterministic, bounded fixture quarantine.

Quarantine is a control/isolation mechanism only -- it never inspects
or alters analytics results, and it never invents a threshold-based
policy (e.g. "quarantine after N errors"). A fixture is quarantined
exactly when the caller (safety.orchestrator) determines, from an
actual failure encountered while processing that specific fixture,
that continued processing of it is unsafe. This module only stores
that decision, bounded, and lets it be queried and reversed.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Optional

#: Default maximum number of distinct quarantined fixtures retained at
#: once. Deliberately bounded: under sustained, unusual conditions
#: where more than this many fixtures would need quarantining
#: simultaneously, the oldest quarantine record is evicted (and that
#: fixture would need to fail again to be re-quarantined) rather than
#: growing memory without limit.
DEFAULT_MAX_QUARANTINE_RECORDS = 256


class QuarantineReason(str, Enum):
    """Deterministic, fixture-scoped reasons a fixture may be quarantined.

    Intentionally limited to categories that map directly onto actual
    exceptions raised by the existing Stage 2B/2A contracts while
    processing one specific fixture -- no generic "too many errors"
    or "timeout" category is included, since those would require an
    invented threshold policy.
    """

    MALFORMED_EVENT = "MALFORMED_EVENT"
    STATE_PROCESSING_FAILURE = "STATE_PROCESSING_FAILURE"
    ANALYTICS_FAILURE = "ANALYTICS_FAILURE"


@dataclass(frozen=True)
class QuarantineRecord:
    """An immutable record of one fixture's quarantine."""

    fixture_id: str
    reason: QuarantineReason
    detail: str
    timestamp: float


class Quarantine:
    """Bounded, concurrency-safe store of currently-quarantined fixtures.

    At most `max_records` fixtures are tracked at once; the oldest
    (by insertion/update order) is evicted first when the bound would
    otherwise be exceeded (a small, fixed-size FIFO, not an unbounded
    set/dict).
    """

    def __init__(self, max_records: int = DEFAULT_MAX_QUARANTINE_RECORDS) -> None:
        """Initialize an empty, bounded quarantine store.

        Args:
            max_records: maximum number of distinct fixtures retained
                at once. Must be a positive integer.

        Raises:
            ValueError: if max_records is not a positive integer.
        """
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ValueError(f"max_records must be a positive int, got {max_records!r}")
        self._max_records = max_records
        self._records: Dict[str, QuarantineRecord] = {}
        self._order: Deque[str] = deque()
        self._lock = asyncio.Lock()

    def __len__(self) -> int:
        """Return the number of currently-quarantined fixtures."""
        return len(self._records)

    def is_quarantined(self, fixture_id: str) -> bool:
        """Return True if fixture_id is currently quarantined."""
        return fixture_id in self._records

    def get(self, fixture_id: str) -> Optional[QuarantineRecord]:
        """Return the current quarantine record for fixture_id, or None."""
        return self._records.get(fixture_id)

    def fixture_ids(self) -> tuple:
        """Return a sorted tuple of all currently-quarantined fixture ids."""
        return tuple(sorted(self._records.keys()))

    async def quarantine_fixture(
        self, fixture_id: str, reason: QuarantineReason, detail: str = ""
    ) -> QuarantineRecord:
        """Quarantine (or re-quarantine, with updated detail) one fixture.

        Args:
            fixture_id: the fixture to quarantine.
            reason: the deterministic, fixture-scoped reason.
            detail: optional human-readable detail (e.g. the original
                exception message).

        Returns:
            The QuarantineRecord that was stored.
        """
        record = QuarantineRecord(fixture_id=fixture_id, reason=reason, detail=detail, timestamp=time.time())
        async with self._lock:
            if fixture_id in self._records:
                self._order.remove(fixture_id)
            elif len(self._order) >= self._max_records:
                evicted = self._order.popleft()
                self._records.pop(evicted, None)
            self._records[fixture_id] = record
            self._order.append(fixture_id)
        return record

    async def release(self, fixture_id: str) -> bool:
        """Remove a fixture from quarantine, if present.

        Args:
            fixture_id: the fixture to release.

        Returns:
            True if the fixture was quarantined and is now released,
            False if it was not quarantined.
        """
        async with self._lock:
            if fixture_id not in self._records:
                return False
            self._records.pop(fixture_id, None)
            self._order.remove(fixture_id)
            return True
