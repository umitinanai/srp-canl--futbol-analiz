"""Minimal, explicit kill switch.

A one-way control: once activated, it stays activated for the lifetime
of the instance (a fresh KillSwitch is created for a fresh system run,
mirroring SystemLifecycle's one-shot design). No external
coordination, no distributed state -- purely local, in-process control.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class KillSwitchState(str, Enum):
    """Observable kill-switch state."""

    ARMED = "ARMED"
    ACTIVATED = "ACTIVATED"


class KillSwitchActivatedError(RuntimeError):
    """Raised when new processing is attempted after the kill switch has activated."""


@dataclass(frozen=True)
class KillSwitchActivation:
    """Record of when and why the kill switch was activated."""

    reason: str
    timestamp: float


class KillSwitch:
    """A deterministic, idempotent, concurrency-safe one-way kill switch.

    activate() is safe to call repeatedly and concurrently: exactly one
    caller's activation is recorded (the first to acquire the lock);
    all others observe ACTIVATED and no-op. check() is a cheap,
    lock-free read suitable for calling before dispatching every new
    unit of work.
    """

    def __init__(self) -> None:
        self._state: KillSwitchState = KillSwitchState.ARMED
        self._lock = asyncio.Lock()
        self._activation: Optional[KillSwitchActivation] = None

    @property
    def state(self) -> KillSwitchState:
        """Return the current kill-switch state."""
        return self._state

    @property
    def is_activated(self) -> bool:
        """Return True if the kill switch has been activated."""
        return self._state == KillSwitchState.ACTIVATED

    @property
    def activation(self) -> Optional[KillSwitchActivation]:
        """Return the activation record, or None if never activated."""
        return self._activation

    async def activate(self, reason: str = "") -> bool:
        """Activate the kill switch, idempotently and concurrency-safely.

        Args:
            reason: a short human-readable reason for the activation.
                Only the first successful activation's reason is
                retained.

        Returns:
            True if this call performed the activation (i.e. the switch
            was ARMED beforehand), False if it was already ACTIVATED
            (a harmless no-op).
        """
        async with self._lock:
            if self._state == KillSwitchState.ACTIVATED:
                return False
            self._state = KillSwitchState.ACTIVATED
            self._activation = KillSwitchActivation(reason=reason, timestamp=time.time())
            return True

    def check(self) -> None:
        """Raise if the kill switch is activated; otherwise return normally.

        Intended to be called immediately before dispatching any new
        unit of work (e.g. before processing the next event for a
        fixture). Does not affect work already in flight.

        Raises:
            KillSwitchActivatedError: if the kill switch has been activated.
        """
        if self._state == KillSwitchState.ACTIVATED:
            reason = self._activation.reason if self._activation else ""
            raise KillSwitchActivatedError(reason or "kill switch activated")
