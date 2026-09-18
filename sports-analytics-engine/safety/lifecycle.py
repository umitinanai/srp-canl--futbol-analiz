"""Explicit, minimal system lifecycle state machine.

Deliberately the smallest correct abstraction: a fixed transition
table plus a single asyncio.Lock. No state-machine framework, no
hidden global state -- each SystemLifecycle instance owns its own
state and lock.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional


class LifecycleState(str, Enum):
    """Explicit system lifecycle states."""

    CREATED = "CREATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class LifecycleTransitionError(RuntimeError):
    """Raised when an invalid lifecycle transition is attempted.

    An invalid transition is rejected loudly rather than silently
    succeeding or being coerced into a "nearest valid" state.
    """


#: The complete, explicit transition table. FAILED and STOPPED are
#: terminal: no transition out of them is valid (a new SystemLifecycle
#: instance must be created to run again).
_VALID_TRANSITIONS: Dict[LifecycleState, FrozenSet[LifecycleState]] = {
    LifecycleState.CREATED: frozenset({LifecycleState.STARTING}),
    LifecycleState.STARTING: frozenset({LifecycleState.RUNNING, LifecycleState.FAILED}),
    LifecycleState.RUNNING: frozenset({LifecycleState.STOPPING, LifecycleState.FAILED}),
    LifecycleState.STOPPING: frozenset({LifecycleState.STOPPED, LifecycleState.FAILED}),
    LifecycleState.STOPPED: frozenset(),
    LifecycleState.FAILED: frozenset(),
}


@dataclass(frozen=True)
class LifecycleError:
    """Record of the error that caused (or accompanied) a FAILED transition."""

    error_type: str
    message: str
    timestamp: float


class SystemLifecycle:
    """A single system's explicit lifecycle state, owned by its instance.

    Concurrency-safe: all state transitions are serialized through a
    single per-instance asyncio.Lock. Since no transition performs any
    `await` while holding the lock, a transition can never be
    interrupted mid-flight by cancellation while the lock is held --
    cancellation can only occur while *waiting* for the lock, before
    any state mutation happens, which is always safe.
    """

    def __init__(self) -> None:
        self._state: LifecycleState = LifecycleState.CREATED
        self._lock = asyncio.Lock()
        self._last_error: Optional[LifecycleError] = None

    @property
    def state(self) -> LifecycleState:
        """Return the current lifecycle state."""
        return self._state

    @property
    def last_error(self) -> Optional[LifecycleError]:
        """Return the error that accompanied the most recent FAILED transition, if any."""
        return self._last_error

    async def transition_to(self, new_state: LifecycleState) -> None:
        """Attempt an explicit state transition, validated against the transition table.

        Args:
            new_state: the target lifecycle state.

        Raises:
            LifecycleTransitionError: if new_state is not a valid
                transition from the current state. The state is left
                unchanged when this is raised.
        """
        async with self._lock:
            allowed = _VALID_TRANSITIONS.get(self._state, frozenset())
            if new_state not in allowed:
                raise LifecycleTransitionError(
                    f"Invalid lifecycle transition: {self._state.value} -> {new_state.value}"
                )
            self._state = new_state

    async def mark_failed(self, error_type: str, message: str) -> None:
        """Transition to FAILED and record the associated error, if a valid transition.

        Args:
            error_type: a short machine-readable category for the failure.
            message: a human-readable description of the failure.

        Raises:
            LifecycleTransitionError: if FAILED is not a valid
                transition from the current state (i.e. already
                STOPPED or FAILED).
        """
        await self.transition_to(LifecycleState.FAILED)
        self._last_error = LifecycleError(error_type, message, time.time())

    async def stop(self) -> None:
        """Idempotently drive the lifecycle towards STOPPED.

        Safe to call repeatedly and from CREATED, RUNNING, STOPPING,
        STOPPED, or FAILED: it is a no-op unless the state is currently
        RUNNING (or already STOPPING, in which case it simply finishes
        that transition). Calling stop() on a CREATED, STOPPED, or
        FAILED lifecycle is a harmless no-op rather than an error,
        satisfying "repeated shutdown calls harmless".
        """
        async with self._lock:
            if self._state == LifecycleState.RUNNING:
                self._state = LifecycleState.STOPPING
            elif self._state != LifecycleState.STOPPING:
                # CREATED, STOPPED, FAILED: nothing to do.
                return

        async with self._lock:
            if self._state == LifecycleState.STOPPING:
                self._state = LifecycleState.STOPPED
