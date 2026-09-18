"""Programmatic (non-UI) observability types for the Stage 3 control surface.

Machine-readable, typed, deterministic status snapshots only -- no
dashboard, no web/UI surface. See safety.orchestrator.Orchestrator.status()
for the function that assembles a SystemStatus from live components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from data.provider import ProviderConnectionState
from safety.kill_switch import KillSwitchState
from safety.lifecycle import LifecycleState


@dataclass(frozen=True)
class ErrorRecord:
    """A single observed failure, retained for operational diagnosis.

    Attributes:
        error_type: a short machine-readable category (e.g.
            "provider_failure", "analytics_failure").
        message: a human-readable description of the failure.
        timestamp: unix epoch seconds when the failure was recorded.
        source: which layer/component the failure originated from
            (e.g. "provider", "state", "analytics", "orchestrator").
        fixture_id: the affected fixture, if the failure was
            fixture-scoped; None for system/provider-scoped failures.
    """

    error_type: str
    message: str
    timestamp: float
    source: str
    fixture_id: Optional[str] = None


@dataclass(frozen=True)
class SystemStatus:
    """A single, typed, deterministic snapshot of Stage 3's control surface.

    Attributes:
        lifecycle_state: the system's current LifecycleState.
        provider_state: the tracked provider's current
            ProviderConnectionState, or None if no provider is
            associated with this orchestrator.
        kill_switch_state: the current KillSwitchState.
        quarantined_fixture_count: number of fixtures currently quarantined.
        last_error: the most recently recorded ErrorRecord, or None if
            no failure has been observed.
    """

    lifecycle_state: LifecycleState
    provider_state: Optional[ProviderConnectionState]
    kill_switch_state: KillSwitchState
    quarantined_fixture_count: int
    last_error: Optional[ErrorRecord]
