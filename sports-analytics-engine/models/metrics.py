"""Result contracts produced by the analytics and agent layers.

These dataclasses are the stable "wire format" passed between agents
(Quant Agent -> Auditor Agent -> Journal Agent -> DB writer) and between
agents and the dashboard. Stage 2+ will populate these via real
analytics implementations; this module only defines the shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class AuditVerdict(str, Enum):
    """Outcome of an independent audit pass over a QuantResult."""

    VALID = "VALID"
    INVALID = "INVALID"
    UNDEFINED = "UNDEFINED"


@dataclass(frozen=True)
class QuantResult:
    """Aggregated output of the Quant Agent for one fixture at one instant.

    All fields default to None to allow partial construction during
    incremental pipeline stages; Stage 2's SQS implementation enforces
    an explicit missing-data policy rather than silently zero-filling.
    """

    fixture_id: str
    timestamp: float

    lambda_home: Optional[float] = None
    lambda_away: Optional[float] = None
    home_win_probability: Optional[float] = None
    draw_probability: Optional[float] = None
    away_win_probability: Optional[float] = None

    monte_carlo_home_win: Optional[float] = None
    monte_carlo_draw: Optional[float] = None
    monte_carlo_away_win: Optional[float] = None
    monte_carlo_simulations: Optional[int] = None

    mi_rate: Optional[float] = None
    z_mi: Optional[float] = None

    pressure_index: Optional[float] = None
    pressure_acceleration: Optional[float] = None

    xg_proxy: Optional[float] = None
    shot_quality_proxy: Optional[float] = None

    market_reaction_elasticity: Optional[float] = None
    fair_probabilities: Dict[str, float] = field(default_factory=dict)

    signal_quality_score: Optional[float] = None
    data_quality: Optional[float] = None
    calibration_confidence: Optional[float] = None

    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditResult:
    """Independent verification outcome for a given QuantResult."""

    fixture_id: str
    timestamp: float
    verdict: AuditVerdict
    checks_passed: Dict[str, bool] = field(default_factory=dict)
    reasons: Dict[str, str] = field(default_factory=dict)
    payload_checksum: str = ""


@dataclass(frozen=True)
class RiskMetrics:
    """Historical/simulation-only risk analytics for a fixture.

    No field in this dataclass represents a real trade, a real capital
    allocation, or an executable order.
    """

    fixture_id: str
    timestamp: float
    model_confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    volatility: Optional[float] = None
    simulated_exposure: Optional[float] = None
    simulated_drawdown: Optional[float] = None
    risk_score: Optional[float] = None
    calibration_error: Optional[float] = None
