"""Central runtime settings, loaded from environment variables / .env.

No secret or credential is ever hard-coded here. All provider API keys
and other secrets are read exclusively from the environment at runtime
via load_settings(); if a required secret is absent, the corresponding
field is left as None and provider-integration code is responsible for
failing fast with a clear error at the point of use.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only if dependency missing
    _DOTENV_AVAILABLE = False


def _env_float(name: str, default: float) -> float:
    """Read an environment variable as float, falling back to default."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    """Read an environment variable as int, falling back to default."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_str(name: str, default: Optional[str]) -> Optional[str]:
    """Read an environment variable as string, falling back to default."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw


#: Exact set of component keys required in Settings.pressure_weights.
#: Mirrors the pressure input components defined in Section 12 of the
#: specification (shots, shots_on_target, dangerous_attacks, corners,
#: possession_changes, xg_proxy).
EXPECTED_PRESSURE_WEIGHT_KEYS = frozenset(
    {
        "shots",
        "shots_on_target",
        "dangerous_attacks",
        "corners",
        "possession_changes",
        "xg_proxy",
    }
)

#: Exact set of component keys required in Settings.sqs_weights.
#: Mirrors the SQS formula in Section 15 of the specification.
EXPECTED_SQS_WEIGHT_KEYS = frozenset(
    {
        "z_mi",
        "z_pai",
        "z_sqp",
        "z_mre",
        "data_quality",
        "calibration_confidence",
    }
)


def _validate_finite_positive(field_name: str, value: float) -> None:
    """Raise ValueError if value is not a finite, strictly positive number.

    Args:
        field_name: name of the field, used in the error message.
        value: the numeric value to validate.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {type(value)!r}")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite, got {value}")
    if value <= 0:
        raise ValueError(f"{field_name} must be strictly positive, got {value}")


def _validate_weight_map(
    field_name: str, weights: Mapping[str, float], expected_keys: frozenset
) -> Mapping[str, float]:
    """Validate a weight map's key set and that its values sum to ~1.0.

    Args:
        field_name: name of the field, used in error messages.
        weights: the weight mapping to validate.
        expected_keys: the exact set of keys required in weights.

    Returns:
        An immutable (MappingProxyType) copy of weights.

    Raises:
        ValueError: if the key set does not exactly match expected_keys,
            if any weight is not a finite number, or if the weights do
            not sum to 1.0 within a small numerical tolerance.
    """
    immutable_weights = MappingProxyType(dict(weights))

    actual_keys = frozenset(immutable_weights.keys())
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        raise ValueError(
            f"{field_name} keys must exactly equal {sorted(expected_keys)}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    for key, value in immutable_weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name}[{key!r}] must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"{field_name}[{key!r}] must be finite, got {value}")

    weight_sum = sum(immutable_weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"{field_name} must sum to 1.0, got {weight_sum}")

    return immutable_weights


@dataclass(frozen=True)
class Settings:
    """Immutable, validated snapshot of the system's runtime configuration."""

    env: str = "development"
    log_level: str = "INFO"

    db_path: str = "./data/vanguard.db"
    db_batch_size: int = 50
    db_queue_maxsize: int = 2000
    db_busy_timeout_ms: int = 5000

    phase1_interval_seconds: float = 270.0
    phase2_interval_seconds: float = 60.0
    phase3_interval_seconds: float = 15.0
    phase3_max_concurrent_matches: int = 5

    data_age_green_seconds: float = 5.0
    data_age_yellow_seconds: float = 10.0

    monte_carlo_simulations: int = 50_000
    monte_carlo_seed: int = 42

    # NOTE: Mapping[str, float] here, not Dict[str, float] -- __post_init__
    # always replaces whatever mapping is passed in (default or
    # user-supplied) with an immutable MappingProxyType wrapper, so no
    # caller can mutate a Settings instance's weights after construction.
    pressure_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "shots": 0.15,
            "shots_on_target": 0.25,
            "dangerous_attacks": 0.25,
            "corners": 0.10,
            "possession_changes": 0.10,
            "xg_proxy": 0.15,
        }
    )

    sqs_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "z_mi": 0.25,
            "z_pai": 0.20,
            "z_sqp": 0.20,
            "z_mre": 0.15,
            "data_quality": 0.10,
            "calibration_confidence": 0.10,
        }
    )

    worker_count: int = 4
    orchestrator_cycle_seconds: float = 15.0

    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000

    provider_api_key: Optional[str] = None
    provider_base_url: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_finite_positive("db_batch_size", self.db_batch_size)
        _validate_finite_positive("db_queue_maxsize", self.db_queue_maxsize)
        _validate_finite_positive("db_busy_timeout_ms", self.db_busy_timeout_ms)

        _validate_finite_positive("phase1_interval_seconds", self.phase1_interval_seconds)
        _validate_finite_positive("phase2_interval_seconds", self.phase2_interval_seconds)
        _validate_finite_positive("phase3_interval_seconds", self.phase3_interval_seconds)
        _validate_finite_positive(
            "phase3_max_concurrent_matches", self.phase3_max_concurrent_matches
        )

        if not math.isfinite(self.data_age_green_seconds) or self.data_age_green_seconds < 0:
            raise ValueError("data_age_green_seconds must be finite and non-negative")
        if not math.isfinite(self.data_age_yellow_seconds):
            raise ValueError("data_age_yellow_seconds must be finite")
        if self.data_age_yellow_seconds < self.data_age_green_seconds:
            raise ValueError(
                "data_age_yellow_seconds must be >= data_age_green_seconds"
            )

        if self.monte_carlo_simulations < 50_000:
            raise ValueError("monte_carlo_simulations must be >= 50000")

        _validate_finite_positive("worker_count", self.worker_count)
        _validate_finite_positive(
            "orchestrator_cycle_seconds", self.orchestrator_cycle_seconds
        )

        if not (1 <= self.dashboard_port <= 65535):
            raise ValueError(
                f"dashboard_port must be within 1..65535, got {self.dashboard_port}"
            )

        object.__setattr__(
            self,
            "pressure_weights",
            _validate_weight_map(
                "pressure_weights", self.pressure_weights, EXPECTED_PRESSURE_WEIGHT_KEYS
            ),
        )
        object.__setattr__(
            self,
            "sqs_weights",
            _validate_weight_map(
                "sqs_weights", self.sqs_weights, EXPECTED_SQS_WEIGHT_KEYS
            ),
        )


def load_settings(dotenv_path: Optional[str] = None) -> Settings:
    """Load Settings from environment variables, optionally loading a .env file.

    Args:
        dotenv_path: optional explicit path to a .env file. If None and
            python-dotenv is installed, the default .env discovery
            behavior of python-dotenv is used.

    Returns:
        A fully validated Settings instance.
    """
    if _DOTENV_AVAILABLE:
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path)
        else:
            load_dotenv()

    return Settings(
        env=_env_str("VANGUARD_ENV", "development"),
        log_level=_env_str("VANGUARD_LOG_LEVEL", "INFO"),
        db_path=_env_str("VANGUARD_DB_PATH", "./data/vanguard.db"),
        db_batch_size=_env_int("VANGUARD_DB_BATCH_SIZE", 50),
        db_queue_maxsize=_env_int("VANGUARD_DB_QUEUE_MAXSIZE", 2000),
        db_busy_timeout_ms=_env_int("VANGUARD_DB_BUSY_TIMEOUT_MS", 5000),
        phase1_interval_seconds=_env_float("VANGUARD_PHASE1_INTERVAL_SECONDS", 270.0),
        phase2_interval_seconds=_env_float("VANGUARD_PHASE2_INTERVAL_SECONDS", 60.0),
        phase3_interval_seconds=_env_float("VANGUARD_PHASE3_INTERVAL_SECONDS", 15.0),
        phase3_max_concurrent_matches=_env_int(
            "VANGUARD_PHASE3_MAX_CONCURRENT_MATCHES", 5
        ),
        data_age_green_seconds=_env_float("VANGUARD_DATA_AGE_GREEN_SECONDS", 5.0),
        data_age_yellow_seconds=_env_float("VANGUARD_DATA_AGE_YELLOW_SECONDS", 10.0),
        monte_carlo_simulations=_env_int("VANGUARD_MONTE_CARLO_SIMULATIONS", 50_000),
        monte_carlo_seed=_env_int("VANGUARD_MONTE_CARLO_SEED", 42),
        worker_count=_env_int("VANGUARD_WORKER_COUNT", 4),
        orchestrator_cycle_seconds=_env_float(
            "VANGUARD_ORCHESTRATOR_CYCLE_SECONDS", 15.0
        ),
        dashboard_host=_env_str("VANGUARD_DASHBOARD_HOST", "0.0.0.0"),
        dashboard_port=_env_int("VANGUARD_DASHBOARD_PORT", 8000),
        provider_api_key=_env_str("VANGUARD_PROVIDER_API_KEY", None),
        provider_base_url=_env_str("VANGUARD_PROVIDER_BASE_URL", None),
    )
