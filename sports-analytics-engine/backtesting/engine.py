"""Independent backtesting/replay engine.

Replays historical (lambda_home, lambda_away, outcome) rows
chronologically through the SAME analytics.poisson implementation used
by the live engine -- there is no separate/duplicate mathematical
implementation here, per the "one authoritative implementation per
concept" requirement. Pandas is used only at the CSV/DataFrame ingestion
boundary of this module; it is never required by (or leaked into) the
live hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from analytics.poisson import (
    away_win_probability,
    draw_probability,
    home_win_probability,
    score_probability_matrix,
)
from backtesting.metrics import brier_score, calibration_error, log_loss

#: Valid outcome labels for a backtest row.
_VALID_OUTCOMES = frozenset({"HOME", "DRAW", "AWAY"})


class BacktestValidationError(ValueError):
    """Raised when backtest input data or configuration is invalid."""


@dataclass(frozen=True)
class BacktestRow:
    """One replayed historical observation.

    Attributes:
        fixture_id: identifier of the historical fixture.
        lambda_home: home scoring intensity used for the prediction.
        lambda_away: away scoring intensity used for the prediction.
        outcome: the actual observed result: "HOME", "DRAW", or "AWAY".
        timestamp: optional chronological ordering key. Rows are sorted
            ascending by this value when present on all rows.
    """

    fixture_id: str
    lambda_home: float
    lambda_away: float
    outcome: str
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class BacktestResult:
    """Aggregated outcome of a backtest replay."""

    rows_evaluated: int
    home_win_brier: float
    draw_brier: float
    away_win_brier: float
    log_loss_value: float
    calibration_error_value: float
    predicted_probabilities: Tuple[Tuple[float, float, float], ...]
    outcomes: Tuple[str, ...]


def _validate_row(raw_row: Mapping[str, Any]) -> BacktestRow:
    """Validate and coerce a single raw row into a BacktestRow.

    Args:
        raw_row: a mapping with at least "fixture_id", "lambda_home",
            "lambda_away", and "outcome" keys, and optionally
            "timestamp".

    Returns:
        A validated BacktestRow.

    Raises:
        BacktestValidationError: if any required field is missing or invalid.
    """
    for required in ("fixture_id", "lambda_home", "lambda_away", "outcome"):
        if required not in raw_row or raw_row[required] is None:
            raise BacktestValidationError(f"row is missing required field {required!r}")

    outcome = str(raw_row["outcome"]).strip().upper()
    if outcome not in _VALID_OUTCOMES:
        raise BacktestValidationError(
            f"outcome must be one of {sorted(_VALID_OUTCOMES)}, got {raw_row['outcome']!r}"
        )

    lambda_home = _validate_finite("lambda_home", raw_row["lambda_home"])
    if lambda_home < 0:
        raise BacktestValidationError(f"lambda_home must be non-negative, got {lambda_home}")
    lambda_away = _validate_finite("lambda_away", raw_row["lambda_away"])
    if lambda_away < 0:
        raise BacktestValidationError(f"lambda_away must be non-negative, got {lambda_away}")

    timestamp = raw_row.get("timestamp")
    if timestamp is not None:
        timestamp = _validate_finite("timestamp", timestamp)

    return BacktestRow(
        fixture_id=str(raw_row["fixture_id"]),
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        outcome=outcome,
        timestamp=timestamp,
    )


def _validate_finite(name: str, value: Any) -> float:
    """Validate that value is a finite, non-NaN real number.

    Args:
        name: name of the field, used in error messages.
        value: the value to validate.

    Returns:
        value, coerced to float.

    Raises:
        BacktestValidationError: if value is not numeric, NaN, or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BacktestValidationError(f"{name} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value):
        raise BacktestValidationError(f"{name} is NaN")
    if math.isinf(value):
        raise BacktestValidationError(f"{name} is infinite")
    return value


def run_backtest(rows: Iterable[Mapping[str, Any]]) -> BacktestResult:
    """Replay historical rows chronologically through the Poisson analytics engine.

    Args:
        rows: an iterable of raw row mappings, each containing at least
            "fixture_id", "lambda_home", "lambda_away", "outcome" (one
            of "HOME"/"DRAW"/"AWAY"), and optionally "timestamp" for
            chronological ordering.

    Returns:
        A BacktestResult with per-outcome Brier scores, overall log
        loss, calibration error, and the raw predicted-probability /
        outcome pairs.

    Raises:
        BacktestValidationError: if rows is empty or any row is invalid.
    """
    validated_rows = [_validate_row(raw_row) for raw_row in rows]
    if len(validated_rows) == 0:
        raise BacktestValidationError("rows must contain at least one observation")

    if all(row.timestamp is not None for row in validated_rows):
        validated_rows.sort(key=lambda row: row.timestamp)

    predicted_probabilities = []
    outcomes = []
    home_predictions, home_outcomes = [], []
    draw_predictions, draw_outcomes = [], []
    away_predictions, away_outcomes = [], []
    log_loss_predictions, log_loss_outcomes = [], []

    for row in validated_rows:
        matrix = score_probability_matrix(row.lambda_home, row.lambda_away)
        p_home = home_win_probability(matrix)
        p_draw = draw_probability(matrix)
        p_away = away_win_probability(matrix)

        predicted_probabilities.append((p_home, p_draw, p_away))
        outcomes.append(row.outcome)

        home_predictions.append(p_home)
        home_outcomes.append(1 if row.outcome == "HOME" else 0)
        draw_predictions.append(p_draw)
        draw_outcomes.append(1 if row.outcome == "DRAW" else 0)
        away_predictions.append(p_away)
        away_outcomes.append(1 if row.outcome == "AWAY" else 0)

        actual_probability = {"HOME": p_home, "DRAW": p_draw, "AWAY": p_away}[row.outcome]
        log_loss_predictions.append(actual_probability)
        log_loss_outcomes.append(1)

    return BacktestResult(
        rows_evaluated=len(validated_rows),
        home_win_brier=brier_score(home_predictions, home_outcomes),
        draw_brier=brier_score(draw_predictions, draw_outcomes),
        away_win_brier=brier_score(away_predictions, away_outcomes),
        log_loss_value=log_loss(log_loss_predictions, log_loss_outcomes),
        calibration_error_value=calibration_error(home_predictions, home_outcomes),
        predicted_probabilities=tuple(predicted_probabilities),
        outcomes=tuple(outcomes),
    )


def run_backtest_from_csv(path: str) -> BacktestResult:
    """Load a CSV file and run a backtest against its rows.

    Expects columns: fixture_id, lambda_home, lambda_away, outcome, and
    optionally timestamp. Pandas is used here (CSV ingestion boundary)
    only -- see module docstring.

    Args:
        path: filesystem path to the CSV file.

    Returns:
        A BacktestResult.

    Raises:
        BacktestValidationError: if the file is empty or any row is invalid.
    """
    import pandas as pd

    dataframe = pd.read_csv(path)
    return run_backtest_from_dataframe(dataframe)


def run_backtest_from_dataframe(dataframe: Any) -> BacktestResult:
    """Run a backtest against a pandas DataFrame of historical rows.

    Args:
        dataframe: a pandas DataFrame with columns fixture_id,
            lambda_home, lambda_away, outcome, and optionally timestamp.

    Returns:
        A BacktestResult.

    Raises:
        BacktestValidationError: if the DataFrame is empty or any row is invalid.
    """
    records: Sequence[Mapping[str, Any]] = dataframe.to_dict(orient="records")
    if len(records) == 0:
        raise BacktestValidationError("dataframe must contain at least one row")
    return run_backtest(records)
