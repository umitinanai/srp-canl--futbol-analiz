"""League configuration: regulation minutes and identifying metadata.

Poisson lambda scaling depends on regulation_minutes, so this is kept
as an explicit, testable configuration table rather than a hard-coded
constant scattered across the analytics layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

DEFAULT_REGULATION_MINUTES: int = 90


@dataclass(frozen=True)
class LeagueConfig:
    """Static configuration describing a supported league.

    Attributes:
        league_id: stable unique identifier for the league.
        name: human-readable league name.
        regulation_minutes: standard match length in minutes (usually 90).
        country: country or region the league belongs to.
    """

    league_id: str
    name: str
    regulation_minutes: int = DEFAULT_REGULATION_MINUTES
    country: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if self.regulation_minutes <= 0:
            raise ValueError("regulation_minutes must be positive")


_LEAGUES: Dict[str, LeagueConfig] = {
    "GENERIC_FOOTBALL": LeagueConfig(
        league_id="GENERIC_FOOTBALL",
        name="Generic Football League",
        regulation_minutes=90,
        country="GLOBAL",
    ),
    "PREMIER_LEAGUE": LeagueConfig(
        league_id="PREMIER_LEAGUE",
        name="English Premier League",
        regulation_minutes=90,
        country="ENGLAND",
    ),
    "SUPER_LIG": LeagueConfig(
        league_id="SUPER_LIG",
        name="Turkiye Super Lig",
        regulation_minutes=90,
        country="TURKIYE",
    ),
}


def get_league(league_id: str) -> LeagueConfig:
    """Return the LeagueConfig for a given league_id.

    Args:
        league_id: identifier of the league to look up.

    Returns:
        The matching LeagueConfig.

    Raises:
        KeyError: if league_id is not a registered league. Unknown
            leagues must be registered explicitly rather than silently
            defaulting, so that regulation_minutes assumptions are
            always intentional.
    """
    if league_id not in _LEAGUES:
        raise KeyError(f"Unknown league_id: {league_id!r}")
    return _LEAGUES[league_id]


def list_leagues() -> Dict[str, LeagueConfig]:
    """Return a copy of the full league registry."""
    return dict(_LEAGUES)


def register_league(config: LeagueConfig) -> None:
    """Register or overwrite a league configuration.

    Args:
        config: the LeagueConfig to register under its own league_id.
    """
    _LEAGUES[config.league_id] = config
