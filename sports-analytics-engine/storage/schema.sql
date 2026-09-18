-- Vanguard Live Quant Engine v5.0 -- SQLite schema (WAL mode)
--
-- This schema stores analytical / simulation data only. There is no
-- execution ledger, order table, or real-money position table anywhere
-- in this schema.

CREATE TABLE IF NOT EXISTS matches (
    fixture_id          TEXT PRIMARY KEY,
    league_id           TEXT NOT NULL,
    home_team           TEXT NOT NULL,
    away_team           TEXT NOT NULL,
    phase               TEXT NOT NULL,
    minute              INTEGER NOT NULL DEFAULT 0,
    home_goals          INTEGER NOT NULL DEFAULT 0,
    away_goals          INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);

-- snapshots.payload_checksum: SHA-256 of canonical payload content only.
--   NOT unique -- identical content may legitimately recur across
--   different events/fixtures. Used for content-equality / audit-cache
--   lookups (Section 22), not for duplicate prevention.
-- snapshots.identity_checksum: SHA-256 of
--   fixture_id + event_timestamp + payload_checksum. Uniquely identifies
--   a single snapshot occurrence. UNIQUE constraint below is the
--   authoritative, DB-level duplicate-prevention mechanism required by
--   Section 20 ("no duplicate snapshots").
CREATE TABLE IF NOT EXISTS snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id          TEXT NOT NULL,
    event_timestamp     REAL NOT NULL,
    received_timestamp  REAL NOT NULL,
    data_age            REAL NOT NULL,
    data_age_status     TEXT NOT NULL,
    payload_checksum    TEXT NOT NULL,
    identity_checksum   TEXT NOT NULL,
    payload_json        TEXT NOT NULL,
    created_at          REAL NOT NULL,
    FOREIGN KEY (fixture_id) REFERENCES matches (fixture_id),
    UNIQUE (identity_checksum)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_fixture_id ON snapshots (fixture_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_payload_checksum ON snapshots (payload_checksum);

CREATE TABLE IF NOT EXISTS market_ticks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id          TEXT NOT NULL,
    market_type         TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    price                REAL NOT NULL,
    timestamp           REAL NOT NULL,
    created_at          REAL NOT NULL,
    FOREIGN KEY (fixture_id) REFERENCES matches (fixture_id)
);

CREATE INDEX IF NOT EXISTS idx_market_ticks_fixture_id ON market_ticks (fixture_id);

CREATE TABLE IF NOT EXISTS analytics_results (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id                  TEXT NOT NULL,
    timestamp                   REAL NOT NULL,
    lambda_home                 REAL,
    lambda_away                 REAL,
    home_win_probability        REAL,
    draw_probability             REAL,
    away_win_probability        REAL,
    monte_carlo_home_win        REAL,
    monte_carlo_draw             REAL,
    monte_carlo_away_win        REAL,
    monte_carlo_simulations     INTEGER,
    mi_rate                     REAL,
    z_mi                        REAL,
    pressure_index               REAL,
    pressure_acceleration        REAL,
    xg_proxy                    REAL,
    shot_quality_proxy          REAL,
    market_reaction_elasticity  REAL,
    signal_quality_score        REAL,
    data_quality                 REAL,
    calibration_confidence      REAL,
    metadata_json                TEXT,
    created_at                  REAL NOT NULL,
    FOREIGN KEY (fixture_id) REFERENCES matches (fixture_id)
);

CREATE INDEX IF NOT EXISTS idx_analytics_results_fixture_id
    ON analytics_results (fixture_id);

CREATE TABLE IF NOT EXISTS audit_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id          TEXT NOT NULL,
    timestamp           REAL NOT NULL,
    verdict             TEXT NOT NULL,
    checks_json         TEXT NOT NULL,
    reasons_json         TEXT NOT NULL,
    payload_checksum    TEXT NOT NULL,
    created_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_events_fixture_id ON audit_events (fixture_id);

CREATE TABLE IF NOT EXISTS journal_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type          TEXT NOT NULL,
    fixture_id          TEXT,
    timestamp           REAL NOT NULL,
    details_json         TEXT NOT NULL,
    created_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_journal_events_fixture_id ON journal_events (fixture_id);
CREATE INDEX IF NOT EXISTS idx_journal_events_event_type ON journal_events (event_type);

CREATE TABLE IF NOT EXISTS quarantine_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id          TEXT NOT NULL,
    reason               TEXT NOT NULL,
    timestamp           REAL NOT NULL,
    details_json         TEXT NOT NULL,
    created_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quarantine_events_fixture_id
    ON quarantine_events (fixture_id);

CREATE TABLE IF NOT EXISTS shadow_analysis (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id          TEXT NOT NULL,
    timestamp           REAL NOT NULL,
    model_confidence    REAL,
    uncertainty          REAL,
    volatility           REAL,
    simulated_exposure   REAL,
    simulated_drawdown   REAL,
    risk_score           REAL,
    calibration_error    REAL,
    created_at          REAL NOT NULL,
    FOREIGN KEY (fixture_id) REFERENCES matches (fixture_id)
);

CREATE INDEX IF NOT EXISTS idx_shadow_analysis_fixture_id
    ON shadow_analysis (fixture_id);
