# Vanguard Live Quant Engine v5.0

High-frequency quantitative sports analytics engine. Analytical /
simulation system only -- **no real-money order execution, no
brokerage integration, no capital allocation ledger** exists anywhere
in this codebase.

## Production status

Built in 5 stages, plus hardening sub-stages. Current status:

- [x] **Stage 1 — Models, Config & Storage**
- [x] **Stage 1.1 — Storage/Config Hardening Patch**
- [x] **Stage 2 — Analytics Engine** (Poisson, Monte Carlo, Momentum, Pressure, xG proxy, market reaction, Bayesian dynamic lambda, regime detection, Stage 2B live pipeline)
- [x] **Stage 3 — Safety & Guards** (Data Age classification, Kill Switch, Quarantine, System Lifecycle, Orchestrator control plane)
- [x] **Stage 4 — Six Agents** (Data, Quant, Market, Risk, Auditor, Journal)
- [x] **Stage 5 — Dashboard, Application Orchestrator, Main & Final Integration** (this delivery)

## Stage 1 / 1.1 contents

```text
models/      -- immutable data contracts (Match, Snapshot, Market, Metrics, Events)
config/      -- Settings (env/.env driven, immutable, strictly validated) and League registry
storage/     -- SQLite WAL Database wrapper (token-gated writes) + single-queue batched DBWriter
tests/       -- unit tests for storage, models and config
```

## Stage 4 contents

```text
agents/      -- the six agents (Data, Quant, Market, Risk, Auditor, Journal),
                each an independently usable, constructor-injected building
                block. No application wiring (main.py, dashboard, fixture
                discovery, process supervisor) lives here -- that is Stage 5.
```

- **DataAgent** is a thin adapter over the existing provider ->
  normalizer -> Stage 3 `safety.Orchestrator` path: it builds and
  persists a checksummed `Snapshot` per raw event (via the existing,
  previously-uncalled `data.normalizer.build_snapshot()`), then
  dispatches the event through `Orchestrator.process()` unchanged. It
  does not reimplement `process_event()`, does not own Bayesian priors,
  and does not perform fixture discovery or phase-based polling
  scheduling (out of Stage 4 scope; see agent docstrings).
- **QuantAgent** persists an already-computed `QuantResult` (from the
  `RecalculationOutcome` the pipeline already produced) and journals a
  fact for lightweight-only recalculations, without ever recomputing
  analytics itself.
- **MarketAgent** accepts existing `PriceTick`/market contracts and
  reuses `analytics.market_reaction` for fair probabilities and
  rolling MRE; it never mutates a `QuantResult` and does not implement
  a real odds-provider integration.
- **RiskAgent** produces `RiskMetrics` using only repository-backed
  calculations (existing Monte Carlo confidence intervals,
  `backtesting.metrics.calibration_error`); fields with no
  authoritative formula in the repository (`volatility`,
  `simulated_exposure`, `simulated_drawdown`, `risk_score`) are always
  left unavailable rather than fabricated.
- **AuditorAgent** independently re-invokes the existing
  `data.pipeline.run_full_recalculation()` and a small set of
  self-consistency checks to produce a deterministic `AuditResult`/
  `AuditVerdict`; an `INVALID` verdict may quarantine the fixture via
  the existing `safety.quarantine` API, but no kill-switch policy is
  invented.
- **JournalAgent** is the sole Stage 4 translator from domain
  contracts/events into `storage.db_writer.WriteRequest`s, enqueued on
  one shared `DBWriter`. No other agent imports `storage.db_writer` or
  `storage.database`, and no agent requests its own `WriterToken`.

## Stage 5 contents

```text
runtime/     -- ApplicationRuntime: composition-only application orchestrator
                (dependency wiring, fixture registration, per-fixture runtime
                state, six-agent routing, graceful shutdown). Named `runtime`,
                not `orchestrator`, to stay clearly distinct from
                safety.orchestrator.Orchestrator, which it constructs and
                calls but does not replace.
dashboard/   -- read-only, stdlib-only HTTP status/history server.
main.py      -- thin process entry point (load Settings, construct
                ApplicationRuntime + dashboard, signal handling, shutdown).
```

- **Global vs. fixture-local safety**: `safety.orchestrator.Orchestrator`
  owns its own `SystemLifecycle`/`Quarantine`/(unused) `KillSwitch`
  internally, so `ApplicationRuntime` constructs **one Orchestrator per
  registered fixture** -- an ordinary provider/malformed/state/analytics
  failure on one fixture only ever touches that fixture's own Orchestrator,
  never any other. Exactly **one** separate, truly-global
  `safety.kill_switch.KillSwitch` is owned by `ApplicationRuntime` itself
  and checked before any fixture's event is dispatched, preserving
  Stage 3's existing "kill switch is a global emergency stop" semantics
  without duplicating or weakening it.
- **Fixture registration** (`ApplicationRuntime.register_fixture()`) is
  the only way to create the `matches` parent row a `snapshots`/
  `market_ticks`/`analytics_results`/`shadow_analysis` write requires
  (their existing `FOREIGN KEY` constraints are never disabled). Metadata
  is always caller-supplied -- no fixture-discovery source exists in this
  repository, and none is fabricated. The parent `matches` write is
  awaited to successful enqueue on the shared `DBWriter` before the
  fixture becomes visible to anything else (`ingest_event`,
  `submit_price_tick`, `start_provider_task`, `fixture_ids`) -- a failed
  or in-flight registration can never be raced by a child write.
  Repeated registration with the *same* `league_id`/`home_team`/
  `away_team` is a safe, idempotent upsert of mutable fields
  (phase/minute/goals); repeated registration that would *change* that
  identity is rejected with `FixtureIdentityMismatchError` rather than
  silently updating the database under an already-running fixture. An
  unregistered fixture's ingest is rejected in-process before any
  database write is attempted.
- **Bayesian priors** are owned per-fixture by `ApplicationRuntime`
  (`FixtureContext.home_prior`/`away_prior`), created once at
  registration and held unchanged for that fixture's lifetime, matching
  how `data.pipeline.compute_dynamic_lambda()` actually consumes them
  (a fresh update derived from the same prior + current cumulative
  evidence on every call, never an incrementally-mutated one). A
  documented, explicitly-non-authoritative default
  (`GammaPoissonPrior(shape=1.0, rate=1.0)`) is used only when a caller
  does not supply one.
- **`JournalAgent` remains the sole persistence translator**:
  `ApplicationRuntime` never calls `Database.execute`/`execute_many`
  directly and constructs exactly one shared `DBWriter`. A DBWriter
  reaching its terminal `FAILED` state triggers global shutdown
  escalation (`ApplicationRuntime.request_shutdown(..., is_failure=True)`)
  rather than continuing to run with dead persistence.
- **Dashboard** (`dashboard/server.py`) is stdlib-only (`http.server`,
  `sqlite3`, `threading` -- no new dependency): live status is read via
  `asyncio.run_coroutine_threadsafe()` into the runtime's own event loop,
  and historical tables are read through a completely separate,
  independent, read-only (`file:...?mode=ro`) SQLite connection that
  cannot write even in principle. It exposes `GET`-only endpoints
  (`/status`, `/history/<allowlisted-table>`) and has no kill-switch or
  mutation control of any kind.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest -q
```

## Design notes

- SQLite runs in `WAL` mode with `synchronous=NORMAL` and a configurable
  `busy_timeout`, targeting a 2 GB RAM / 2 vCPU VPS.
- **Single-writer enforcement is structural, not conventional.** `Database`
  issues exactly one `WriterToken` per instance via `issue_writer_token()`;
  a second call raises `RuntimeError`. `Database.execute()` /
  `execute_many()` require a valid token or raise `UnauthorizedWriteError`.
  `DBWriter` claims the one token on construction, so no other component
  can perform an application-level write even if it imports `Database`
  directly. Read methods (`fetch_all` / `fetch_one`) remain token-free.
- **`DBWriter` lifecycle is an explicit state machine:**
  `CREATED -> RUNNING -> DRAINING -> STOPPED`, with a terminal `FAILED`
  state reachable from `RUNNING` if a batch flush exhausts its retry
  budget. `enqueue()` is rejected once the writer leaves `RUNNING`, and
  `stop()` is idempotent and safe to call on a `FAILED` writer. Flush
  failures retry with bounded exponential backoff
  (`max_flush_retries`, `base_retry_delay_seconds`) before transitioning
  to `FAILED` -- there is no infinite retry loop and no silently-dead
  consumer task.
- **Snapshot deduplication uses two distinct SHA-256 checksums:**
  `payload_checksum` (content only, reusable across events, powers the
  Section 22 audit cache) and `identity_checksum`
  (`fixture_id + event_timestamp + payload_checksum`, unique per
  snapshot occurrence). `snapshots.identity_checksum` carries a real
  `UNIQUE` constraint at the database level -- duplicate snapshot
  delivery is rejected by SQLite itself, not just by an application-side
  check.
- All `MatchState` history fields are bounded `deque`s to guarantee
  constant memory usage regardless of match duration.
- `Settings` is a frozen dataclass; `pressure_weights` and `sqs_weights`
  are wrapped in `MappingProxyType` in `__post_init__`, so mutation
  attempts raise `TypeError` even though the underlying dict is never
  exposed. Validation additionally enforces exact expected key sets for
  both weight maps, weight sums ~= 1.0, finite/positive intervals and
  timeouts, and a valid `dashboard_port` range.
- No secret or API key is hard-coded; `config.settings.load_settings()`
  reads exclusively from the environment / `.env`.

