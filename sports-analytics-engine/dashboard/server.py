"""Stage 5 read-only dashboard.

Serves runtime/system visibility ONLY -- no mutation endpoints, no
execution/control endpoints, no kill-switch activation button. Uses
only the Python standard library (http.server, json, sqlite3,
threading): repository evidence (requirements.txt) shows zero
web-framework dependency, and none is added here.

Structurally incapable of writing to the database: historical-table
reads use a SEPARATE, independent `sqlite3` connection opened directly
against `Settings.db_path` in read-only URI mode
(`file:<path>?mode=ro`) -- never the shared `storage.database.Database`
/ `WriterToken` the application runtime uses. This requires no
WriterToken at all, and a write attempt through this connection would
be rejected by SQLite itself, not merely by convention.

Live runtime status (lifecycle/kill-switch/quarantine/DBWriter state)
is read via `asyncio.run_coroutine_threadsafe()` into the application
runtime's own event loop, so the snapshot is always built on that loop
-- never by reading runtime internals directly from this module's own
background HTTP thread.

A dashboard failure (a bad request, a locked/missing database file, a
slow/blocked runtime) is always converted to an HTTP error response
inside the request handler -- it can never propagate into, or affect,
the application runtime's own processing.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from runtime.application import ApplicationRuntime

#: Explicit allowlist of table -> a FIXED SELECT query. Table names
#: from a request are matched against this dict's keys only -- never
#: interpolated into SQL -- eliminating any injection surface even
#: though this is read-only internal tooling.
_HISTORY_QUERIES: Dict[str, str] = {
    "journal_events": "SELECT * FROM journal_events ORDER BY id DESC LIMIT 50",
    "audit_events": "SELECT * FROM audit_events ORDER BY id DESC LIMIT 50",
    "analytics_results": "SELECT * FROM analytics_results ORDER BY id DESC LIMIT 50",
    "shadow_analysis": "SELECT * FROM shadow_analysis ORDER BY id DESC LIMIT 50",
    "quarantine_events": "SELECT * FROM quarantine_events ORDER BY id DESC LIMIT 50",
    "market_ticks": "SELECT * FROM market_ticks ORDER BY id DESC LIMIT 50",
}


def _fetch_history(db_path: str, table: str) -> Optional[List[Dict[str, Any]]]:
    """Read recent rows from an allowlisted table via an independent read-only connection."""
    query = _HISTORY_QUERIES.get(table)
    if query is None:
        return None
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


class _DashboardRequestHandler(BaseHTTPRequestHandler):
    server: "_DashboardHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep console/test output quiet; failures still surface via HTTP status codes

    def _write_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib-mandated method name
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("/", "/status"):
                self._write_json(200, self.server.fetch_status())
                return
            if parsed.path.startswith("/history/"):
                table = parsed.path[len("/history/"):]
                rows = self.server.fetch_history(table)
                if rows is None:
                    self._write_json(404, {"error": f"unknown or disallowed table {table!r}"})
                    return
                self._write_json(200, rows)
                return
            self._write_json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001 - a dashboard failure must never propagate further
            self._write_json(503, {"error": "dashboard temporarily unavailable", "detail": str(exc)})


class _DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: Any,
        handler_class: type,
        runtime: ApplicationRuntime,
        loop: asyncio.AbstractEventLoop,
        db_path: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self._runtime = runtime
        self._loop = loop
        self._db_path = db_path

    def fetch_status(self) -> Dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(self._runtime.async_status_snapshot(), self._loop)
        status = future.result(timeout=5.0)
        return asdict(status)

    def fetch_history(self, table: str) -> Optional[List[Dict[str, Any]]]:
        return _fetch_history(self._db_path, table)


class DashboardServer:
    """Owns the background read-only HTTP thread. start()/stop() are idempotent."""

    def __init__(self, runtime: ApplicationRuntime, host: str, port: int, db_path: str) -> None:
        self._runtime = runtime
        self._host = host
        self._port = port
        self._db_path = db_path
        self._server: Optional[_DashboardHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Bind and start the server. Must be called from within a running event loop."""
        if self._server is not None:
            return
        loop = asyncio.get_running_loop()
        self._server = _DashboardHTTPServer(
            (self._host, self._port), _DashboardRequestHandler, self._runtime, loop, self._db_path
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="dashboard-server")
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    @property
    def address(self) -> Optional[Any]:
        return None if self._server is None else self._server.server_address
