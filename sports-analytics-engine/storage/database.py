"""Async SQLite database access layer, configured for WAL mode.

This module owns the single physical connection lifecycle and schema
initialization.

Write access control
---------------------
Application-level writes (execute / execute_many) are gated behind a
WriterToken, obtainable only via Database.issue_writer_token(). A given
Database instance issues at most one token, ever. storage.db_writer.DBWriter
is the sole intended holder of that token, so any code path that tries
to bypass DBWriter and write directly must first acquire its own token,
which fails once DBWriter already holds the single token for that
Database. This makes the single-writer architecture structurally hard
to bypass, rather than merely a documentation convention.

Read access (fetch_all / fetch_one) requires no token and remains
freely usable by any component, including the dashboard and agents.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import aiosqlite

_SCHEMA_FILENAME = "schema.sql"


class UnauthorizedWriteError(PermissionError):
    """Raised when a write is attempted without a valid WriterToken."""


class WriterToken:
    """Opaque capability granting write access to exactly one Database.

    Instances are only constructed by Database.issue_writer_token() and
    should not be constructed directly by application code. Holding a
    WriterToken is the only way to call Database.execute() /
    Database.execute_many(); storage.db_writer.DBWriter is the intended
    sole holder.
    """

    def __init__(self, database: "Database") -> None:
        self._database = database


class Database:
    """Owns a single aiosqlite connection configured for WAL durability.

    Attributes:
        db_path: filesystem path to the SQLite database file.
        busy_timeout_ms: SQLite busy_timeout in milliseconds.
    """

    def __init__(self, db_path: str, busy_timeout_ms: int = 5000) -> None:
        """Initialize the Database wrapper without opening a connection yet.

        Args:
            db_path: filesystem path to the SQLite database file. Parent
                directories are created lazily on connect().
            busy_timeout_ms: SQLite busy_timeout in milliseconds, applied
                on connect().
        """
        self.db_path = db_path
        self.busy_timeout_ms = busy_timeout_ms
        self._connection: Optional[aiosqlite.Connection] = None
        self._writer_token_issued = False

    @property
    def connection(self) -> aiosqlite.Connection:
        """Return the active connection.

        Raises:
            RuntimeError: if connect() has not been called yet.
        """
        if self._connection is None:
            raise RuntimeError("Database is not connected; call connect() first")
        return self._connection

    async def connect(self) -> None:
        """Open the SQLite connection and apply WAL/durability pragmas."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        await self._connection.execute("PRAGMA journal_mode=WAL;")
        await self._connection.execute("PRAGMA synchronous=NORMAL;")
        await self._connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
        await self._connection.execute("PRAGMA foreign_keys=ON;")
        await self._connection.commit()

    async def init_schema(self, schema_path: Optional[str] = None) -> None:
        """Create all tables/indices defined in schema.sql if not present.

        Args:
            schema_path: optional explicit path to a schema SQL file.
                Defaults to schema.sql located alongside this module.
        """
        if schema_path is None:
            schema_path = str(Path(__file__).parent / _SCHEMA_FILENAME)

        with open(schema_path, "r", encoding="utf-8") as handle:
            schema_sql = handle.read()

        await self.connection.executescript(schema_sql)
        await self.connection.commit()

    def issue_writer_token(self) -> WriterToken:
        """Issue the single WriterToken permitted for this Database instance.

        Returns:
            A new WriterToken bound to this Database.

        Raises:
            RuntimeError: if a WriterToken has already been issued for
                this Database instance. Only one writer is permitted,
                matching the single-writer architecture.
        """
        if self._writer_token_issued:
            raise RuntimeError(
                "A writer token has already been issued for this Database; "
                "only a single writer is permitted"
            )
        self._writer_token_issued = True
        return WriterToken(self)

    def _validate_token(self, token: WriterToken) -> None:
        """Validate that token grants write access to this Database.

        Args:
            token: the WriterToken presented by the caller.

        Raises:
            UnauthorizedWriteError: if token is not a valid WriterToken
                issued by this exact Database instance.
        """
        if not isinstance(token, WriterToken) or token._database is not self:
            raise UnauthorizedWriteError(
                "Invalid or foreign WriterToken; application-level writes "
                "must go through the single DBWriter for this Database"
            )

    async def execute(
        self, token: WriterToken, query: str, params: Sequence[Any] = ()
    ) -> None:
        """Execute a single write statement and commit.

        Args:
            token: WriterToken proving the caller is the authorized writer.
            query: parametrized SQL statement.
            params: positional parameters for the statement.

        Raises:
            UnauthorizedWriteError: if token is invalid for this Database.
        """
        self._validate_token(token)
        await self.connection.execute(query, params)
        await self.connection.commit()

    async def execute_many(
        self, token: WriterToken, query: str, params_list: Iterable[Sequence[Any]]
    ) -> None:
        """Execute a batch of write statements against the same query and commit.

        Args:
            token: WriterToken proving the caller is the authorized writer.
            query: parametrized SQL statement, shared across all rows.
            params_list: iterable of positional parameter tuples/lists.

        Raises:
            UnauthorizedWriteError: if token is invalid for this Database.
        """
        self._validate_token(token)
        await self.connection.executemany(query, list(params_list))
        await self.connection.commit()

    async def fetch_all(
        self, query: str, params: Sequence[Any] = ()
    ) -> List[aiosqlite.Row]:
        """Execute a read query and return all resulting rows.

        Args:
            query: parametrized SQL SELECT statement.
            params: positional parameters for the statement.

        Returns:
            A list of aiosqlite.Row objects.
        """
        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return rows

    async def fetch_one(
        self, query: str, params: Sequence[Any] = ()
    ) -> Optional[aiosqlite.Row]:
        """Execute a read query and return the first resulting row, if any.

        Args:
            query: parametrized SQL SELECT statement.
            params: positional parameters for the statement.

        Returns:
            A single aiosqlite.Row, or None if no rows matched.
        """
        cursor = await self.connection.execute(query, params)
        row = await cursor.fetchone()
        await cursor.close()
        return row

    async def close(self) -> None:
        """Close the underlying connection, if open."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def __aenter__(self) -> "Database":
        await self.connect()
        await self.init_schema()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
