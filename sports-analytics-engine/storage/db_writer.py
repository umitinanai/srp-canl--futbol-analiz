"""Single-writer, batched database write pipeline.

Every write in the system (snapshots, market ticks, analytics results,
audit events, journal events, quarantine events, shadow analysis) must
flow through exactly one DBWriter instance's bounded asyncio.Queue, so
that SQLite -- which tolerates only one writer well even in WAL mode --
never sees concurrent write contention from multiple agents.

DBWriter obtains the single WriterToken for its Database on
construction (via Database.issue_writer_token()), which structurally
prevents any other component from writing to that Database: a second
attempt to issue a token raises RuntimeError.

Lifecycle
---------
DBWriter moves through an explicit state machine::

    CREATED --start()--> RUNNING --stop()--> DRAINING --> STOPPED
                            |
                            | unrecoverable flush failure
                            v
                          FAILED

- CREATED: initial state; no consumer loop running yet.
- RUNNING: consumer loop active; enqueue() is accepted.
- DRAINING: stop() has been called; no NEW enqueue() calls are accepted,
  but the consumer loop keeps flushing whatever was already queued.
- STOPPED: draining finished cleanly; terminal state.
- FAILED: a flush failed and exhausted its retry budget; the consumer
  loop has exited and any items still queued at that point are NOT
  flushed. This is a terminal state signalling that operator
  intervention is required. No further enqueue() calls are accepted.

enqueue() and stop() both acquire the same internal asyncio.Lock
(_lifecycle_lock) around their state check/transition and their
corresponding queue.put() call. This makes the accept-and-enqueue
critical section and the stop-and-drain critical section mutually
exclusive: stop() cannot observe RUNNING, flip to DRAINING and insert
the STOP_SENTINEL while an enqueue() call that already observed RUNNING
is still in the process of placing its item into the queue -- stop()
simply blocks on the lock until that enqueue() finishes. This removes
any dependence on asyncio.Queue's internal waiter-ordering semantics:
every enqueue() that is accepted (does not raise
DBWriterNotAcceptingError) is guaranteed to have its item in the queue,
strictly before the STOP_SENTINEL, before stop() can proceed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

from storage.database import Database, WriterToken

logger = logging.getLogger(__name__)

_STOP_SENTINEL = object()


class DBWriterState(str, Enum):
    """Lifecycle state of a DBWriter instance."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class WriteRequest:
    """A single queued write operation.

    Attributes:
        table: logical table name this write targets (used for grouping
            and observability only).
        query: full parametrized INSERT/UPDATE statement using "?"
            placeholders. Requests sharing the exact same query string
            are batched together via executemany for efficiency.
        params: positional parameters matching the query's placeholders.
    """

    table: str
    query: str
    params: Sequence[object]


class DBWriterQueueFullError(RuntimeError):
    """Raised when a non-blocking enqueue is attempted on a full queue."""


class DBWriterNotAcceptingError(RuntimeError):
    """Raised when enqueue is attempted while DBWriter is not RUNNING."""


class DBWriterFailedError(RuntimeError):
    """Raised internally when a flush exhausts its retry budget."""


class DBWriter:
    """Consumes WriteRequest objects from a bounded queue and batch-inserts them.

    Attributes:
        database: the Database instance writes are flushed to.
        batch_size: maximum number of requests flushed in a single batch.
        flush_interval_seconds: maximum time a request may wait in the
            queue before a partial batch is flushed anyway.
        max_flush_retries: number of retry attempts for a failed flush
            before the writer transitions to FAILED. Retries are NOT
            infinite.
        base_retry_delay_seconds: base delay for exponential backoff
            between retry attempts.
    """

    def __init__(
        self,
        database: Database,
        queue_maxsize: int = 2000,
        batch_size: int = 50,
        flush_interval_seconds: float = 1.0,
        max_flush_retries: int = 3,
        base_retry_delay_seconds: float = 0.1,
    ) -> None:
        """Initialize the DBWriter and claim the single WriterToken for database.

        Args:
            database: Database instance writes are flushed to. Must not
                already have had a WriterToken issued to another writer.
            queue_maxsize: maximum number of pending WriteRequest objects
                the internal queue will hold before backpressure kicks in.
            batch_size: number of requests to accumulate before flushing.
            flush_interval_seconds: maximum wait time before flushing a
                partial (non-full) batch.
            max_flush_retries: maximum retry attempts per failing batch
                flush before transitioning to FAILED. Must be >= 0.
            base_retry_delay_seconds: base delay (seconds) for exponential
                backoff between retries. Must be > 0.

        Raises:
            ValueError: if any numeric parameter is out of range.
            RuntimeError: if database already has a WriterToken issued.
        """
        if queue_maxsize < 1:
            raise ValueError("queue_maxsize must be a positive integer")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be strictly positive")
        if max_flush_retries < 0:
            raise ValueError("max_flush_retries must be >= 0")
        if base_retry_delay_seconds <= 0:
            raise ValueError("base_retry_delay_seconds must be strictly positive")

        self.database = database
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.max_flush_retries = max_flush_retries
        self.base_retry_delay_seconds = base_retry_delay_seconds

        self._writer_token: WriterToken = database.issue_writer_token()
        self._queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=queue_maxsize)
        self._consumer_task: "asyncio.Task[None] | None" = None
        self._state: DBWriterState = DBWriterState.CREATED
        self._lifecycle_lock: asyncio.Lock = asyncio.Lock()

        self.total_flushed: int = 0
        self.total_batches: int = 0
        self.total_retries: int = 0
        self.last_error: Optional[str] = None

    @property
    def state(self) -> DBWriterState:
        """Return the current lifecycle state."""
        return self._state

    async def start(self) -> None:
        """Start the background consumer loop, transitioning CREATED -> RUNNING.

        Idempotent: calling start() while already RUNNING is a no-op.

        Raises:
            RuntimeError: if the writer is DRAINING, STOPPED or FAILED
                (a DBWriter cannot be restarted after being stopped).
        """
        if self._state == DBWriterState.RUNNING:
            return
        if self._state != DBWriterState.CREATED:
            raise RuntimeError(
                f"Cannot start a DBWriter in state {self._state.value}; "
                "a stopped/failed DBWriter cannot be restarted"
            )
        self._state = DBWriterState.RUNNING
        self._consumer_task = asyncio.create_task(self._consumer_loop())

    async def stop(self) -> None:
        """Signal the consumer loop to stop, drain pending writes, and await it.

        Transitions RUNNING -> DRAINING -> STOPPED. Idempotent: calling
        stop() again on an already STOPPED or FAILED writer is a no-op.
        No new enqueue() calls are accepted once this method begins.

        The DRAINING transition and the STOP_SENTINEL insertion happen
        under the same lock enqueue() uses, so this call cannot "cut in
        front of" an enqueue() call that is already in the process of
        placing its item into the queue (see module docstring).
        """
        if self._state in (DBWriterState.STOPPED, DBWriterState.FAILED):
            return
        if self._state == DBWriterState.CREATED:
            self._state = DBWriterState.STOPPED
            return

        async with self._lifecycle_lock:
            # Re-check inside the lock: another stop() call may have
            # already completed the transition while we were waiting
            # for the lock.
            if self._state in (DBWriterState.STOPPED, DBWriterState.FAILED):
                return
            self._state = DBWriterState.DRAINING
            await self._queue.put(_STOP_SENTINEL)

        if self._consumer_task is not None:
            await self._consumer_task
            self._consumer_task = None

        if self._state == DBWriterState.DRAINING:
            self._state = DBWriterState.STOPPED

    async def enqueue(self, request: WriteRequest) -> None:
        """Enqueue a write request, applying backpressure if the queue is full.

        Args:
            request: the WriteRequest to persist.

        Raises:
            DBWriterNotAcceptingError: if the writer is not RUNNING at
                the moment this call is made. Guaranteed to be raised
                (rather than silently racing with a concurrent stop())
                because the state check and the queue.put() below run
                inside the same lock stop() uses for its DRAINING
                transition and STOP_SENTINEL insertion -- see module
                docstring.
        """
        async with self._lifecycle_lock:
            if self._state != DBWriterState.RUNNING:
                raise DBWriterNotAcceptingError(
                    f"DBWriter is not accepting writes in state {self._state.value}"
                )
            await self._queue.put(request)

    def enqueue_nowait(self, request: WriteRequest) -> None:
        """Enqueue a write request without blocking.

        Args:
            request: the WriteRequest to persist.

        Raises:
            DBWriterNotAcceptingError: if the writer is not RUNNING.
            DBWriterQueueFullError: if the queue is currently full.
        """
        if self._state != DBWriterState.RUNNING:
            raise DBWriterNotAcceptingError(
                f"DBWriter is not accepting writes in state {self._state.value}"
            )
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull as exc:
            raise DBWriterQueueFullError(
                "DBWriter queue is full; backpressure limit reached"
            ) from exc

    async def _consumer_loop(self) -> None:
        """Background loop: accumulate requests and flush them in batches."""
        batch: List[WriteRequest] = []
        deadline = time.monotonic() + self.flush_interval_seconds

        while True:
            timeout = max(0.0, deadline - time.monotonic())
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if batch:
                    try:
                        await self._flush(batch)
                    except DBWriterFailedError:
                        return
                    batch = []
                deadline = time.monotonic() + self.flush_interval_seconds
                continue

            if item is _STOP_SENTINEL:
                if batch:
                    try:
                        await self._flush(batch)
                    except DBWriterFailedError:
                        return
                return

            batch.append(item)  # type: ignore[arg-type]

            if len(batch) >= self.batch_size:
                try:
                    await self._flush(batch)
                except DBWriterFailedError:
                    return
                batch = []
                deadline = time.monotonic() + self.flush_interval_seconds

    async def _flush(self, batch: List[WriteRequest]) -> None:
        """Group a batch by query and persist each group, with retry on failure.

        Args:
            batch: list of WriteRequest objects to persist. Grouped by
                identical query string so each group can be executed as
                a single executemany() call.

        Raises:
            DBWriterFailedError: if any group exhausts its retry budget.
                The writer has already been transitioned to FAILED by
                the time this is raised.
        """
        if not batch:
            return

        grouped: Dict[str, List[Sequence[object]]] = {}
        for request in batch:
            grouped.setdefault(request.query, []).append(request.params)

        for query, params_list in grouped.items():
            await self._execute_with_retry(query, params_list)

        self.total_flushed += len(batch)
        self.total_batches += 1

    async def _execute_with_retry(
        self, query: str, params_list: List[Sequence[object]]
    ) -> None:
        """Execute a batched write, retrying with exponential backoff on failure.

        Args:
            query: parametrized SQL statement shared across params_list.
            params_list: positional parameter tuples for executemany().

        Raises:
            DBWriterFailedError: once max_flush_retries is exhausted.
                The writer's state is set to FAILED before this is raised.
        """
        attempt = 0
        while True:
            try:
                await self.database.execute_many(self._writer_token, query, params_list)
                return
            except Exception as exc:  # noqa: BLE001 - broad by design, see retry policy
                self.last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_flush_retries:
                    logger.error(
                        "DBWriter exhausted %d retries flushing %d rows; "
                        "transitioning to FAILED. Last error: %s",
                        self.max_flush_retries,
                        len(params_list),
                        self.last_error,
                    )
                    self._state = DBWriterState.FAILED
                    raise DBWriterFailedError(self.last_error) from exc

                delay = self.base_retry_delay_seconds * (2**attempt)
                logger.warning(
                    "DBWriter flush attempt %d/%d failed (%s); retrying in %.3fs",
                    attempt + 1,
                    self.max_flush_retries,
                    self.last_error,
                    delay,
                )
                self.total_retries += 1
                attempt += 1
                await asyncio.sleep(delay)

    @property
    def pending(self) -> int:
        """Return the current number of items waiting in the queue."""
        return self._queue.qsize()
