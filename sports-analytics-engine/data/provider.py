"""Provider-independent asynchronous live data acquisition.

The analytics engine must never depend on a specific external data
provider or its field naming. This module defines a small abstract
contract (LiveDataProvider) plus two concrete implementations:

- WebSocketLiveDataProvider: a real, replaceable adapter for providers
  that expose a JSON-over-WebSocket live feed, using the mature
  `websockets` library rather than hand-rolled networking.
- StaticReplayProvider: an in-memory adapter that replays a bounded,
  pre-supplied sequence of raw provider-shaped events. Used for tests
  and for feeding synthetic/backtest-adjacent data through the exact
  same async interface real providers use.

Raw events yielded by any provider are plain dicts in the provider's
own field naming -- normalization into canonical fields happens
strictly at the data.normalizer boundary, never inside a provider.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Dict, Optional, Sequence


class ProviderError(RuntimeError):
    """Raised when a provider adapter encounters an unrecoverable error."""


class ProviderConnectionState(str, Enum):
    """Observable connection lifecycle state of a live data provider.

    Exposed so upstream components (a future orchestrator, health
    checks, or Stage 3's data-freshness/kill-switch layer) can react to
    provider health without Stage 2B implementing that reaction itself.
    """

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


class LiveDataProvider(ABC):
    """Abstract contract for an asynchronous live match data source.

    Implementations must be replaceable: nothing outside this module
    (or a concrete provider's own construction) may depend on which
    concrete provider is in use.
    """

    def __init__(self) -> None:
        self._connection_state: ProviderConnectionState = ProviderConnectionState.DISCONNECTED

    @property
    def connection_state(self) -> ProviderConnectionState:
        """Return this provider's current connection lifecycle state."""
        return self._connection_state

    @abstractmethod
    def stream_events(self, match_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Yield raw, provider-shaped event dicts for a given match.

        Args:
            match_id: the provider's own identifier for the match to
                stream. This module does not interpret or validate the
                identifier's format -- that is the provider's concern.

        Returns:
            An async iterator of raw event dicts, in the provider's own
            field naming. Callers must pass these through
            data.normalizer before using them anywhere else in the
            system.
        """
        raise NotImplementedError


class StaticReplayProvider(LiveDataProvider):
    """Replays a bounded, in-memory list of raw events for one match.

    Useful for tests and for driving the exact same async pipeline used
    by real providers with deterministic, pre-recorded data.
    """

    def __init__(self, events: Sequence[Dict[str, Any]]) -> None:
        """Initialize the replay provider with a fixed event sequence.

        Args:
            events: the raw event dicts to yield, in order. Stored as a
                tuple (bounded, immutable) rather than retained as a
                mutable list.
        """
        super().__init__()
        self._events = tuple(events)

    async def stream_events(self, match_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Yield each pre-supplied event in order.

        Args:
            match_id: unused by this provider (present to satisfy the
                LiveDataProvider contract); all events were supplied at
                construction time.

        Yields:
            Each raw event dict from the sequence supplied at
            construction, in order.
        """
        self._connection_state = ProviderConnectionState.CONNECTED
        try:
            for event in self._events:
                yield event
        finally:
            self._connection_state = ProviderConnectionState.DISCONNECTED


class WebSocketLiveDataProvider(LiveDataProvider):
    """A LiveDataProvider backed by a JSON-over-WebSocket live feed.

    Uses the `websockets` library's async client. Each received text
    frame is parsed as JSON and yielded as-is (still in the provider's
    own field naming); malformed frames are skipped rather than
    crashing the stream, since a single malformed live message must
    never silently enter the analytics engine as garbage, nor should it
    take down an otherwise-healthy live stream.

    Reconnect strategy: on a connection failure (handshake failure,
    dropped connection, or the server closing the stream), the provider
    waits with linear backoff (reconnect_backoff_seconds * attempt
    number) and retries, up to max_reconnect_attempts consecutive
    failures. A successful reconnect resets the attempt counter. After
    exhausting max_reconnect_attempts, connection_state becomes FAILED
    and a ProviderError is raised to the caller -- Stage 2B does not
    implement a kill-switch/quarantine reaction to that state, only
    reports it.

    Backpressure: this provider never buffers messages ahead of the
    consumer. stream_events() is a pull-based async generator -- the
    next WebSocket frame is only read once the caller asks for the next
    item via `async for`/`__anext__`, so memory usage never grows with
    message rate; there is no separate unbounded internal buffer to
    overflow.

    Cancellation: asyncio.CancelledError is never caught for retry
    purposes -- if the consuming task is cancelled, cancellation
    propagates immediately (after best-effort connection cleanup via
    the `async with` context manager), rather than being treated as a
    reconnectable failure.
    """

    def __init__(
        self,
        url_template: str,
        open_timeout_seconds: float = 10.0,
        max_reconnect_attempts: int = 5,
        reconnect_backoff_seconds: float = 1.0,
    ) -> None:
        """Initialize the WebSocket provider.

        Args:
            url_template: a URL template containing a "{match_id}"
                placeholder, e.g. "wss://provider.example/live/{match_id}".
            open_timeout_seconds: how long to wait for the WebSocket
                handshake to complete before giving up. Must be
                strictly positive.
            max_reconnect_attempts: maximum consecutive reconnect
                attempts after a connection failure before giving up
                and raising ProviderError. Must be a non-negative int
                (0 disables reconnection entirely).
            reconnect_backoff_seconds: base delay between reconnect
                attempts; attempt N waits
                reconnect_backoff_seconds * N. Must be strictly positive.

        Raises:
            ProviderError: if url_template does not contain the
                required "{match_id}" placeholder, or if
                open_timeout_seconds/reconnect_backoff_seconds is not
                strictly positive, or max_reconnect_attempts is negative.
        """
        super().__init__()
        if "{match_id}" not in url_template:
            raise ProviderError('url_template must contain a "{match_id}" placeholder')
        if open_timeout_seconds <= 0:
            raise ProviderError("open_timeout_seconds must be strictly positive")
        if isinstance(max_reconnect_attempts, bool) or not isinstance(max_reconnect_attempts, int) or max_reconnect_attempts < 0:
            raise ProviderError("max_reconnect_attempts must be a non-negative int")
        if reconnect_backoff_seconds <= 0:
            raise ProviderError("reconnect_backoff_seconds must be strictly positive")

        self._url_template = url_template
        self._open_timeout_seconds = open_timeout_seconds
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_backoff_seconds = reconnect_backoff_seconds

    async def stream_events(self, match_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Connect (with reconnection) and yield parsed JSON events for match_id.

        Args:
            match_id: the provider's identifier for the match, substituted
                into the configured url_template.

        Yields:
            Each successfully JSON-decoded frame, as a dict. Frames that
            are not valid JSON, or whose decoded value is not a dict,
            are skipped.

        Raises:
            ProviderError: once max_reconnect_attempts consecutive
                connection failures have occurred.
        """
        attempt = 0
        while True:
            self._connection_state = (
                ProviderConnectionState.RECONNECTING if attempt > 0
                else ProviderConnectionState.CONNECTING
            )
            try:
                async for event in self._connect_and_stream(match_id):
                    self._connection_state = ProviderConnectionState.CONNECTED
                    attempt = 0
                    yield event
                # Generator ended without error: the server closed the
                # stream cleanly. Treated the same as a failure for
                # reconnection purposes -- a live match feed ending
                # early is not an expected terminal state at this layer.
                raise ProviderError(f"WebSocket stream for {match_id!r} ended unexpectedly")
            except asyncio.CancelledError:
                self._connection_state = ProviderConnectionState.DISCONNECTED
                raise
            except ProviderError:
                attempt += 1
                if attempt > self._max_reconnect_attempts:
                    self._connection_state = ProviderConnectionState.FAILED
                    raise
                await asyncio.sleep(self._reconnect_backoff_seconds * attempt)
                continue

    async def _connect_and_stream(self, match_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Open one WebSocket connection and yield parsed events until it closes.

        Args:
            match_id: the provider's identifier for the match.

        Yields:
            Each successfully JSON-decoded frame, as a dict.

        Raises:
            ProviderError: if the connection cannot be established or
                is dropped.
        """
        import websockets  # imported lazily so environments without a
        # live provider configured never need the dependency importable
        # at module load time.

        url = self._url_template.format(match_id=match_id)
        try:
            connection_ctx = websockets.connect(url, open_timeout=self._open_timeout_seconds)
            async with connection_ctx as websocket:
                async for raw_message in websocket:
                    event = _safe_parse_json_event(raw_message)
                    if event is not None:
                        yield event
        except asyncio.CancelledError:
            raise
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalized to one provider error type
            raise ProviderError(f"WebSocket connection to {url} failed or was dropped: {exc}") from exc


def _safe_parse_json_event(raw_message: Any) -> Optional[Dict[str, Any]]:
    """Parse a raw WebSocket frame into a dict, or None if it is unusable.

    Args:
        raw_message: the raw frame payload (str or bytes) received from
            the WebSocket connection.

    Returns:
        The parsed dict, or None if raw_message is not valid JSON, or
        decodes to something other than a JSON object.
    """
    try:
        parsed = json.loads(raw_message)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
