"""Stage 5 process entry point.

Thin composition only: load Settings, construct the application
runtime, start the read-only dashboard, wait for a shutdown signal (or
an internal terminal-failure escalation from the runtime itself), tear
everything down, and return an exit code reflecting whether shutdown
was clean.

No fixture registration or provider wiring happens here: no
fixture-discovery source exists anywhere in this repository (see
runtime/application.py's module docstring and the Stage 5 preflight),
so this entry point does not invent one. With zero fixtures
registered, the application starts and idles safely -- the dashboard
serves an empty/idle status -- rather than fabricating live data.
A caller with a real fixture/provider source uses
`ApplicationRuntime.register_fixture()` / `.start_provider_task()`
directly (e.g. from a small script, or a future Stage 6 integration),
not through this file.
"""

from __future__ import annotations

import asyncio
import signal
import sys

from config.settings import load_settings
from dashboard.server import DashboardServer
from runtime.application import ApplicationRuntime


async def _async_main() -> int:
    settings = load_settings()
    runtime = ApplicationRuntime(settings)
    await runtime.start()

    dashboard = DashboardServer(
        runtime, host=settings.dashboard_host, port=settings.dashboard_port, db_path=settings.db_path
    )
    dashboard.start()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: runtime.request_shutdown(f"signal {s.name}"))
        except (NotImplementedError, AttributeError):
            # add_signal_handler is unsupported on some platforms/loops
            # (notably Windows' default event loop) -- the default
            # Python SIGINT -> KeyboardInterrupt behavior below still
            # provides a clean shutdown path in that case.
            pass

    try:
        await runtime.wait_for_shutdown()
    except KeyboardInterrupt:
        runtime.request_shutdown("KeyboardInterrupt")

    await runtime.shutdown()
    dashboard.stop()

    return 1 if runtime.terminal_error else 0


def main() -> int:
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
