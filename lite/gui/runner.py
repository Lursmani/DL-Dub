"""Run the dub in a worker thread, streaming its console output.

autodub() prints progress to stdout (the poll status redraws itself with \\r).
We capture it with contextlib.redirect_stdout/redirect_stderr — which swap
the GLOBAL sys.stdout/sys.stderr, not thread-locals. That is safe here ONLY
because runs are serialized: the run button uses concurrency_limit=1 and
run_stages additionally holds RUN_LOCK.

The generator yields the accumulated log at least once per second even when
the job is silent (the API can poll for many minutes) — share tunnels drop
long-idle SSE streams, and the keepalive traffic prevents that. Even if the
browser disconnects mid-run, the worker thread finishes and the result lands
in work/; re-running the same episode returns the cached output.
"""
from __future__ import annotations

import contextlib
import io
import queue
import threading
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path

from pipeline.config import Config
from pipeline.util import workdir_for

# Held for the duration of a run.
RUN_LOCK = threading.Lock()

MAX_LOG_LINES = 300


class _QueueWriter(io.TextIOBase):
    """File-like sink emitting ("append"|"replace", line) ops onto a queue.

    The poll status redraws its line with carriage returns; we translate `\\r`
    into "replace" ops so it becomes one self-updating log line instead of
    hundreds of appended ones.
    """

    def __init__(self, q: queue.Queue):
        self.q = q
        self.cur = ""        # content of the current, not-yet-terminated line
        self.emitted = False  # has self.cur already been shown as a partial?

    def _flush_partial(self) -> None:
        if self.cur:
            self.q.put(("replace" if self.emitted else "append", self.cur))
            self.emitted = True

    def _close_line(self) -> None:
        if self.emitted:
            if self.cur:  # final content differs from last shown partial
                self.q.put(("replace", self.cur))
        else:
            self.q.put(("append", self.cur))  # includes blank lines
        self.cur = ""
        self.emitted = False

    def write(self, s: str) -> int:
        i = 0
        while i < len(s):
            nl = s.find("\n", i)
            cr = s.find("\r", i)
            if nl == -1 and cr == -1:
                self.cur += s[i:]
                break
            if cr != -1 and (nl == -1 or cr < nl):
                # \r: cursor to column 0 — following text overwrites this line.
                self.cur += s[i:cr]
                self._flush_partial()
                self.cur = ""
                i = cr + 1
            else:
                self.cur += s[i:nl]
                self._close_line()
                i = nl + 1
        self._flush_partial()
        return len(s)

    def flush(self) -> None:  # pragma: no cover - required by file protocol
        pass


def run_stages(
    stages: list[tuple[str, Callable]],
    video: Path,
    config_path: Path,
) -> Iterator[str]:
    """Yield the accumulated log text (last MAX_LOG_LINES lines), ~1/second."""
    if not RUN_LOCK.acquire(blocking=False):
        yield "Another run is already in progress — wait for it to finish."
        return
    try:
        q: queue.Queue = queue.Queue()
        done = threading.Event()

        def worker() -> None:
            writer = _QueueWriter(q)
            try:
                with contextlib.redirect_stdout(writer), \
                        contextlib.redirect_stderr(writer):
                    cfg = Config.load(config_path)
                    workdir = workdir_for(video)
                    for name, fn in stages:
                        print(f"=== {name} ===")
                        fn(video, workdir, cfg)
                    print("=== done ===")
            except BaseException as e:  # noqa: BLE001 - incl. stray SystemExit
                q.put(("append", f"[ERROR] {e}"))
                traceback.print_exc(file=writer)
            finally:
                done.set()

        threading.Thread(target=worker, daemon=True).start()

        lines: list[str] = []
        while not (done.is_set() and q.empty()):
            try:
                op, line = q.get(timeout=1.0)
                if op == "replace" and lines:
                    lines[-1] = line
                else:
                    lines.append(line)
            except queue.Empty:
                pass  # keepalive: fall through and yield unchanged log
            yield "\n".join(lines[-MAX_LOG_LINES:])
        yield "\n".join(lines[-MAX_LOG_LINES:])
    finally:
        RUN_LOCK.release()
