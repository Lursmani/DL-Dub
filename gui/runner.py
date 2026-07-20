"""Run pipeline stages in a worker thread, streaming their console output.

Stage functions print progress to stdout and tqdm draws its bar on stderr.
We capture both with contextlib.redirect_stdout/redirect_stderr — which swap
the GLOBAL sys.stdout/sys.stderr, not thread-locals. That is safe here ONLY
because runs are serialized: every heavy button in the app shares
concurrency_id="heavy", and run_stages additionally holds RUN_LOCK.

The generator yields the accumulated log at least once per second even when
the stages are silent (Demucs can go minutes without output) — Colab's share
tunnel drops long-idle SSE streams, and the keepalive traffic prevents that.
Even if the browser disconnects mid-run, the worker thread finishes and the
manifest is saved; "Refresh from disk" in the UI recovers the state.
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
from pipeline.util import Manifest, workdir_for

# Held for the duration of a stage run; light handlers (e.g. the translation
# table save) check .locked() and refuse manifest writes during a run.
RUN_LOCK = threading.Lock()

MAX_LOG_LINES = 300


class _QueueWriter(io.TextIOBase):
    """File-like sink emitting ("append"|"replace", line) ops onto a queue.

    tqdm redraws its bar with carriage returns; we translate `\\r` into
    "replace" ops so the progress bar becomes one self-updating log line
    instead of hundreds of appended ones.
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


def _resolve_stages(names: list[str]) -> list[tuple[str, Callable]]:
    from dub import STAGES  # import here: dub imports all stage modules

    return [(n, fn) for n, fn in STAGES if n in names]


def run_stages(
    stage_names: list[str],
    video: Path,
    config_path: Path,
    stages: list[tuple[str, Callable]] | None = None,
) -> Iterator[str]:
    """Yield the accumulated log text (last MAX_LOG_LINES lines), ~1/second.

    `stages` overrides the stage list for testing; by default the names are
    resolved against dub.STAGES so ordering always matches the CLI.
    """
    if not RUN_LOCK.acquire(blocking=False):
        yield "Another run is already in progress — wait for it to finish."
        return
    try:
        q: queue.Queue = queue.Queue()
        done = threading.Event()
        to_run = stages if stages is not None else _resolve_stages(stage_names)

        def worker() -> None:
            writer = _QueueWriter(q)
            try:
                with contextlib.redirect_stdout(writer), \
                        contextlib.redirect_stderr(writer):
                    cfg = Config.load(config_path)
                    workdir = workdir_for(video)
                    manifest = Manifest.load_or_init(workdir, video)
                    for name, fn in to_run:
                        print(f"=== {name} ===")
                        fn(video, workdir, manifest, cfg)
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
