from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Mapping

from .runtime.observer import RunEvent, RunObserver


def run_process_with_live_output(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    input_text: str | None,
    timeout_seconds: int,
    observer: RunObserver,
    trial_id: str | None,
    phase: str | None,
    stage_name: str | None,
    stdout_source: str,
    stderr_source: str,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env),
        bufsize=1,
    )

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    queue: Queue[tuple[str, str]] = Queue()

    def pump(stream, *, source: str, target: list[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                target.append(line)
                queue.put((source, line))
        finally:
            stream.close()

    def write_stdin() -> None:
        if input_text is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(input_text)
        except BrokenPipeError:
            return
        finally:
            proc.stdin.close()

    stdout_thread = threading.Thread(
        target=pump,
        kwargs={"stream": proc.stdout, "source": stdout_source, "target": stdout_parts},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=pump,
        kwargs={"stream": proc.stderr, "source": stderr_source, "target": stderr_parts},
        daemon=True,
    )
    stdin_thread = threading.Thread(target=write_stdin, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    stdin_thread.start()

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 and proc.poll() is None:
            timed_out = True
            proc.kill()
        try:
            source, line = queue.get(timeout=max(0.05, min(0.2, max(remaining, 0.0))))
            observer.emit(
                RunEvent(
                    kind="process_output",
                    message=line.rstrip("\n"),
                    trial_id=trial_id,
                    phase=phase,
                    stage_name=stage_name,
                    source=source,
                )
            )
        except Empty:
            if proc.poll() is not None and not stdout_thread.is_alive() and not stderr_thread.is_alive():
                break

    stdout_thread.join()
    stderr_thread.join()
    stdin_thread.join()
    returncode = proc.wait()
    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout_seconds, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)
