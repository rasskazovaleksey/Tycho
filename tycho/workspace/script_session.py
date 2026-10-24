"""A run_python script that plays (TYCHO_PY_ACTIONS): ``action(...)`` inside the model's code.

Franzen's harness lets the model call ``action(...)`` inside its Python tool, so one LLM call can probe,
observe and react several times. A Tycho turn ends at take_action, so every observation costs another LLM
call (~0.6 actions per call against ~2.1 for the baseline). Here the script's process stays alive across
environment steps: each ``action(...)`` sends a request to the agent and blocks; the agent plays it like a
batch step and answers with the next observation.

Protocol (JSON lines): the script writes requests ``{"action", "row", "col"}`` to a private copy of its
original stdout and reads answers from stdin: ``{"obs": {...}}``, ``{"error": msg}`` (action rejected,
raised as ValueError) or ``{"stop": reason}`` (raised as ActionStopped). The model's prints go to a file
(stdout and stderr are re-pointed at fd level), so nothing it prints can corrupt the protocol. When the
code ends the script sends ``{"done": true}``. A watchdog kills the process after ``lifetime`` seconds.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

# Runs first, before the run_python preamble, so that everything printed (the preamble's own messages
# included) goes to the output file and only protocol lines reach the agent.
ACT_HEAD = r'''
import os as _os, sys as _sys, json as _json, traceback as _tb
_proto = _os.fdopen(_os.dup(1), "w", buffering=1)
_out = open(_os.environ["TYCHO_ACT_OUT"], "w", buffering=1)
_os.dup2(_out.fileno(), 1)
_os.dup2(_out.fileno(), 2)
_sys.stdout = _sys.stderr = open(1, "w", buffering=1, closefd=False)


class ActionStopped(Exception):
    """The harness stopped the script: the level completed, the game ended or the action limit was hit."""


def action(name, row=None, col=None):
    """Play one game action and wait for the result. Returns the observation dict: grid (numpy array),
    state, levels_completed, changed (what changed inside the board), available (actions offered now).
    The preloaded `grid` is updated too."""
    global grid
    _proto.write(_json.dumps({"action": str(name), "row": row, "col": col}) + "\n")
    _proto.flush()
    line = _sys.stdin.readline()
    if not line:
        raise ActionStopped("the harness closed the session")
    msg = _json.loads(line)
    if "error" in msg:
        raise ValueError(msg["error"])
    if "stop" in msg:
        raise ActionStopped(msg["stop"])
    obs = msg["obs"]
    obs["grid"] = np.array(obs["grid"], dtype=np.int16)
    grid = obs["grid"]
    return obs
'''

ACT_TAIL = r'''
try:
    exec(compile(_TYCHO_CODE, "<run_python>", "exec"), globals())
except ActionStopped as _e:
    print(f"[script stopped by the harness: {_e}]")
except SystemExit:
    pass
except BaseException:
    _tb.print_exc()
finally:
    _sys.stdout.flush()
    _proto.write(_json.dumps({"done": True}) + "\n")
    _proto.flush()
'''


class ScriptSession:
    """One live script. ``next_event`` returns its next request ({"action": ...}), {"done": True} once its
    code ended, or {"timeout": True} / {"exited": True}; ``send`` answers a request; ``finish`` stops the
    process if needed and returns everything it printed."""

    def __init__(self, command: list[str], workspace: Path, run_dir: Path, *, script: Path | None = None,
                 lifetime: float = 900.0):
        self.run_dir = run_dir
        self.script = script
        self.out_path = run_dir / "out.txt"
        env = {**os.environ, "TYCHO_ACT_OUT": str(self.out_path), "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(command, cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=env)
        self._buf = b""
        self.actions = 0
        self.done = False
        self._watchdog = threading.Timer(lifetime, self._kill)
        self._watchdog.daemon = True
        self._watchdog.start()

    @classmethod
    def start(cls, sandbox, workspace, preamble: str, code: str, *, lifetime: float = 900.0) -> "ScriptSession":
        workspace = Path(workspace)
        run_dir = Path(tempfile.mkdtemp(prefix=".tycho-act-", dir=workspace))
        # in the workspace root, like run_python's scripts, so `import wmlib` / `import helpers` resolve
        script = workspace / f"{run_dir.name}.py"
        script.write_text(ACT_HEAD + preamble + f"_TYCHO_CODE = {code!r}\n" + ACT_TAIL)
        command = sandbox.command(workspace, script, name=f"tycho-act-{run_dir.name}")
        return cls(command, workspace, run_dir, script=script, lifetime=lifetime)

    def _kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()

    def next_event(self, timeout: float) -> dict:
        """The script's next protocol message, waiting at most ``timeout`` seconds of its compute."""
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                return {"timeout": True}
            chunk = os.read(fd, 65536)
            if not chunk:
                self.done = True
                return {"exited": True}
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return self.next_event(timeout)
        if msg.get("done"):
            self.done = True
        elif "action" in msg:
            self.actions += 1
        return msg

    def send(self, msg: dict) -> None:
        try:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.done = True

    def finish(self, stop: str | None = None, timeout: float = 10.0, max_chars: int = 6000) -> str:
        """End the session: tell a still-acting script why it must stop, give it ``timeout`` seconds to
        print and exit, then kill it. Returns its output (head and tail kept past ``max_chars``)."""
        if not self.done and self.proc.poll() is None:
            if stop is not None:
                self.send({"stop": stop})
            for _ in range(20):  # a script that catches ActionStopped and acts again is stopped again
                ev = self.next_event(timeout)
                if ev.get("done") or ev.get("exited") or ev.get("timeout"):
                    break
                self.send({"stop": stop or "the session ended"})
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self._watchdog.cancel()
        try:
            out = self.out_path.read_text(errors="replace")
        except OSError:
            out = ""
        shutil.rmtree(self.run_dir, ignore_errors=True)
        if self.script is not None:
            self.script.unlink(missing_ok=True)
        if len(out) > max_chars:
            out = out[: max_chars // 2] + "\n...[output clipped]...\n" + out[-max_chars // 2:]
        return out.strip()
