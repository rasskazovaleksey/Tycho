"""Tycho base tools — the file-system + compute primitives an LLM agent is trained on,
scoped to a GameWorkspace, plus the world-action and zoom tools. No experiment/explore/
plan scaffolding — the agent decides its own flow.

MODEL-SPECIFIC TOOL NAMES (#13): models are post-trained on particular tool names.
Opus knows Claude-Code names (Read/Write/Edit/Bash); Gemma knows Gemini-CLI names
(read_file/write_file/run_shell_command). We expose the native names per model but map
them to ONE internal executor — names are just labels, behavior is identical. Pick the
naming with tool_specs(flavor=...).

`take_action` is handled by the agent (parses the move + ends the turn); the rest run
in ToolExecutor. Grids everywhere are SPACE-SEPARATED single chars (one token/cell, #16).
"""
from __future__ import annotations

# the WM-feedback probe is a PYTHON SCRIPT (run in the workspace), not an LLM prompt — it lives in
# tycho/workspace/templates/ alongside the other seeded code (see wm_templates._load_template).

import ast
import re
from pathlib import Path

from tycho.workspace.sandbox import PythonSandbox


def _clip_ends(s: str, cap: int) -> str:
    """Flood guard that NEVER drops the head. A full 64x64 grid print is ~13 KB and the agent
    reads it top-down, so a tail clip ([-N:]) cut row 0. Below `cap`, return verbatim; above it,
    keep head + tail (3:1) with an explicit middle marker, so nothing is silently lost and a
    runaway loop still can't flood context. cap counts characters."""
    s = s or ""
    if len(s) <= cap:
        return s
    head = (cap * 3) // 4
    tail = cap - head
    return s[:head] + f"\n… [+{len(s) - cap} chars elided] …\n" + s[-tail:]

# Run after a world_model.py edit (in the workspace dir). Two-part, kept TIGHT + actionable:
#   1. verify vs every observed transition (changing-transition accuracy is the headline —
#      exact-frame accuracy is vacuous on static games where most frames are no-ops);
#   2. IF the model verifies, try to plan to the goal so the agent learns immediately whether
#      the goal is reachable from here. If not reachable (within a node budget) it may mean
#      the model/goal is wrong OR a state-changing move (undo/reset/a different branch) is
#      needed first — we say so rather than leave the agent guessing.
# VERIFY-ONLY on the prediction side (no per-action next-grid dump): meaningless for click games
# (ACTION6 = 4096 targets). For a hypothetical next grid the agent calls world_model.transition +
# render in run_python directly. Self-contained; prints the import/run error if the model doesn't load.
_WM_FEEDBACK_PROBE = (Path(__file__).resolve().parent / "templates" / "wm_feedback_probe.py.tmpl").read_text()

# Conventional ARC-AGI-3 action semantics (docs.arcprize.org/actions). A HINT — some
# games rebind actions, so the agent must verify by observing effects.
ACTION_HINT = (
    "RESET is always available to restart the current level/attempt as an API recovery control, "
    "not a learned game mechanic. Game-action convention (VERIFY per game — some games differ): "
    "ACTION1-ACTION4 often behave like "
    "directions (ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right), ACTION5 often behaves "
    "like interact/select, ACTION6=click a cell at (row,col) in 0-63, and ACTION7 is commonly "
    "intended as undo but may still be game-specific. "
    "The current turn header lists the valid take_action choices: RESET plus the frame-declared "
    "game actions."
)

# Internal canonical tool names -> per-flavor display names. The executor keys off the
# canonical name; the model sees the flavored name.
_FLAVOR_NAMES = {
    # Model-native names for the file ops (models are post-trained on these). run_python
    # is named run_python EVERYWHERE — it runs Python, not a shell; the trained shell names
    # (Bash/run_shell_command) would wrongly suggest arbitrary shell + setup.
    "claude": {  # Opus / Claude Code
        "read_file": "Read", "write_file": "Write", "edit_file": "Edit", "ls": "Glob",
        "run_python": "run_python", "set_verbosity": "set_verbosity",
        "edit_function": "edit_function",
        "invoke_builder": "invoke_builder", "take_action": "take_action"},
    "gemini": {  # Gemma 4 / Gemini CLI
        "read_file": "read_file", "write_file": "write_file", "edit_file": "replace",
        "ls": "list_directory",
        "run_python": "run_python", "set_verbosity": "set_verbosity",
        "edit_function": "edit_function",
        "invoke_builder": "invoke_builder", "take_action": "take_action"},
    "generic": {  # default / JSON-fallback
        "read_file": "read_file", "write_file": "write_file", "edit_file": "edit_file", "ls": "ls",
        "run_python": "run_python", "set_verbosity": "set_verbosity",
        "edit_function": "edit_function",
        "invoke_builder": "invoke_builder", "take_action": "take_action"},
}

_CANON = {  # reverse map: any display name -> canonical, for the executor
    name: canon for flavor in _FLAVOR_NAMES.values() for canon, name in flavor.items()}


def canonical(name: str) -> str:
    return _CANON.get(name, name)


_DESCRIPTIONS = {
    "ls": "List a directory in the workspace (default '.').",
    "read_file": "Read a file in the workspace.",
    "write_file": "Write (or overwrite) a file in the workspace.",
    "edit_file": "Edit a file by replacing an exact string. Args: path, old (exact text to "
                 "find — must be unique in the file), new (replacement). For small whole-file "
                 "changes, write_file is fine; use this for targeted edits.",
    "edit_function": "Replace ONE top-level Python function by NAME (the function-granular "
                     "editor). Args: path (a .py file, usually world_model.py), name (the "
                     "function to replace, e.g. \"transition\"), code (the FULL new definition: "
                     "exactly one `def <name>(...):` block, decorators ok, nothing else — no "
                     "imports/constants/other defs). Preferred over edit_file for iterating on "
                     "world_model.py: you send only the function that changed (cheap, no "
                     "string-matching), e.g. tighten transition() or outcome() without "
                     "re-emitting the whole file. The candidate is syntax-checked before "
                     "writing; if `name` doesn't exist yet it is APPENDED. For module-level "
                     "imports/constants, or non-Python files, use write_file/edit_file.",
    "run_python": "Run Python in the workspace and return its stdout. PRE-LOADED for you: "
                  "`grid` (the current frame as a numpy int16 array, shape HxW) and `np` "
                  "(numpy), with numpy printing set to never ellipsize arrays. Tool output has a "
                  "16k flood cap, so query slices or write/read a file for larger dumps. "
                  "So `print(grid[20:25, 25:35])` "
                  "shows that exact sub-region, `print(grid)` shows the whole board, and you "
                  "can run any numpy analysis (np.argwhere(grid==5), np.unique(grid), masks, …). "
                  "Use 2-D indexing grid[r0:r1, c0:c1] (NOT grid[r0:r1][c0:c1]). Also preloaded: "
                  "`wmlib`; `wmlib.frames()` returns current-attempt dict {(level, turn): grid}, and "
                  "`wmlib.transitions()` returns observed {level, turn, prev, next, action, row, col} "
                  "records. After RESET, `wmlib.attempts()` lists prior attempt metadata; pass an "
                  "attempt's `root` to frames()/transitions()/animation_index() to inspect it; pass "
                  "the same root to animation_grids() when loading an archived event. `wmlib.death_events()` "
                  "returns a list of API GAME_OVER event dicts; `wmlib.terminal_events()` returns a "
                  "dict {level: solved-terminal event}. `wmlib.animation_index(level=..., turn=..., last=N)` "
                  "lists compact metadata for matching saved transient animation events, "
                  "and `wmlib.animation_grids(event_or_dir, keyframes=True/False, indices=[...])` loads "
                  "only the animation frames you need. `wmlib.segment(grid)` / `wmlib.segment_summary(grid)` give compact "
                  "connected-component metadata, including exact row-run shape inside each bbox; use "
                  "include_cells=True only when you need exact reconstruction. Saved turn_*.txt files include row/column labels "
                  "and are not np.loadtxt-compatible; use `wmlib.frames()` for prior grids. "
                  "Each run_python call is a fresh process; in-memory variables/functions do not persist, "
                  "but workspace files do. You may create reusable Python helper modules with write_file "
                  "and import them in later calls. cwd is the workspace root, so `import helper` and "
                  "`import world_model` work normally. Default timeout is 15s. For a bounded planner "
                  "or exhaustive analysis, optionally pass `timeout_s` up to 300; prefer the default "
                  "for ordinary inspection. Pass the Python source as `code`.",
    "set_verbosity": "Control how much frame detail the harness inlines each turn, to manage "
                     "your context budget — you decide based on the game. grid: 'on' inlines the "
                     "full grid text every turn; 'off' does not. diff: 'on' inlines the exact "
                     "cell changes; 'summary' inlines only their deterministic count + bounding-box "
                     "header; 'off' does not inline them. The current state is shown in each turn's frame "
                     "header (the starting defaults depend on configuration). E.g. a mostly-static "
                     "game → grid='off' (the diff alone is enough); a busy game → keep both on; "
                     "minimal → both off. The full grid is ALWAYS on disk and readable via "
                     "run_python regardless. Pass either or both args; persists until changed.",
    "invoke_builder": "Hand off to the WORLD-MODEL BUILDER subagent: a focused helper whose only "
                      "job is to construct/refine world_model.py from the transitions observed so "
                      "far and report back (it does NOT take game actions). Call this once you "
                      "have a hypothesis about the mechanics, or after enough exploration that the "
                      "dynamics are learnable — you do not write world_model.py yourself in this "
                      "mode. It returns a compact report (confidence, verify accuracy, a "
                      "recommended next action + subgoal, or 'no model yet'). The report is "
                      "ADVISORY — you decide the action; discount low confidence; if it says no "
                      "model yet, keep exploring. Optional arg `reason`: why you're invoking (what "
                      "you want modeled).",
    "take_action": "Commit ONE environment control; this ENDS the turn and the resulting frame "
                   "arrives next. Args: action (e.g. \"ACTION1\" or \"RESET\"); for ACTION6 (click) also "
                   "pass integer row and col (0-63), matching the grid's y/x labels. "
                   "Example: {\"action\":\"ACTION6\",\"row\":40,\"col\":12}.",
}

_RUN_PYTHON_NO_WM_DESCRIPTION = (
    "Run Python in the workspace and return its stdout. PRE-LOADED for you: `grid` "
    "(the current frame as a numpy int16 array, shape HxW) and `np` (numpy), with "
    "numpy printing set to never ellipsize arrays. Tool output has a 16k flood cap, "
    "so query slices or write/read a file for larger dumps. So `print(grid[20:25, 25:35])` shows that exact "
    "sub-region, `print(grid)` shows the whole board, and you can run any numpy "
    "analysis (np.argwhere(grid==5), np.unique(grid), masks, …). Use 2-D indexing "
    "grid[r0:r1, c0:c1] (NOT grid[r0:r1][c0:c1]). cwd is the workspace root. `wmlib` "
    "is available for observed-frame inspection: `wmlib.frames()` returns the current-attempt "
    "dict {(level, turn): grid}, "
    "and `wmlib.transitions()` returns observed {level, turn, prev, next, action, row, col} records. "
    "After RESET, `wmlib.attempts()` lists prior attempt metadata; pass an attempt's `root` to "
    "frames()/transitions()/animation_index() to inspect it, and to animation_grids() when loading "
    "an archived event. `wmlib.death_events()` returns a list of API GAME_OVER "
    "event dicts. `wmlib.terminal_events()` returns a dict {level: solved-terminal event}. "
    "`wmlib.animation_index(level=..., turn=..., last=N)` lists compact metadata for matching "
    "saved transient animation events. "
    "`wmlib.animation_grids(event_or_dir, keyframes=True/False, indices=[...])` loads only the animation frames you need. "
    "`wmlib.segment(grid)` / `wmlib.segment_summary(grid)` give compact connected-component metadata, "
    "including exact row-run shape inside each bbox; use include_cells=True only when you need exact reconstruction. "
    "Saved turn_*.txt files include row/column labels and are not np.loadtxt-compatible; "
    "use `wmlib.frames()` for prior grids. Each run_python call is a fresh process; in-memory "
    "variables/functions do not persist, but workspace files do. You may create reusable Python "
    "analysis helper modules with write_file and import them in later calls. "
    "Default timeout is 15s. For bounded analysis, optionally pass `timeout_s` up to 300; "
    "prefer the default for ordinary inspection. Pass the Python source as `code`."
)

_SCHEMAS = {
    "ls": {"type": "object", "properties": {"path": {"type": "string"}}},
    "read_file": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "write_file": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    "edit_file": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]},
    "edit_function": {"type": "object", "properties": {"path": {"type": "string"}, "name": {"type": "string"}, "code": {"type": "string"}}, "required": ["path", "name", "code"]},
    "run_python": {"type": "object", "properties": {
        "code": {"type": "string"},
        "timeout_s": {"type": "integer", "minimum": 1, "maximum": 300},
    }, "required": ["code"]},
    "set_verbosity": {"type": "object", "properties": {
                      "grid": {"type": "string", "enum": ["on", "off"]},
                      "diff": {"type": "string", "enum": ["on", "summary", "off"]}}},
    "invoke_builder": {"type": "object", "properties": {"reason": {"type": "string"}}},
    "take_action": {"type": "object", "properties": {"action": {"type": "string"},
                    "row": {"type": "integer"}, "col": {"type": "integer"}}, "required": ["action"]},
}

_ORDER = ["ls", "read_file", "write_file", "edit_file", "run_python", "set_verbosity", "take_action"]

# Multi-action commit (TYCHO_BATCH_ACTIONS): take_action also accepts `actions`, executed in order by
# the agent without further LLM calls until a guard stops it (see TychoAgent._batch_guard).
_TAKE_ACTION_BATCH_DESCRIPTION = (
    "Commit environment controls; this ENDS the turn and the resulting frame arrives next. Either "
    "ONE control: action (e.g. \"ACTION1\" or \"RESET\"; for ACTION6 (click) also integer row and "
    "col, 0-63, matching the grid's y/x labels), or a SEQUENCE: actions, a list of {action, row?, "
    "col?} executed in order, up to {batch_max}. A sequence stops early, skipping the rest, when the "
    "level completes, the game ends, an action changes nothing inside the board, the available "
    "actions change, or a validated plan's trajectory diverges; the next turn reports what ran and "
    "why it stopped. RESET may only be the first action of a sequence. Examples: "
    "{\"action\":\"ACTION6\",\"row\":40,\"col\":12} or "
    "{\"actions\":[{\"action\":\"ACTION4\"},{\"action\":\"ACTION4\"},{\"action\":\"ACTION1\"}]}."
)


def _take_action_batch_schema(batch_max: int) -> dict:
    step = {"type": "object", "properties": {"action": {"type": "string"},
                                              "row": {"type": "integer"}, "col": {"type": "integer"}},
            "required": ["action"]}
    return {"type": "object", "properties": {
        "action": {"type": "string"}, "row": {"type": "integer"}, "col": {"type": "integer"},
        "actions": {"type": "array", "items": step, "maxItems": batch_max},
    }}


def tool_specs(flavor: str = "generic", include_builder: bool = False,
               include_edit_func: bool = False, world_model_enabled: bool = True,
               include_take_action: bool = True,
               include_set_verbosity: bool = True,
               batch_max: int = 0) -> list[dict]:
    """Build a model-flavored tool surface.

    ``include_builder`` adds the actor-only builder handoff, ``include_edit_func`` adds the
    function-granular Python editor, and ``include_set_verbosity`` controls the actor's persistent
    frame-display setting. Builders use the editor but omit verbosity and environment actions.
    ``batch_max > 0`` lets take_action commit a sequence of up to that many actions.
    """
    names = _FLAVOR_NAMES.get(flavor, _FLAVOR_NAMES["generic"])
    order = [c for c in _ORDER if include_take_action or c != "take_action"]
    if not include_set_verbosity:
        order.remove("set_verbosity")
    if include_edit_func:
        order.insert(order.index("edit_file") + 1, "edit_function")
    if include_builder:
        if not include_take_action:
            raise ValueError("include_builder requires include_take_action so the actor can still act")
        order.insert(order.index("take_action"), "invoke_builder")
    out = []
    for c in order:
        desc = _RUN_PYTHON_NO_WM_DESCRIPTION if (c == "run_python" and not world_model_enabled) else _DESCRIPTIONS[c]
        schema = _SCHEMAS[c]
        if c == "take_action" and batch_max > 0:
            desc = _TAKE_ACTION_BATCH_DESCRIPTION.replace("{batch_max}", str(batch_max))
            schema = _take_action_batch_schema(batch_max)
        out.append({"name": names[c], "description": desc, "schema": schema})
    return out


# back-compat default (generic names, used where no flavor is chosen)
TOOL_SPECS = tool_specs("generic")


class ToolExecutor:
    """Executes non-action tools against a workspace. Keys off CANONICAL tool names, so
    it works regardless of the per-model display flavor. take_action is intercepted by
    the agent. Grid inspection is via run_python (`grid`/`np` are pre-loaded) — no bespoke
    zoom tool needed."""

    def __init__(
        self,
        workspace,
        wm_feedback_enabled: bool = True,
        python_sandbox: PythonSandbox | None = None,
    ):
        self.ws = workspace
        self.wm_feedback_enabled = wm_feedback_enabled
        self._python_sandbox = python_sandbox
        self.cur_grid = None  # kept for back-compat (agent sets it); no longer used by a tool
        self.available = []   # the agent sets this each turn (for world-model auto-feedback)

    def execute(self, name: str, args: dict) -> str:
        """Execute one tool and keep host filesystem locations out of model-visible output."""
        try:
            output = self._execute(name, args)
        except Exception as e:
            output = f"(tool {name} error: {type(e).__name__}: {e})"
        return self._portable_output(output)

    def _execute(self, name: str, args: dict) -> str:
        c = canonical(name)
        if c == "ls":
            return self.ws.ls(args.get("path") or ".")
        if c == "read_file":
            return self.ws.read_file(args["path"])
        if c == "write_file":
            before_ast = self._wm_ast(args["path"])
            out = self.ws.write_file(args["path"], args.get("content", ""))
            return out + self._wm_feedback(args["path"], before_ast)
        if c == "edit_file":
            before_ast = self._wm_ast(args["path"])
            out = self.ws.edit_file(args["path"], args.get("old", ""), args.get("new", ""))
            # only auto-verify on a SUCCESSFUL edit (the helper returns "(edit failed…")
            return out + (self._wm_feedback(args["path"], before_ast) if out.startswith("edited") else "")
        if c == "edit_function":
            before_ast = self._wm_ast(args["path"])
            out = self.ws.edit_function(args["path"], args.get("name", ""), args.get("code", ""))
            # auto-verify only on a SUCCESSFUL splice/append (failures start with "(edit_function:")
            return out + (self._wm_feedback(args["path"], before_ast) if out.startswith("edit_function:") and
                          ("replaced" in out or "APPENDED" in out) else "")
        if c == "run_python":
            timeout = args.get("timeout_s", 15)
            if type(timeout) is not int:
                return "(run_python: timeout_s must be an integer from 1 to 300)"
            timeout = max(1, min(300, timeout))
            return self._run_python(args.get("code", ""), timeout=timeout)
        return f"(unknown tool '{name}')"

    def _portable_output(self, value: str) -> str:
        text = value or ""
        workspace = str(self.ws.dir.resolve())
        text = text.replace(workspace, "/workspace")
        home = str(Path.home())
        if home and home != "/":
            # Paths elsewhere below the host home are neither reachable nor useful to the
            # agent. Remove the complete token, including checkout-specific directory names.
            home_path = re.compile(re.escape(home) + r"""(?:/[^\s'"<>()[\]{},;:]*)?""")
            text = home_path.sub("<host-path>", text)
        return text

    # Files whose edit triggers automatic world-model feedback. Only the editable model.
    _WM_FILES = ("world_model.py", "worldmodel.py")

    def _wm_ast(self, path: str) -> tuple[bool, str | None]:
        """Return a parseable semantic fingerprint for a model file."""
        if Path(path).name not in self._WM_FILES:
            return False, None
        try:
            source = self.ws._resolve(path).read_text(errors="replace")
            return True, ast.dump(ast.parse(source), include_attributes=False)
        except (OSError, SyntaxError):
            return False, None

    def _wm_feedback(self, path: str, before_ast: tuple[bool, str | None] | None = None) -> str:
        """After the agent edits world_model.py, AUTOMATICALLY verify it against all observed
        transitions and hand the result back as part of the tool result. Rationale: the agent
        needs this signal anyway; doing it for free (no extra LLM turn to run verify.py) makes
        the world model immediately useful and tightens the build→check loop. If the model
        doesn't import/run, we report the error (after several edits the agent may ignore a
        persistent compile error and keep fixing — that's fine)."""
        if Path(path).name not in self._WM_FILES:
            return ""
        if not self.wm_feedback_enabled:
            return ""
        after_ast = self._wm_ast(path)
        if before_ast and before_ast[0] and after_ast[0] and before_ast[1] == after_ast[1]:
            return ("\n\n[auto world-model feedback]\n"
                    "skipped: Python syntax tree is unchanged (comments/formatting only)")
        out = self._run_python(_WM_FEEDBACK_PROBE, timeout=20)
        return "\n\n[auto world-model feedback]\n" + out.strip()

    # Auto-prepended to every run_python call so the agent can inspect the board with no
    # boilerplate: `np` (numpy), `wmlib` (so the prompt's wmlib.frames()/transitions()/
    # current_grid() work — and to debug a failing turn, e.g. wmlib.frames()[(3,67)] loads
    # the grid at level 3 turn 67), and `grid` (the latest frame) are pre-defined. numpy
    # printing NEVER truncates (threshold=inf) and keeps a 64-wide row on one line
    # (linewidth=300). `grid` is None only before the first frame. The agent never imports
    # numpy/wmlib or sets print options itself.
    # Also preload `frames` (dict {(level,turn): grid}) and `transitions` (list of observed
    # {level,turn,prev,next,action,row,col}) ALREADY EVALUATED — so prior frames/states are one
    # variable away, not a function the agent has to remember to call. The world-model builder in
    # particular works programmatically over these. Computed in a try/except so a pre-first-frame
    # call (or a transient read) can't break run_python.
    _PREAMBLE = (
        "import numpy as _np, wmlib as _wmlib\n"
        "np = _np; wmlib = _wmlib\n"
        "_np.set_printoptions(threshold=_np.inf, linewidth=300)\n"
        "grid = _wmlib.current_grid()\n"
        "try:\n"
        "    frames = _wmlib.frames(); transitions = _wmlib.transitions()\n"
        "except Exception:\n"
        "    frames = {}; transitions = []\n"
    )

    @staticmethod
    def _shellish_to_python(code: str) -> str:
        """run_python executes Python, not a shell — but the prompt (and a model's trained
        instinct) phrases CLI helpers as `python plan.py [args]`. Translate a single such
        line into the equivalent runpy call so the natural form just works instead of
        raising SyntaxError. Only a lone `[python|python3] FILE.py [args...]` line is
        rewritten; anything else (real Python) is passed through untouched."""
        import shlex
        s = code.strip()
        if "\n" in s:
            return code  # multi-line → real Python, don't touch
        try:
            parts = shlex.split(s)
        except ValueError:
            return code
        if parts and parts[0] in ("python", "python3"):
            parts = parts[1:]
        if len(parts) >= 1 and parts[0].endswith(".py"):
            script, argv = parts[0], parts[1:]
            return (f"import sys, runpy\n"
                    f"sys.argv = {[script] + argv!r}\n"
                    f"runpy.run_path({script!r}, run_name='__main__')\n")
        return code

    def _run_python(self, code: str, timeout: int = 15) -> str:
        if not code.strip():
            return "(no code)"
        code = self._shellish_to_python(code)
        if self._python_sandbox is None:
            self._python_sandbox = PythonSandbox()
        result = self._python_sandbox.run_source(
            self.ws.dir,
            self._PREAMBLE + code,
            timeout=timeout,
        )
        if result.timed_out:
            parts = [
                f"(run_python did not finish within {timeout}s; no conclusion about model correctness)"
            ]
            out = _clip_ends(result.stdout or "", 16000).strip()
            err = _clip_ends(result.stderr or "", 4000).strip()
            if out:
                parts.append("[partial stdout before timeout]\n" + out)
            if err:
                parts.append("[partial stderr before timeout]\n" + err)
            return "\n".join(parts)
        # Keep the HEAD, not the tail: a full 64x64 grid print is ~13 KB and the agent reads it
        # top-to-bottom — clipping to the last N chars cut row 0 off, so the model literally could
        # not see the top of the board (observed: Qwen "the output was truncated, let me look at the
        # first row"). Cap generously (a full grid + analysis fits) and, if still over, keep BOTH
        # ends with an explicit middle marker so a runaway loop can't flood context but nothing is
        # silently dropped. The run_python tool desc promises "never truncate" — honor it for any
        # realistic print; the cap is only a flood guard.
        out = _clip_ends(result.stdout or "", 16000)
        err = _clip_ends(result.stderr or "", 4000)
        return out + (f"\n[stderr]\n{err}" if err.strip() else "") or "(no output)"
