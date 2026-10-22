"""Tycho's per-game on-disk working memory.

The harness records each observed frame as text, metadata, and a rendered PNG, plus an exact
within-level cell diff. The agent uses ordinary file and Python tools to inspect this evidence and
maintain notes or an executable world model.

Levels start from a fresh grid, so turns are nested under the level and the turn counter resets at
each boundary:
    <root>/<game>/
      level_0/turn_000.txt      grid, SPACE-SEPARATED single chars (one token/cell, #16)
      level_0/turn_000.png      rendered grid (auto)
      level_0/turn_000.json     {level, turn, action, x, y, state, available}
      level_0/diff_000_001.txt  exact cell deltas within the level (count + bbox + list)
      level_0/death_003.json    GAME_OVER evidence: pre-action grid, terminal grid, action
      attempts/level_0_attempt_000/  immutable prior-attempt observation root
      notes/                     AGENT-OWNED (actor_beliefs.md, world_model.md, level_L_insights.md, scratch)

Grid text uses space-separated hexadecimal cells so multimodal agents can address cells reliably.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from tycho.workspace.version_store import SNAPSHOT_SCHEMA, WorkspaceVersionStore

# Grid serialization format (TYCHO_GRID_FORMAT). Measured token cost vs spaced_hex on a real
# 64x64 frame (Gemma tokenizer): dense 0.51x (merges cells — NOT addressable), spaced 1.0x,
# spaced_rownum 1.04x, spaced_grid8 1.09x, json 1.51x. Default spaced_grid8: row numbers +
# a '|' every 8 columns give orientation (so the LLM can locate diff coords like (1,29))
# for only +9% tokens — directly mitigates the grid-disorientation we observed.
GRID_FORMAT = os.environ.get("TYCHO_GRID_FORMAT", "spaced_grid8")


def grid_text(grid, fmt: str | None = None, col_stride: int = 8) -> str:
    fmt = fmt or GRID_FORMAT
    g = [[int(c) for c in row] for row in grid]
    if fmt == "dense_hex":
        return "\n".join("".join(f"{c:x}" for c in row) for row in g)
    if fmt == "json":
        import json as _j
        return _j.dumps(g)
    if fmt == "spaced_hex":
        return "\n".join(" ".join(f"{c:x}" for c in row) for row in g)
    if fmt == "spaced_rownum":
        return "\n".join(f"y{i:02d}: " + " ".join(f"{c:x}" for c in row) for i, row in enumerate(g))
    # spaced_grid8: explicit x/y axes + a '|' separating every `col_stride` columns, with a
    # header naming each column block (e.g. '00-07 | 08-15 | ...') for orientation, so the
    # agent can locate a coordinate like (1,29) without counting 30 cells.
    W = len(g[0]) if g else 0
    blocks = [(s, min(s + col_stride - 1, W - 1)) for s in range(0, W, col_stride)]
    blockw = col_stride * 2 - 1   # rendered width of one block of single-char cells
    hdr = "x:    " + " | ".join(f"{a:02d}-{b:02d}".ljust(blockw) for a, b in blocks)
    lines = [hdr]
    for i, row in enumerate(g):
        parts = []
        for a, b in blocks:
            parts.append(" ".join(f"{row[j]:x}" for j in range(a, b + 1)))
        lines.append(f"y{i:02d}: " + " | ".join(parts))
    return "\n".join(lines)


def grid_hex(grid) -> str:
    """Legacy COMPACT hex (one char/cell, no separator). Kept for on-disk files the
    agent greps; NOT used for the in-prompt grid (tokenizer merges cells)."""
    return "\n".join("".join(f"{int(c):x}" for c in row) for row in grid)


def _cell_deltas(a, b) -> list[tuple[int, int, int, int]]:
    out = []
    for r in range(min(len(a), len(b))):
        for c in range(min(len(a[r]), len(b[r]))):
            if a[r][c] != b[r][c]:
                out.append((r, c, int(a[r][c]), int(b[r][c])))
    return out


def _runs(cols: list[int]) -> str:
    """Collapse a sorted col list into comma-separated ranges: [36,37,38,40] -> '36-38,40'."""
    cs = sorted(cols); out = []; s = p = cs[0]
    for c in cs[1:]:
        if c == p + 1:
            p = c
        else:
            out.append((s, p)); s = p = c
    out.append((s, p))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in out)


def diff_text(prev, cur) -> str:
    """Compact cell deltas grouped by color transition, with contiguous cells collapsed to
    rectangles/row-ranges (#15). A block recolor that used to be 38 one-cell lines becomes
    one line ('9->8 rows 44-49, cols 36-41') — this matters a lot for context budget, since
    a diff is emitted every changing turn (the dominant context cost on editing-heavy games).
    Lossless: the exact cells are recoverable from the ranges."""
    from collections import defaultdict
    d = _cell_deltas(prev, cur)
    if not d:
        return "no cells changed"
    rs = [r for r, _, _, _ in d]; cs = [c for _, c, _, _ in d]
    bbox = f"rows {min(rs)}-{max(rs)}, cols {min(cs)}-{max(cs)}"
    by_trans: dict = defaultdict(lambda: defaultdict(list))   # (old,new) -> row -> [cols]
    for r, c, o, n in d:
        by_trans[(o, n)][r].append(c)
    lines = []
    for (o, n), rows in sorted(by_trans.items()):
        # per-row col-ranges, then merge consecutive rows that share an identical col-spec
        rowspec = [(r, _runs(rows[r])) for r in sorted(rows)]
        i = 0
        while i < len(rowspec):
            r0, spec = rowspec[i]; j = i
            while j + 1 < len(rowspec) and rowspec[j + 1][1] == spec and rowspec[j + 1][0] == rowspec[j][0] + 1:
                j += 1
            rng = f"rows {r0}-{rowspec[j][0]}" if j > i else f"row {r0}"
            lines.append(f"  {o}->{n}  {rng}, cols {spec}")
            i = j + 1
    return f"{len(d)} cells changed; region [{bbox}]\n" + "\n".join(lines)


def _parse_grid_text(text: str) -> list[list[int]]:
    """Parse any grid_text() format currently written to turn_*.txt."""
    s = text.strip()
    if not s:
        return []
    if s.startswith("["):
        return [list(map(int, row)) for row in json.loads(s)]
    out: list[list[int]] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line or line.startswith("x:"):
            continue
        if ":" in line and line.split(":", 1)[0].strip().startswith(("y", "r")):
            line = line.split(":", 1)[1]
        line = line.replace("|", " ")
        toks = line.split()
        if not toks:
            continue
        if len(toks) == 1 and all(ch.lower() in "0123456789abcdef" for ch in toks[0]):
            out.append([int(ch, 16) for ch in toks[0]])
        else:
            out.append([int(tok, 16) if tok.lower() in "abcdef" else int(tok) for tok in toks])
    return out


class GameWorkspace:
    def __init__(self, game_id: str, root: str | None = None, render: bool = True,
                 available_actions: list | None = None, resume: bool = False,
                 render_scale: int | None = None, seed_world_model: bool = True):
        short = game_id.split("-")[0]
        base = Path(root) if root else Path(tempfile.mkdtemp(prefix="arcws_"))
        self.dir = base / short
        if root and not resume and self.dir.exists():
            shutil.rmtree(self.dir)
        (self.dir / "notes").mkdir(parents=True, exist_ok=True)
        self.render = render
        # px/cell for the frame PNG. The agent passes its per-model vision profile's scale (Qwen→32
        # for 1-token-per-cell, others→6); None falls back to the old env/default for callers that
        # don't supply one (tests, smoke). NOT read from env here so the model-aware choice can't be
        # silently overridden — the override (TYCHO_RENDER_SCALE) is resolved in vision.vision_profile.
        self.render_scale = render_scale
        self.prev_grid = None      # for the within-level diff; reset on a new level
        self.turns: list[dict] = []  # lightweight observation index
        # Exact resume restores the complete committed directory before constructing this object.
        # Do not refresh even harness-authored helpers here: changing any byte would make the
        # resumed workspace differ from the checkpoint the actor previously observed.
        self._seed_worldmodel(available_actions or [], resume=resume, seed_world_model=seed_world_model)
        self._version_store = WorkspaceVersionStore(self.dir)

    def _seed_worldmodel(self, available: list, resume: bool = False, seed_world_model: bool = True) -> None:
        """Seed the world-modelling files into the workspace root so `import wmlib`,
        `import world_model`, and `python verify.py/plan.py` just work. The editable
        world_model.py has actions() PRE-FILLED with this game's available actions
        (stable per game; ACTION6 gets a no-coords placeholder the agent must fill).

        resume=True leaves the exact restored checkpoint untouched."""
        import tycho.workspace.wm_templates as T
        from pathlib import Path as _P
        if resume:
            return
        # wmlib + helper scripts are OURS (not agent-authored) — always refresh to current version.
        wmlib_src = _P(__file__).with_name("wmlib_template.py").read_text()
        (self.dir / "wmlib.py").write_text(wmlib_src)
        if seed_world_model:
            (self.dir / "verify.py").write_text(T.VERIFY)
            (self.dir / "plan.py").write_text(T.PLAN)
        else:
            for stale in ("verify.py", "plan.py", "world_model.py", "predict.py"):
                (self.dir / stale).unlink(missing_ok=True)
            return
        # remove a stale predict.py from older workspaces (resume); harmless on a fresh dir.
        stale_predict = self.dir / "predict.py"
        if stale_predict.exists():
            stale_predict.unlink()
        # world_model.py is AGENT-AUTHORED: seed it only on a fresh start, preserve it on resume.
        wm_path = self.dir / "world_model.py"
        if not (resume and wm_path.exists()):
            names = [a if isinstance(a, str) else a.name for a in available]
            names = [n for n in names if n != "RESET"]
            items = ", ".join(f'{{"action": "{n}", "row": None, "col": None}}' for n in names) or \
                    '{"action": "ACTION1", "row": None, "col": None}'
            wm_path.write_text(T.WORLD_MODEL.replace("{ACTIONS_LINE}", f"[{items}]"))

    def _ldir(self, level: int) -> Path:
        d = self.dir / f"level_{level}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- harness side: called once per turn with the observed frame ----
    def record(self, grid, *, level: int, turn_in_level: int, action: str = "",
               row=None, col=None, state: str = "", available=None) -> None:
        g = [list(map(int, r)) for r in grid]
        ld = self._ldir(level)
        meta = {"level": level, "turn": turn_in_level, "action": action, "row": row, "col": col,
                "state": state, "available": list(available or [])}
        (ld / f"turn_{turn_in_level:03d}.txt").write_text(grid_text(g))
        (ld / f"turn_{turn_in_level:03d}.json").write_text(json.dumps(meta))
        # No automatic object decomposition: the agent chooses the state abstraction.
        if self.render:
            self._render_png(ld / f"turn_{turn_in_level:03d}.png", g)
        # within-level diff only (turn_in_level 0 starts a fresh level → no diff)
        if turn_in_level > 0 and self.prev_grid is not None:
            (ld / f"diff_{turn_in_level-1:03d}_{turn_in_level:03d}.txt").write_text(
                diff_text(self.prev_grid, g))
        self.prev_grid = g
        self.turns.append(meta)
        (self.dir / ".current_frame.json").write_text(json.dumps({
            "level": level,
            "turn": turn_in_level,
            "path": f"level_{level}/turn_{turn_in_level:03d}.txt",
        }))

    def record_terminal(self, level: int, terminal_grid, action: str = "", row=None, col=None) -> None:
        """Record a level's OBSERVED winning terminal in a SEPARATE channel (level_<L>/terminal.json),
        NOT as a turn_*.txt. The terminal is the goal-reached frame the winning action produced (the
        modeled post-action state of this level); the agent never ACTS from it, so it is not a decision
        frame and must not enter frames()/transitions() as ordinary dynamics. verify_outcome reads it as
        the render ground-truth: replay to the pre-win state, apply the winning action, then require
        outcome(s_terminal)=="level_complete" AND render(s_terminal)==this grid. pre_turn = the last observed decision
        turn of the level (the state the winning action was taken from)."""
        ld = self._ldir(level)
        import glob as _glob
        pre_turn = len(_glob.glob(str(ld / "turn_*.txt"))) - 1  # last recorded decision turn
        g = [list(map(int, r)) for r in terminal_grid]
        event = {"level": level, "pre_turn": pre_turn, "outcome": "win",
                 "action": {"action": action, "row": row, "col": col},
                 "terminal_grid": g}
        (ld / "terminal.json").write_text(json.dumps(event))
        (ld / "terminal.txt").write_text(grid_text(g))
        if self.render:
            self._render_png(ld / "terminal.png", g)

    def record_solved(self, level: int, action: str, row=None, col=None) -> None:
        """Record the ACTION that COMPLETED a level.

        The winning action is needed even when the observed terminal grid is recorded separately in
        terminal.json: verify_outcome replays to the pre-win state, applies this action, then checks that
        the modeled post-action state satisfies outcome()=="level_complete" and, when available, renders the observed
        terminal. Every ordinary turn_*.txt frame remains a pre-terminal decision state.
        """
        ld = self._ldir(level)
        (ld / "solved.json").write_text(json.dumps({"action": action, "row": row, "col": col}))

    def record_game_over(self, *, level: int, turn_in_level: int, action: str,
                         row=None, col=None, prev_grid=None, game_over_grid=None) -> dict | None:
        """Record a fatal action as negative evidence without adding it to normal transitions.

        GAME_OVER is followed by a RESET to a fresh start. reset_level() must still delete ordinary
        turn/diff files for the failed attempt, otherwise wmlib.transitions() can stitch a pre-death
        frame to a reset frame as if it were a normal mechanic. This side-channel preserves the
        important evidence: the exact frame before the fatal action and the frame the engine returned
        as GAME_OVER.
        """
        if prev_grid is None or game_over_grid is None:
            return
        prev = [list(map(int, r)) for r in prev_grid]
        nxt = [list(map(int, r)) for r in game_over_grid]
        ld = self._ldir(level)
        attempt_start, attempt_actions = self._attempt_prefix(level)
        event = {
            "level": level,
            "turn": turn_in_level,
            "action": action,
            "row": row,
            "col": col,
            "state": "GAME_OVER",
            "prev": prev,
            "next": nxt,
        }
        if attempt_start is not None:
            event["attempt_start"] = attempt_start
            event["attempt_actions"] = attempt_actions
        stem = self._death_stem(ld, turn_in_level)
        event["stem"] = stem
        (ld / f"{stem}.json").write_text(json.dumps(event))
        (ld / f"{stem}_prev.txt").write_text(grid_text(prev))
        (ld / f"{stem}_next.txt").write_text(grid_text(nxt))
        (ld / f"{stem}_diff.txt").write_text(diff_text(prev, nxt))
        if self.render:
            self._render_png(ld / f"{stem}_prev.png", prev)
            self._render_png(ld / f"{stem}_next.png", nxt)
        return event

    def record_animation_event(self, *, level: int, turn_in_level: int, action: str,
                               row=None, col=None, terminal: str = "nonterminal",
                               frames=None, selected_indices=None, decision=None,
                               summary: str = "", reused: bool = False, skipped: str = "",
                               signature=None, frame_hash: str = "", keep_per_level: int = 5,
                               storage_root: str | None = None) -> dict | None:
        """Persist transient animation frames for later agent inspection.

        These are not decision frames and must not enter frames()/transitions(). They are an
        auxiliary side-channel analogous to death/terminal evidence: useful when the engine
        returned informative intermediate frames but the next playable grid alone is ambiguous.
        """
        if keep_per_level <= 0 or frames is None:
            return None
        frames = list(frames)
        if not frames:
            return None
        selected = list(selected_indices or [])
        base = self.dir if storage_root is None else self._resolve(storage_root)
        ld = base / f"level_{level}"
        ld.mkdir(parents=True, exist_ok=True)
        stem = self._animation_stem(ld, turn_in_level, action)
        event_dir = ld / stem
        event_dir.mkdir(parents=True, exist_ok=True)

        all_frame_files: list[dict] = []
        frame_files_by_index: dict[int, dict] = {}
        for idx, frame in enumerate(frames):
            grid = [list(map(int, r)) for r in frame]
            txt_name = f"frame_{idx:03d}.txt"
            png_name = f"frame_{idx:03d}.png"
            (event_dir / txt_name).write_text(grid_text(grid))
            if self.render:
                self._render_png(event_dir / png_name, grid)
            item = {
                "index": int(idx),
                "txt": f"level_{level}/{stem}/{txt_name}",
                "png": f"level_{level}/{stem}/{png_name}" if self.render else None,
            }
            all_frame_files.append(item)
            frame_files_by_index[idx] = item

        selected_frame_files: list[dict] = []
        for idx in selected:
            if idx in frame_files_by_index:
                selected_frame_files.append(frame_files_by_index[idx])

        meta = {
            "level": level,
            "turn": turn_in_level,
            "action": action,
            "row": row,
            "col": col,
            "terminal": terminal,
            "directory": f"level_{level}/{stem}",
            "original_frame_count": int(len(frames)),
            "selected_frame_indices": [int(i) for i in selected],
            "selected_frame_files": selected_frame_files,
            "all_frame_files": all_frame_files,
            "decision": decision or {},
            "summary": summary,
            "summary_reused": bool(reused),
            "summary_skipped": skipped,
            "signature": signature,
            "selected_frame_hash": frame_hash,
        }
        (event_dir / "meta.json").write_text(json.dumps(meta))
        self._prune_animation_events(ld, keep_per_level)
        if storage_root is not None:
            meta["workspace_directory"] = f"{storage_root.rstrip('/')}/{meta['directory']}"
        return meta

    def _animation_stem(self, ld: Path, turn_in_level: int, action: str) -> str:
        safe_action = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(action or "action"))
        safe_action = safe_action[:48].strip("_") or "action"
        base = f"animation_{turn_in_level:03d}_{safe_action}"
        if not (ld / base).exists():
            return base
        i = 2
        while (ld / f"{base}_{i:03d}").exists():
            i += 1
        return f"{base}_{i:03d}"

    def _animation_event_sort_key(self, p: Path) -> tuple[int, int, str]:
        try:
            meta = json.loads((p / "meta.json").read_text())
            return int(meta.get("turn", -1)), 0, p.name
        except Exception:  # noqa: BLE001
            pass
        try:
            return int(p.name.split("_", 2)[1]), 1, p.name
        except Exception:  # noqa: BLE001
            return -1, 2, p.name

    def _prune_animation_events(self, ld: Path, keep_per_level: int) -> None:
        events = sorted(
            [p for p in ld.glob("animation_*") if p.is_dir()],
            key=self._animation_event_sort_key,
        )
        for p in events[:-keep_per_level]:
            shutil.rmtree(p, ignore_errors=True)

    def _death_stem(self, ld: Path, turn_in_level: int) -> str:
        base = f"death_{turn_in_level:03d}"
        if not (ld / f"{base}.json").exists():
            return base
        i = 2
        while (ld / f"{base}_{i:03d}.json").exists():
            i += 1
        return f"{base}_{i:03d}"

    def _attempt_prefix(self, level: int) -> tuple[list[list[int]] | None, list[dict]]:
        ld = self._ldir(level)
        start_path = ld / "turn_000.txt"
        if not start_path.exists():
            return None, []
        try:
            attempt_start = _parse_grid_text(start_path.read_text(errors="replace"))
        except Exception:  # noqa: BLE001
            return None, []
        actions: list[dict] = []
        def _turn_num(path: Path) -> int:
            try:
                return int(path.stem.split("_")[1])
            except Exception:  # noqa: BLE001
                return 0

        for p in sorted(ld.glob("turn_*.json"), key=_turn_num):
            try:
                meta = json.loads(p.read_text(errors="replace"))
            except Exception:  # noqa: BLE001
                continue
            act = meta.get("action")
            if not act or act == "RESET":
                continue
            actions.append({"action": act, "row": meta.get("row"), "col": meta.get("col")})
        return attempt_start, actions

    def new_level(self) -> None:
        """Called when the level advances: the next frame starts fresh (no diff back)."""
        self.prev_grid = None

    def _archive_attempt(self, level: int, *, reason: str) -> dict | None:
        """Move the current attempt's decision evidence to an immutable observation root.

        A reset must remove these files from ``level_<L>`` so the next reset frame cannot be
        mistaken for a normal transition. Archiving them first preserves experiments the agent
        already paid for. Existing wmlib.frames(root=...)/transitions(root=...) can read the
        archive without adding parallel history APIs.
        """
        ld = self._ldir(level)
        turn_files = sorted(ld.glob("turn_*.txt"))
        if not turn_files:
            return None
        attempts_dir = self.dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"level_{level}_attempt_"
        used = []
        for path in attempts_dir.glob(f"{prefix}*"):
            try:
                used.append(int(path.name.removeprefix(prefix)))
            except ValueError:
                continue
        attempt = max(used, default=-1) + 1
        archive = attempts_dir / f"{prefix}{attempt:03d}"
        archived_level = archive / f"level_{level}"
        archived_level.mkdir(parents=True)

        patterns = (
            "turn_*.txt", "turn_*.json", "turn_*.png", "diff_*.txt", "scene_*.json",
            "animation_*",
        )
        moved = []
        for pat in patterns:
            for path in sorted(ld.glob(pat)):
                shutil.move(str(path), str(archived_level / path.name))
                moved.append(path.name)
        animation_names = [name for name in moved if name.startswith("animation_")]
        if animation_names:
            # Keep the durable note's direct file pointers valid after moving the attempt.
            note = self.dir / "notes" / "animation_evidence.md"
            if note.exists():
                text = note.read_text(errors="replace")
                archive_rel = archive.relative_to(self.dir).as_posix()
                for name in animation_names:
                    old = f"frames: level_{level}/{name}"
                    new = f"frames: {archive_rel}/level_{level}/{name}"
                    text = text.replace(old, new)
                note.write_text(text)
        manifest = {
            "level": int(level),
            "attempt": int(attempt),
            "reason": str(reason),
            "n_frames": len(turn_files),
            "files": moved,
        }
        (archive / "attempt.json").write_text(json.dumps(manifest, sort_keys=True))
        return {**manifest, "root": archive.relative_to(self.dir).as_posix()}

    def reset_level(self, level: int, *, reason: str = "reset") -> dict | None:
        """Called after an engine GAME_OVER/RESET on the same level.

        The reset frame is a fresh start, not the result of the preceding action. Archive and clear
        the current attempt's observed turn files so wmlib.transitions() cannot stitch a pre-reset
        frame to the reset frame as a fake transition. Prior attempts stay explicitly available via
        wmlib.attempts() and wmlib.frames(root=attempt["root"]).
        """
        ld = self._ldir(level)
        archived = self._archive_attempt(level, reason=reason)
        for pat in ("turn_*.txt", "turn_*.json", "turn_*.png", "diff_*.txt"):
            for p in ld.glob(pat):
                p.unlink(missing_ok=True)
        for p in ld.glob("scene_*.json"):
            p.unlink(missing_ok=True)
        self.prev_grid = None
        self.turns = [t for t in self.turns if t.get("level") != level]
        cur = self.dir / ".current_frame.json"
        if cur.exists():
            try:
                data = json.loads(cur.read_text())
            except Exception:  # noqa: BLE001 - stale metadata should not block reset cleanup
                data = {}
            if data.get("level") == level:
                cur.unlink(missing_ok=True)
        return archived

    def _render_png(self, path, grid) -> None:
        # px/cell. The agent supplies render_scale from the model's vision profile (vision.py):
        # Qwen3.x → 32 (2048x2048 → exactly 4096 visual tokens, one per ARC cell, no downscale);
        # others → 6 (384x384, a layout aid). Fall back to the env/default only when no scale was
        # passed (tests/smoke). The model-aware path is the live one — see vision.vision_profile.
        scale = self.render_scale if self.render_scale is not None \
            else int(os.environ.get("TYCHO_RENDER_SCALE", "6"))
        try:
            from arc_agi.rendering import frame_to_rgb_array
            from PIL import Image
            arr = frame_to_rgb_array(0, np.asarray(grid), scale=scale)
            Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)
        except Exception:
            pass  # rendering is a convenience; never block a turn on it

    def current_png(self, level: int, turn_in_level: int) -> bytes | None:
        p = self.dir / f"level_{level}" / f"turn_{turn_in_level:03d}.png"
        return p.read_bytes() if p.exists() else None

    def zoom(self, grid, r0: int, c0: int, r1: int, c1: int) -> str:
        """#17: a sub-grid at full per-cell resolution, with ABSOLUTE row/col labels so the
        coordinates match the full grid (and the diff coords). Clamped."""
        N, W = len(grid), len(grid[0]) if grid else 0
        r0, r1 = max(0, min(r0, N - 1)), max(0, min(r1, N - 1))
        c0, c1 = max(0, min(c0, W - 1)), max(0, min(c1, W - 1))
        if r1 < r0 or c1 < c0:
            return "(empty bbox)"
        colhdr = "     " + " ".join(f"{c:x}" if c < 16 else str(c) for c in range(c0, c1 + 1))
        # use the real column INDICES as a header (so the agent can read exact x)
        colhdr = "cols: " + " ".join(str(c) for c in range(c0, c1 + 1))
        lines = [f"zoom rows {r0}-{r1}, cols {c0}-{c1} (absolute coords):", colhdr]
        for r in range(r0, r1 + 1):
            lines.append(f"r{r}: " + " ".join(f"{int(grid[r][c]):x}" for c in range(c0, c1 + 1)))
        return "\n".join(lines)

    # ---- agent side: the file primitives, sandboxed to the workspace dir ----
    def _resolve(self, rel: str) -> Path:
        root = self.dir.resolve()
        p = (root / rel).resolve()
        try:
            p.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes workspace: {rel}") from exc
        return p

    def read_file(self, path: str, max_bytes: int = 20000) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"(no such file: {path}). Use ls to see what's available."
        data = p.read_text(errors="replace")
        return data[:max_bytes] + (f"\n…(truncated, {len(data)} bytes)" if len(data) > max_bytes else "")

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    def validated_plan_hint(self, grid, *, level: int, turn: int, available=(), batch_max: int = 0) -> str | None:
        """Return the next canonical-plan action only while its observed trajectory still matches.

        plan.py writes the artifact after replaying a candidate through world_model.py. The actor
        still commits one action at a time; this compact hint carries the plan across turns and
        stops at a changed model, an unavailable action, or the first unexpected frame.
        """
        artifact_path = self.dir / "notes" / "validated_plan.json"
        model_path = self.dir / "world_model.py"
        try:
            artifact = json.loads(artifact_path.read_text())
            start = artifact["start"]
            actions = artifact["actions"]
            expected = artifact["expected_grid_sha256"]
            plan_length = int(artifact["plan_length"])
            if (
                artifact.get("status") != "validated"
                or plan_length != len(actions)
                or plan_length != len(expected)
                or int(start["level"]) != int(level)
            ):
                return None
            model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if model_hash != artifact.get("world_model_sha256"):
                return "Validated plan paused: world_model.py changed after validation; re-plan before continuing."
            offset = int(turn) - int(start["turn"])
            if offset < 0 or offset > plan_length:
                return None
            grid_hash = hashlib.sha256(np.asarray(grid, dtype=np.int16).tobytes()).hexdigest()
            expected_hash = start["grid_sha256"] if offset == 0 else expected[offset - 1]
            if grid_hash != expected_hash:
                return (
                    f"Validated plan paused: observed frame diverged after {offset}/{plan_length} "
                    "planned action(s); inspect the new evidence and re-plan."
                )
            if offset == plan_length:
                return (
                    f"Validated plan trajectory matched all {plan_length} action(s); verify the "
                    "observed outcome before acting again."
                )
            action = actions[offset]
            name = str(action.get("action", ""))
            legal = {str(item) for item in available}
            if legal and name not in legal:
                return (
                    f"Validated plan paused after {offset}/{plan_length} action(s): next action "
                    f"{name or '?'} is not available on this frame."
                )
            if name == "ACTION6" and action.get("row") is not None and action.get("col") is not None:
                next_action = f"ACTION6(row={int(action['row'])},col={int(action['col'])})"
            else:
                next_action = name or "?"
            hint = (
                f"Validated plan continuation: observed trajectory matches {offset}/{plan_length} "
                f"action(s); next action is {next_action}. Stop and re-plan on any later divergence."
            )
            if batch_max > 1 and plan_length - offset > 1:
                # TYCHO_PLAN_BATCH_HINT: the rest of the plan as one take_action sequence; the batch
                # guard stops it at the first frame that leaves the validated trajectory.
                rest = [{k: (int(v) if k in ("row", "col") else str(v)) for k, v in a.items()
                         if k in ("action", "row", "col") and v is not None}
                        for a in actions[offset:offset + batch_max]]
                hint += (f" To play the next {len(rest)} validated action(s) in one commit (it stops "
                         f"by itself on divergence): take_action with actions={json.dumps(rest)}")
            return hint
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def edit_file(self, path: str, old: str, new: str) -> str:
        """Exact string replace (like Claude-Code Edit / Gemini-CLI replace). `old` must
        occur EXACTLY ONCE — else report so (no silent partial edit)."""
        p = self._resolve(path)
        if not p.exists():
            return f"(no such file: {path})"
        data = p.read_text(errors="replace")
        n = data.count(old)
        if n == 0:
            return f"(edit failed: `old` string not found in {path})"
        if n > 1:
            return f"(edit failed: `old` occurs {n}x in {path} — make it unique)"
        p.write_text(data.replace(old, new, 1))
        return f"edited {path} (replaced 1 occurrence)"

    def edit_function(self, path: str, name: str, code: str) -> str:
        """Replace (or append) a TOP-LEVEL Python function by NAME — the function-granular
        editor (TYCHO_EDIT_FUNC). Motivated by the world-model refine loop: re-emitting all of
        world_model.py on every edit→verify cycle is token-wasteful; emitting just one
        transition()/goal_test() is not. More robust than string-match Edit for this case
        (no 'old occurs Nx' / 'not found' failure mode), and 1:1 with the agent's mental model
        (the file IS named functions). Locates `def <name>` at MODULE level via AST (so it
        survives reformatting), validates the candidate parses AND defines exactly `def name`,
        then splices its full line span (decorators included). If the function doesn't exist
        yet, APPENDS it. Falls back to Write/Edit on any structural mismatch — never a silent
        partial edit. Scope is module-level functions only (not methods inside a class); for
        anything else use Write/Edit."""
        import ast
        p = self._resolve(path)
        if p.suffix != ".py":
            return f"(edit_function only edits .py files; {path} is not Python — use edit_file/write_file)"
        # 1. the candidate must itself be a single, parseable, correctly-named function.
        snippet = code.strip("\n")
        try:
            cand = ast.parse(snippet)
        except SyntaxError as e:
            return (f"(edit_function: the `code` you gave does not parse: {e.msg} at line {e.lineno}. "
                    "Fix the syntax and retry, or use write_file for the whole file.)")
        cand_fns = [n for n in cand.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(cand.body) != len(cand_fns) or len(cand_fns) != 1:
            return ("(edit_function: `code` must be EXACTLY ONE top-level function definition "
                    f"(def {name}(...): …) and nothing else. For imports/constants/multiple "
                    "defs use write_file or edit_file.)")
        if cand_fns[0].name != name:
            return (f"(edit_function: `code` defines `{cand_fns[0].name}` but you asked to edit "
                    f"`{name}`. Make them match.)")
        # 2. locate the existing module-level def (None → append).
        existing = p.read_text(errors="replace") if p.exists() else ""
        target = None
        if existing.strip():
            try:
                tree = ast.parse(existing)
            except SyntaxError as e:
                return (f"(edit_function: the EXISTING {path} doesn't parse ({e.msg} at line "
                        f"{e.lineno}), so I can't safely locate `{name}`. Use write_file to "
                        "rewrite the file cleanly.)")
            for node in tree.body:  # module level only
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    target = node
                    break
        lines = existing.splitlines()
        if target is None:
            # append the new function (with a separating blank line if the file is non-empty)
            sep = "" if not existing or existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
            new_text = existing + sep + snippet + "\n"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new_text)
            return f"edit_function: `{name}` not found in {path} → APPENDED it ({len(snippet)} bytes)."
        # splice: replace the def's full span (incl. decorators) with the candidate, preserving
        # the def's own indentation level (module-level = column 0, but be defensive).
        start = (min(d.lineno for d in target.decorator_list) if target.decorator_list
                 else target.lineno) - 1
        end = target.end_lineno  # 1-based inclusive → slice end is exclusive at this value
        indent = " " * target.col_offset
        body = "\n".join((indent + ln if ln else ln) for ln in snippet.splitlines())
        new_lines = lines[:start] + body.splitlines() + lines[end:]
        new_text = "\n".join(new_lines) + ("\n" if existing.endswith("\n") else "")
        p.write_text(new_text)
        return f"edit_function: replaced `{name}` in {path} (lines {start+1}-{end})."

    def ls(self, path: str = ".") -> str:
        p = self._resolve(path)
        if p.is_file():
            return path
        entries = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
        return "\n".join(entries) if entries else "(empty)"

    def snapshot(self, level: int, turn_in_level: int) -> dict:
        """Capture exact causal workspace state plus the current observation for trace analysis.

        Agent-controlled files are content-addressed, including binary maps. Text remains in
        ``contents`` for self-contained trace analysis and is version-deduplicated when the
        record is written. Harness-owned frame/event archives are recorded through their
        dedicated channels and are intentionally excluded from this manifest.
        """
        import base64
        file_versions, contents, snapshot_warnings = self._version_store.capture()
        files = sorted(file_versions)
        ld = self.dir / f"level_{level}"
        cur_grid = (ld / f"turn_{turn_in_level:03d}.txt")
        diffs = sorted(ld.glob(f"diff_*_{turn_in_level:03d}.txt"))
        # exactly ONE image: the current frame (base64), for the #5 inline-prompt thumbnail.
        images = {}
        cur_png = ld / f"turn_{turn_in_level:03d}.png"
        if cur_png.exists():
            try:
                images[str(cur_png.relative_to(self.dir))] = "data:image/png;base64," + \
                    base64.b64encode(cur_png.read_bytes()).decode()
            except Exception:  # noqa: BLE001
                pass
        return {"snapshot_schema": SNAPSHOT_SCHEMA,
                "files": files, "file_versions": file_versions,
                "contents": contents, "images": images,
                "snapshot_warnings": snapshot_warnings,
                "level": level, "turn": turn_in_level,
                "cur_grid": cur_grid.read_text()[:8000] if cur_grid.exists() else "",
                "cur_diff": diffs[-1].read_text()[:4000] if diffs else "(no diff — first turn of level)"}
