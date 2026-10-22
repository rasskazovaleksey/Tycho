"""Typed, validated runtime settings for the Tycho agent core.

Each setting is parsed and range-checked once at launch. Context retention, model-aware vision, and
orchestration have dedicated typed resolvers; this object owns the remaining agent-core settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict


def _envbool(name: str, default: bool = False) -> bool:
    """The repo's bool convention: truthy iff the value is one of 1/on/true/yes (case-insensitive).
    Matches every `os.environ.get(X, "") in ("1","on","true")` site (yes added — harmless superset)."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "on", "true", "yes")


def _envint(name: str, default: int) -> int:
    """int(env or default). Empty string -> default (an explicitly-empty var means 'unset' here)."""
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


def _envfloat(name: str, default: float) -> float:
    """float(env or default). Empty string means unset, matching ``_envint``."""
    v = os.environ.get(name, "")
    return float(v) if v.strip() else default


@dataclass(frozen=True)
class TychoSettings:
    """Fully-resolved agent-core config for one run. Every field is parsed + range-checked at
    from_env(); read these instead of os.environ in agent.py. (Backend/serving knobs stay in
    llm_client; context/vision/mode have their own typed resolvers this composes with.)"""
    effort: str                  # TYCHO_EFFORT, falling back to LLM_EFFORT, then "medium"
    max_tool_steps: int          # cognitive tool calls/turn before a forced action
    max_calls: int               # whole-game LLM-call ceiling
    max_inference_cost_per_game: float   # public-list USD equivalent; 0 = disabled
    max_inference_cost_per_level: float  # public-list USD equivalent; 0 = disabled
    max_tokens: int              # base output budget; Responses adds reasoning headroom
    retry_max_tokens: int        # retry max_tokens (0 = same)
    grid_keep: int               # turns of grid/image kept in context
    reasoning_keep: int          # raw_reasoning turns kept (defaults to grid_keep)
    vision: bool                 # attach the current-frame PNG
    text_grid: str               # full | diff | none
    grid_embeds: bool            # attach exact per-cell prompt_embeds
    tool_flavor: str             # "" = auto from model; else claude|gemini|generic
    mode: str                    # TYCHO_MODE (validated against the MODES table by resolve_mode)
    edit_func: bool              # add edit_function to world-model actors (builders always have it)
    meta_reflect: bool           # dev-only meta-reflection addendum
    agent_graph_json: str        # TYCHO_AGENT_GRAPH (declarative orchestration graph, JSON or "")
    log_io: bool                 # LLM_LOG == "1": capture per-call I/O
    animation_summary: bool      # summarize selected information-rich transition animations
    animation_summary_video: bool
    animation_summary_video_format: str
    animation_summary_max_per_level: int
    animation_evidence_keep_per_level: int
    animation_summary_max_keyframes: int
    animation_summary_include_game_over: bool
    animation_summary_max_tokens: int
    batch_actions: bool          # take_action accepts an `actions` list (multi-action commit)
    batch_max: int               # most actions one take_action may commit
    batch_noop_guard_from_level: int  # 0-based level from which a no-op action stops a batch
    batch_border: int            # outer cells ignored by the batch no-op guard
    strict_receipts: bool        # a 2nd take_action in one reply is answered "NOT executed", not "committed"
    scribe_max_calls: int        # LLM calls the level-boundary summary may make
    scribe_max_tokens: int       # output cap per summary call (0 = max_tokens)
    builder_gate: str            # "" (upstream) | "stuck": also auto-fire the builder on a stuck level
    builder_stuck_tokens: int    # stuck = this many generated tokens on the level per auto-fire so far + 1
    builder_stuck_deaths: int    # ... or this many GAME_OVERs on the level per auto-fire so far + 1
    builder_max_per_level: int   # builder runs per level, actor-pulled or automatic (0 = unlimited)
    builder_max_tokens: int      # output cap per builder call (0 = max_tokens)
    wm_status_line: bool         # one verify line on every frame message while a world model exists
    plan_batch_hint: bool        # the plan hint offers the remaining validated plan as one sequence

    # ---- derived (not env): the two text-grid toggles _frame_message reads ----
    @property
    def show_grid(self) -> bool:
        return self.text_grid == "full"

    @property
    def show_diff(self) -> bool:
        return self.text_grid in ("full", "diff")

    @classmethod
    def from_env(cls) -> "TychoSettings":
        """Resolve + VALIDATE the agent-core config from the environment. Parsing errors / out-of-range
        values raise HERE (launch), not mid-run. Mirrors the exact defaults of the pre-migration inline
        reads in agent.py (asserted byte-equivalent by test_settings)."""
        grid_keep = _envint("TYCHO_GRID_KEEP", 3)
        text_grid = os.environ.get("TYCHO_TEXT_GRID", "full").lower()
        if text_grid not in ("full", "diff", "none"):
            raise ValueError(f"TYCHO_TEXT_GRID={text_grid!r} not in full|diff|none")
        mode = os.environ.get("TYCHO_MODE", "single").lower()
        from tycho.workspace.optional_channels import grid_embeds_enabled

        s = cls(
            effort=os.environ.get("TYCHO_EFFORT", os.environ.get("LLM_EFFORT", "medium")),
            max_tool_steps=_envint("TYCHO_MAX_TOOL_STEPS", 25),
            max_calls=_envint("TYCHO_MAX_LLM_CALLS", 1100),
            max_inference_cost_per_game=_envfloat("TYCHO_MAX_INFERENCE_COST_PER_GAME", 0.0),
            max_inference_cost_per_level=_envfloat("TYCHO_MAX_INFERENCE_COST_PER_LEVEL", 0.0),
            max_tokens=_envint("TYCHO_MAX_TOKENS", 24000),
            retry_max_tokens=_envint("TYCHO_MAX_TOKENS_RETRY", 0),
            grid_keep=grid_keep,
            reasoning_keep=_envint("TYCHO_REASONING_KEEP", grid_keep),
            vision=_envbool("TYCHO_VISION", default=True),  # default "1"
            text_grid=text_grid,
            grid_embeds=grid_embeds_enabled(),
            tool_flavor=os.environ.get("TYCHO_TOOL_FLAVOR", "") or "",
            mode=mode,
            edit_func=_envbool("TYCHO_EDIT_FUNC"),
            meta_reflect=_envbool("TYCHO_META_REFLECT"),
            agent_graph_json=os.environ.get("TYCHO_AGENT_GRAPH", ""),
            log_io=os.environ.get("LLM_LOG") == "1",
            animation_summary=_envbool("TYCHO_ANIMATION_SUMMARY", default=True),
            animation_summary_video=_envbool("TYCHO_ANIMATION_SUMMARY_VIDEO", default=True),
            animation_summary_video_format=os.environ.get("TYCHO_ANIMATION_SUMMARY_VIDEO_FORMAT", "images").lower(),
            animation_summary_max_per_level=_envint("TYCHO_ANIMATION_SUMMARY_MAX_PER_LEVEL", 5),
            animation_evidence_keep_per_level=_envint("TYCHO_ANIMATION_EVIDENCE_KEEP_PER_LEVEL", 50),
            animation_summary_max_keyframes=_envint("TYCHO_ANIMATION_SUMMARY_MAX_KEYFRAMES", 6),
            animation_summary_include_game_over=_envbool("TYCHO_ANIMATION_SUMMARY_INCLUDE_GAME_OVER", default=True),
            animation_summary_max_tokens=_envint("TYCHO_ANIMATION_SUMMARY_MAX_TOKENS", 700),
            batch_actions=_envbool("TYCHO_BATCH_ACTIONS"),
            batch_max=_envint("TYCHO_BATCH_MAX", 24),
            batch_noop_guard_from_level=_envint("TYCHO_BATCH_NOOP_GUARD_FROM_LEVEL", 1),
            batch_border=_envint("TYCHO_BATCH_BORDER", 4),
            strict_receipts=_envbool("TYCHO_STRICT_RECEIPTS"),
            scribe_max_calls=_envint("TYCHO_SCRIBE_MAX_CALLS", 3),
            scribe_max_tokens=_envint("TYCHO_SCRIBE_MAX_TOKENS", 0),
            builder_gate=os.environ.get("TYCHO_BUILDER_GATE", "").lower(),
            builder_stuck_tokens=_envint("TYCHO_BUILDER_STUCK_TOKENS", 40000),
            builder_stuck_deaths=_envint("TYCHO_BUILDER_STUCK_DEATHS", 2),
            builder_max_per_level=_envint("TYCHO_BUILDER_MAX_PER_LEVEL", 0),
            builder_max_tokens=_envint("TYCHO_BUILDER_MAX_TOKENS", 0),
            wm_status_line=_envbool("TYCHO_WM_STATUS_LINE"),
            plan_batch_hint=_envbool("TYCHO_PLAN_BATCH_HINT"),
        )
        s._validate()
        return s

    def _validate(self) -> None:
        for f in ("max_tool_steps", "max_calls", "max_tokens", "grid_keep", "reasoning_keep"):
            if getattr(self, f) < 0:
                raise ValueError(f"TychoSettings.{f} must be >= 0, got {getattr(self, f)}")
        for f in ("max_inference_cost_per_game", "max_inference_cost_per_level"):
            if getattr(self, f) < 0:
                raise ValueError(f"TychoSettings.{f} must be >= 0, got {getattr(self, f)}")
        if self.builder_gate not in ("", "stuck"):
            raise ValueError(f"TYCHO_BUILDER_GATE={self.builder_gate!r} not in '' | stuck")
        if self.max_tool_steps < 1:
            raise ValueError(f"max_tool_steps must be >= 1, got {self.max_tool_steps}")
        if self.animation_summary_max_per_level < 0:
            raise ValueError("animation_summary_max_per_level must be >= 0")
        if self.animation_evidence_keep_per_level < 0:
            raise ValueError("animation_evidence_keep_per_level must be >= 0")
        if self.animation_summary_max_keyframes < 2:
            raise ValueError("animation_summary_max_keyframes must be >= 2")
        if self.animation_summary_max_tokens < 1:
            raise ValueError("animation_summary_max_tokens must be >= 1")
        if self.batch_max < 1:
            raise ValueError("batch_max must be >= 1")
        if self.batch_noop_guard_from_level < 0 or self.batch_border < 0:
            raise ValueError("batch_noop_guard_from_level and batch_border must be >= 0")
        if self.animation_summary_video_format not in ("images", "jpeg_frames", "gif"):
            raise ValueError("animation_summary_video_format must be images|jpeg_frames|gif")

    def resolved(self) -> dict:
        return asdict(self)
