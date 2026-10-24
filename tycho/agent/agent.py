"""Tycho — Freeform agent: an unconstrained file-system agent harness.

The A5 redesign taken to its conclusion (docs/a5-scientist-archetype.md motivated it;
Tycho is the tool-native realization). Instead of reassembling a structured prompt and
parsing a tool choice each turn, Tycho runs ONE GROWING CONVERSATION per game in which the
model freely:
  - reads the workspace (frames as hex + PNG, exact diffs, scene graphs) via base tools
    it's trained on (ls / read_file / run_python),
  - writes/revises beliefs to notes/actor_beliefs.md (notes/world_model.md in single mode);
    this prose IS its captured reasoning — which matters because Opus 4.8 doesn't expose
    reasoning summaries from providers that expose them, and in orch/trigger mode it is also the actor-to-builder channel,
  - commits exactly ONE scored world action via take_action, which ends the turn.
No experiment/explore/plan scaffolding — the agent decides its own scientific flow.

Mechanics:
  - Native tool use through a provider-neutral `chat_tools` protocol.
  - The current frame's PNG is attached in-context each turn; older frames stay on disk,
    retrievable by tool. Hex grids older than TYCHO_GRID_KEEP turns are dropped from context
    (kept on disk + as diffs) so the 64x64 grids don't bloat the conversation.
  - One conversation spans the whole game: take_action's RESULT is the next frame, which
    the harness feeds in on the following choose_action — a natural append flow.

Env: LLM_BACKEND/LLM_MODEL, TYCHO_EFFORT (defaults LLM_EFFORT), TYCHO_MAX_TOOL_STEPS (cognitive
tool calls per turn before a forced action; default 25), TYCHO_MAX_LLM_CALLS (whole-game
ceiling; default 1100 — on hit the agent STOPS cleanly via is_done, it does NOT autopilot),
TYCHO_MAX_INFERENCE_COST_PER_GAME / _PER_LEVEL (frozen public-list USD equivalent; 0 disables),
TYCHO_GRID_KEEP (turns of grid/image kept in context; default 3),
TYCHO_VISION (attach current-frame PNG; default on), TYCHO_MAX_TOKENS (default 16000),
TYCHO_TOOL_FLAVOR (tool naming: claude|gemini|generic; default auto from the model).
"""

from __future__ import annotations

import copy
import hashlib
import os
import shlex

from arcengine import GameAction

from tycho.harness.animation_evidence import (
    AnimationEvidenceConfig,
    analyze_frame_sequence,
    animation_signature,
    contact_sheet_png_bytes,
    selected_frame_gif_bytes,
    selected_frame_jpeg_bytes,
    selected_frame_png_bytes,
)
from tycho.harness.agent import ActionChoice, Agent
from tycho.harness.actions import normalize_game_actions
from tycho.workspace.workspace import GameWorkspace, diff_text, grid_text
from tycho.workspace.agent_tools import tool_specs, ToolExecutor, ACTION_HINT, canonical
from tycho.serving.llm_client import (
    LLMConfig,
    chat_tools,
    is_context_limit_error,
    response_id_continuity_enabled,
)
from tycho.serving.pricing import PRICE_SCHEDULE, price_for, reply_usage, usage_cost_usd

from tycho.agent.colors import COLORMAP as _COLORMAP  # shared with the builder (single source)
from tycho.agent.modes import resolve_mode  # orchestration MODES table (single source of truth)
from tycho.agent.events import Event, AgentSpec, AgentKind, on_events, on_agent_done  # event-driven agent graph
from tycho.agent.context_config import resolve_context_config  # caching/history config (single source)
from tycho.agent.vision import vision_profile  # per-model render scale + image-fidelity facts (single source)
from tycho.config import TychoSettings  # typed, validated agent-core config (single source; replaces inline env reads)
from tycho.prompts.render import render_prompt  # Jinja prompt assembly (all conditionals live in the .j2)

# The actor system prompt (actor.system.j2) selects the world-model paragraph variant (single/orch/
# trigger, via {% include %} of partials/wm_*.j2) and the meta-reflect addendum itself — no Python
# lookup table here; agent.__init__ passes wm_variant + meta_reflect as render context.
# (The perception-sentence conditionals that used to live here as _perception_sys + anchor constants
# are now {% if show_grid %}/{% if lossless_cells %} blocks in actor.system.j2 — visible where edited.)


def _flavor_for(model: str) -> str:
    m = (model or "").lower()
    if "claude" in m or "opus" in m or "sonnet" in m or "fable" in m:
        return "claude"
    if "gemma" in m or "gemini" in m:
        return "gemini"
    # Qwen3.x has no dedicated file-tool vocabulary like Claude-Code/Gemini-CLI; the generic
    # names (read_file/write_file/edit_file/ls/run_python) are exactly what it was trained on.
    # The Qwen-specific piece is the SERVER tool-call parser (--tool-call-parser qwen3_xml),
    # not the tool names — so qwen routes to generic (explicit, not a fall-through accident).
    return "generic"


# Recorded tool-result cap. A full 64x64 grid plus short analysis fits; head-preserving clipping keeps
# row 0 visible. The model-facing result uses a separate cap in `agent_tools._clip_ends`.
_TRACE_RESULT_CAP = 8000
_ELIDE_FRAME_TEXT_TRIGGER = 1500
_ELIDE_TOOL_TEXT_TRIGGER = 200


def _clip_trace(s: str, cap: int = _TRACE_RESULT_CAP) -> str:
    """Head-preserving clip for the recorded trace (not the model-facing output, which is clipped in
    agent_tools._clip_ends). Keeps the start so a grid print shows row 0; appends an explicit
    elision marker so truncation is visible, never silent."""
    s = str(s)
    return s if len(s) <= cap else s[:cap] + f"… [+{len(s) - cap} chars; read the file for the rest]"


class TychoAgent(Agent):
    name = "tycho"
    _ctx_logged = False   # one-shot guard: log the resolved context config once per process, not per game

    def __init__(self, tools=None):
        self.cfg = LLMConfig.from_env()
        # Called with the latest frame right before a turn that may call the LLM (never mid-sequence);
        # an embedding runner may block in it, e.g. to yield a shared inference slot. None: no hook.
        self.before_decision = None
        # Reasoning. Default to "medium" when unset — ARC-AGI-3 is unsolvable without
        # reasoning, and an EMPTY effort silently disables thinking on ALL backends.
        # The 2026-06-17 Gemma run scored 0.28 with effort="" → thinking OFF for all 161
        # calls; not a fair capability test. Set TYCHO_EFFORT="off" to deliberately disable.
        # NOTE on Qwen3.x (vLLM): thinking is a BINARY enable_thinking toggle — there are NO
        # provider-specific low/medium/high tiers. Any truthy TYCHO_EFFORT turns thinking on;
        # the value is ignored on the OpenAI/vLLM path (only truthiness matters). So the Qwen
        # reasoning ablation is on-vs-off, not a tier sweep.
        # Agent-core config resolved + VALIDATED once through the typed TychoSettings (single source;
        # replaces ~15 scattered os.environ.get() reads, each of which had its own default/parse — a
        # DRY hazard). Read self.settings.<field> below. Backend knobs stay in llm_client; context/
        # vision/mode keep their own typed resolvers, composed further down.
        self.settings = TychoSettings.from_env()
        _eff = self.settings.effort
        # "off"/none/0/"" → disable thinking (the agent's policy on top of the raw resolution). Kept
        # here (not in Settings) because it's the agent's interpretation of effort, not a parse.
        self.effort = "" if _eff.lower() in ("", "off", "none", "0") else _eff
        self.max_tool_steps = self.settings.max_tool_steps
        self.max_calls = self.settings.max_calls
        self.max_inference_cost_per_game = self.settings.max_inference_cost_per_game
        self.max_inference_cost_per_level = self.settings.max_inference_cost_per_level
        self._inference_price = price_for(self.cfg.model)
        if (self.max_inference_cost_per_game > 0 or self.max_inference_cost_per_level > 0) and (
            self._inference_price is None or not any(self._inference_price)
        ):
            raise ValueError(
                "A dollar-denominated inference budget requires a positive frozen public price "
                f"for LLM_MODEL={self.cfg.model!r}; no such rate is registered."
            )
        self.grid_keep = self.settings.grid_keep
        # raw_reasoning (Opus extended-thinking) eviction: keep it only on the most recent N
        # assistant turns (the 1M-context-overflow fix; default = grid_keep, resolved in Settings).
        self.reasoning_keep = self.settings.reasoning_keep
        self.vision = self.settings.vision
        # Base output budget. OpenAI Responses adds reasoning headroom; local Qwen reasons within
        # this budget and decodes at ~57 tok/s, so it remains the main wall-clock lever there.
        self.max_tokens = self.settings.max_tokens
        self.retry_max_tokens = self.settings.retry_max_tokens
        self.strict_receipts = self.settings.strict_receipts
        self.scribe_max_calls = self.settings.scribe_max_calls
        self.scribe_max_tokens = self.settings.scribe_max_tokens
        self.builder_gate = self.settings.builder_gate
        self.builder_stuck_tokens = self.settings.builder_stuck_tokens
        self.builder_stuck_deaths = self.settings.builder_stuck_deaths
        self.builder_max_per_level = self.settings.builder_max_per_level
        self.wm_status_line = self.settings.wm_status_line
        self.plan_batch_hint = self.settings.plan_batch_hint
        self.level_start_inline = self.settings.level_start_inline
        self.drain_images = self.settings.drain_images
        self.scribe_mode = self.settings.scribe_mode
        self.level_start_probe = self.settings.level_start_probe
        # H017 perception ablations — TYCHO_TEXT_GRID (full|diff|none) sets the STARTING grid/diff
        # toggles (show_grid/show_diff are derived properties on Settings); the agent can flip them
        # mid-game via set_verbosity. TYCHO_GRID_EMBEDS attaches the exact per-cell prompt_embeds.
        self.show_grid = self.settings.show_grid
        self.show_diff = self.settings.show_diff
        self.grid_embeds = self.settings.grid_embeds
        # PERCEPTION GUARD: the agent must be able to SEE the board through at least one channel —
        # the inline grid text, the rendered image (vision), or the exact-grid embeds. With vision
        # OFF and TYCHO_TEXT_GRID=diff/none and no embeds, the prompt would describe an image/grid
        # that isn't there (worst on turn 0, which has no prior diff either) and the agent would be
        # blind. Force the full grid text on in that case — perception is non-negotiable; the text
        # grid is the always-available fallback. Log it so the silent override is visible.
        if not self.vision and not self.show_grid and not self.grid_embeds:
            import sys as _sys
            print(f"[tycho] vision off + TYCHO_TEXT_GRID={self.settings.text_grid} + no grid_embeds would leave NO board "
                  f"channel — forcing the full grid text ON (perception is required).", file=_sys.stderr)
            self.show_grid = True
            self.show_diff = True
        self.diff_mode = "on" if self.show_diff else "off"
        flavor = self.settings.tool_flavor or _flavor_for(self.cfg.model)
        self._flavor = flavor
        # Resolve the four paper policies through one declarative table. Prompt selection, model
        # ownership, actor-invoked delegation, and harness-triggered repair remain independent axes.
        self.mode = self.settings.mode
        self._mode_spec = resolve_mode(self.mode)
        # A config file may declare a multi-agent topology
        # (orchestration.agents) that env vars can't express. run_parallel passes the validated block
        # as JSON in TYCHO_AGENT_GRAPH; we parse it into AgentDecls here. None → fall back to the
        # MODES preset. A declared builder agent implies the dispatcher
        # even if the preset mode wouldn't — so a custom graph drives the build wiring, not TYCHO_MODE.
        self._agent_graph_decls = None
        if self.settings.agent_graph_json:
            import json as _json
            from tycho.agent.modes import resolve_agent_graph
            self._agent_graph_decls = resolve_agent_graph(_json.loads(self.settings.agent_graph_json))
        graph_has_builder_decl = bool(self._agent_graph_decls) and any(
            d.type == "builder" for d in self._agent_graph_decls)
        if graph_has_builder_decl and self._mode_spec.wm_variant == "none":
            raise ValueError(
                "TYCHO_AGENT_GRAPH declares a builder, but TYCHO_MODE='no_world_model' disables the "
                "executable world-model loop. Use TYCHO_MODE=trigger/orchestrator or remove the builder graph.")
        if graph_has_builder_decl and self._mode_spec.include_builder:
            raise ValueError(
                "TYCHO_AGENT_GRAPH declares a harness-fired builder, but TYCHO_MODE is "
                f"{self._mode_spec.name!r} (actor-pull invoke_builder). Use TYCHO_MODE=trigger "
                "or remove the builder graph; the trigger+subagent hybrid is not wired.")
        # A declared graph with a BUILDER drives the build wiring even when TYCHO_MODE wasn't set to a
        # builder mode (a config may declare `orchestration.agents` with a builder but omit the scalar
        # `mode:`, which is all that flattens to TYCHO_MODE). Without this, the actor would keep the
        # single-mode prompt ("you write world_model.py inline") while the graph builder ALSO owns the
        # model — both think they own it. So when the graph adds a builder the preset didn't, switch
        # the actor's prompt/tooling to the harness-fired `trigger` semantics (builder owns the model;
        # actor consumes its report; no invoke tool). resolve_mode('trigger') is the canonical spec.
        if graph_has_builder_decl and not self._mode_spec.needs_builder:
            from tycho.agent.modes import resolve_mode as _resolve_mode
            self._mode_spec = _resolve_mode("trigger")
        # Keep the public agent mode label aligned with the canonical spec after graph-driven
        # normalization. A builder graph without an explicit scalar mode behaves as trigger, so
        # diagnostics/manifests should not continue to call the agent "single".
        self.mode = self._mode_spec.name
        # Kept as a derived convenience: many call sites still ask "is a builder in play?". It now
        # tracks include_builder (true for orchestrator; false for single AND for the future trigger
        # mode where the harness, not the actor's tool, fires the builder).
        self._orchestrator = self._mode_spec.include_builder
        # TYCHO_EDIT_FUNC=1 adds the function-granular edit_function tool to world-model actors.
        # Builders always receive it: their prompt asks for function-level refinement, while their
        # narrow tool set omits actor-only display controls such as set_verbosity.
        self._edit_func = self.settings.edit_func
        wm_prompt_enabled = self._mode_spec.wm_variant != "none"
        actor_edit_func = self._edit_func and wm_prompt_enabled
        # Multi-action commits: take_action may carry an `actions` sequence that the agent plays out
        # without further LLM calls until a guard stops it (_batch_guard). 0 = upstream behaviour.
        self.batch_max = self.settings.batch_max if self.settings.batch_actions else 0
        self.batch_noop_guard_from_level = self.settings.batch_noop_guard_from_level
        self.batch_border = self.settings.batch_border
        # TYCHO_PY_ACTIONS: run_python may call action(...) and keep playing across steps (a live script
        # process; host sandbox only, where the script is a child of this process).
        self.py_actions = self.settings.py_actions and os.environ.get("TYCHO_SANDBOX_RUNTIME", "") == "host"
        if self.settings.py_actions and not self.py_actions:
            print("[tycho] TYCHO_PY_ACTIONS needs TYCHO_SANDBOX_RUNTIME=host; acting scripts are off", flush=True)
        self.py_actions_max = self.settings.py_actions_max
        self.object_diff = self.settings.object_diff
        self.stuck_critic = self.settings.stuck_critic
        self._tools_spec = tools or tool_specs(flavor, include_builder=self._orchestrator,
                                               include_edit_func=actor_edit_func,
                                               world_model_enabled=wm_prompt_enabled,
                                               batch_max=self.batch_max,
                                               py_helpers=self.settings.py_helpers,
                                               py_actions_max=self.py_actions_max if self.py_actions else 0,
                                               py_retain=self.settings.py_retain,
                                               object_diff=self.settings.object_diff)
        # PER-MODEL vision profile (single source of truth, vision.py): render scale + whether THIS
        # model's encoder sees the image losslessly per cell. Drives BOTH the PNG render scale and
        # the perception sentence — decoupled (lossless ⊥ text-inlined). lossless_cells is asserted
        # in the prompt only when vision is ON (no image → no fidelity claim).
        self._vision_profile = vision_profile(self.cfg.model)
        _img_lossless = self.vision and self._vision_profile.lossless_cells
        # Assemble the actor system prompt from actor.system.j2 — the wm-variant selection, perception
        # sentences, and meta-reflect addendum are all VISIBLE {% if %}/{% include %} in the template;
        # we pass the booleans/variant, not assembled text.
        self._sys = render_prompt(
            "actor.system", colormap=_COLORMAP, wm_variant=self._mode_spec.wm_variant,
            vision=self.vision, show_grid=self.show_grid, show_diff=self.show_diff,
            lossless_cells=_img_lossless, meta_reflect=self.settings.meta_reflect,
            batch_actions=self.batch_max > 0, batch_max=self.batch_max,
            budget_prompt=self.settings.budget_prompt, py_actions=self.py_actions,
            goal_hypotheses=self.settings.goal_hypotheses)
        self.animation_summary_enabled = self.settings.animation_summary and self.vision
        self.animation_summary_video_enabled = (
            self.settings.animation_summary_video and _supports_animation_video(self.cfg)
        )
        self.animation_summary_video_format = self.settings.animation_summary_video_format
        self.animation_summary_include_game_over = self.settings.animation_summary_include_game_over
        self.animation_summary_max_per_level = self.settings.animation_summary_max_per_level
        self.animation_evidence_keep_per_level = self.settings.animation_evidence_keep_per_level
        self.animation_summary_max_tokens = self.settings.animation_summary_max_tokens
        self._animation_evidence_config = AnimationEvidenceConfig(
            max_keyframes=self.settings.animation_summary_max_keyframes)
        self._animation_summary_scale = max(4, min(8, self._vision_profile.render_scale))
        self._animation_summary_video_scale = self._vision_profile.render_scale
        # Context-management + prompt-caching config — resolved through ONE source of truth
        # (tycho/agent/context_config.py) with orthogonal, self-describing switches and CACHE-FRIENDLY
        # DEFAULTS (prompt_caching ON, history=tail_evict). This replaces the old conflated pair
        # (TYCHO_CONTEXT_MODE=legacy + LLM_PROMPT_CACHE=off, both default-off) that silently produced
        # no-cache runs. Old env names still resolve (back-compat); the resolved config is logged at
        # startup + stamped in the manifest, so a silent revert is immediately visible.
        self._ctx_cfg = resolve_context_config(grid_keep_default=self.grid_keep)
        # history structure: tail_evict (append-only, cache-friendly) | truncate (mutate old turns,
        # cache-hostile) | collapse (deprecated). Kept on self.context_mode for the consumers below.
        self.context_mode = self._ctx_cfg.history
        self.cache_image_keep = self._ctx_cfg.image_keep
        self.cache_image_evict_mult = self._ctx_cfg.image_evict_mult
        self.cache_text_keep = self._ctx_cfg.text_keep
        self.evict_each_llm_call = self._ctx_cfg.evict_each_llm_call
        self.context_emergency_compaction = self._ctx_cfg.emergency_compaction
        self.context_emergency_soft_tokens = self._ctx_cfg.emergency_soft_tokens
        self.context_emergency_keep_turns = self._ctx_cfg.emergency_keep_turns
        self.context_emergency_recent_actions = self._ctx_cfg.emergency_recent_actions
        self.context_emergency_tool_results = self._ctx_cfg.emergency_tool_results
        self.context_drain_budget = self._ctx_cfg.drain_budget_tokens
        self.context_drain_tokens = self._ctx_cfg.drain_tokens
        # Hard ceiling on images per request — MUST be <= the server's --limit-mm-per-prompt image,
        # or vLLM 400s. Batched eviction is gated on a threshold that can exceed this cap, so the cap
        # is a non-negotiable backstop after the batched pass. Default 4 matches common servers.
        self.cache_image_cap = self._ctx_cfg.image_cap
        # ANTI-SILENT-REVERT: log the fully-resolved context config ONCE (the agent is built per game;
        # a class flag prevents 25× spam) so a no-cache/wrong-history run is visible in the log on line
        # one — the guardrail that would have caught the $6K no-cache H025 surprise. Also stamped into
        # the manifest (run_parallel) so the config is recoverable from the result, not just the log.
        if not TychoAgent._ctx_logged:
            import sys as _sys
            print(f"[tycho ctx config] {self._ctx_cfg.resolved()}", file=_sys.stderr)
            for _w in self._ctx_cfg.warnings():
                print(f"[tycho ctx config] ⚠ {_w}", file=_sys.stderr)
            _vp = self._vision_profile
            print(f"[tycho vision] model={self.cfg.model} render_scale={_vp.render_scale} "
                  f"lossless_cells={_vp.lossless_cells} ({_vp.note})", file=_sys.stderr)
            TychoAgent._ctx_logged = True

    def reset(self, game_id, available_actions, ws_root=None, resume=False):
        # ws_root: durable workspace base (results/<run>/ws); default = ephemeral mkdtemp.
        # resume: keep the agent-authored world_model.py / notes already on disk (don't re-seed).
        wm_enabled = self._mode_spec.wm_variant != "none"
        self.ws = GameWorkspace(game_id, root=ws_root, render=True,
                                available_actions=available_actions, resume=resume,
                                render_scale=self._vision_profile.render_scale,
                                seed_world_model=wm_enabled)
        self.exec = ToolExecutor(self.ws, wm_feedback_enabled=wm_enabled, py_helpers=self.settings.py_helpers,
                                 py_retain=self.settings.py_retain)
        self.history: list[dict] = []     # the one growing conversation
        self.calls = 0
        self.token_stats = {"tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cache_write": 0,
                            "latency_ms": 0, "length_stops": 0, "max_tokens_out": 0,
                            "context_compactions": 0, "reasoning_chain_rollovers": 0,
                            "max_prompt_tokens": 0}
        self._budget_level = 0
        self._budget_level_start_usd = 0.0
        self._done_reason: str | None = None
        self._last_prompt_tokens = 0
        self._last_context_compaction_at: tuple[int, int] | None = None
        self._last_context_compaction_history_len = -1
        self._context_compaction_streak = 0
        self._reasoning_chain_rollover_pending = False
        # A declared graph may include a harness-fired builder regardless of the preset mode — detect
        # it so the builder object + dispatcher get constructed to match the graph, not just TYCHO_MODE.
        self._graph_has_builder = bool(self._agent_graph_decls) and any(
            d.type == "builder" for d in self._agent_graph_decls)
        self.builder = None
        # A builder OBJECT is constructed whether the actor PULLS it (orchestrator) or the HARNESS
        # fires it (trigger / a declared-graph builder) — all need the same subagent.
        if self._mode_spec.needs_builder or self._graph_has_builder:
            from tycho.agent.builder import WorldModelBuilder
            # builder shares the workspace/executor; its own lean conversation. Its tool set
            # excludes invoke_builder + take_action (it neither recurses nor acts).
            self.builder = WorldModelBuilder(
                self.cfg, self.exec, tool_specs(self._flavor, include_builder=False,
                                                include_edit_func=True,
                                                include_take_action=False,
                                                include_set_verbosity=False,
                                                py_helpers=self.settings.py_helpers,
                                                py_retain=self.settings.py_retain,
                                                object_diff=self.settings.object_diff),
                effort=self.effort, max_tokens=self.settings.builder_max_tokens or self.max_tokens)
        # trigger mode: the harness-side scheduler that fires the builder on (action committed ∧
        # world model mispredicts), rate-limited. None in single/orchestrator (additive — those
        # modes never consult it).
        self.dispatcher = None
        if self._mode_spec.harness_fires_builder or self._graph_has_builder:
            from tycho.agent.dispatcher import TriggerDispatcher
            self.dispatcher = TriggerDispatcher()
        self.builder_invocations = 0
        self._turn_builder_runs: list = []   # builder runs recorded in the current turn
        self._reset_level_counters()
        self._build_agent_graph()
        self.t = 0                        # global trace turn
        self.level = 0                    # current level
        self.turn_in_level = 0            # turn counter WITHIN the level (#19)
        self.prev_levels_done = 0         # to detect a level completion
        self._last_committed_control: str | None = None
        self._pending_game_over_event: dict | None = None
        self._pending_frame_boundary: str | None = "game_start"
        self._reset_animation_summary_level_state()
        self._turn_animation_summaries: list = []
        self._last_tool_trace: list = []
        self._batch: dict | None = None            # the action sequence being played out, if any
        self._prepared_batch: dict | None = None   # a sequence validated this turn, not yet committed
        self._batch_report: dict | None = None     # how the last sequence ended; shown next turn
        # Proprietary GPT reasoning is hidden, but Responses-backed transports can retain it behind
        # an opaque response ID. The chain is reset at level/reset boundaries and by context rescue.
        # Stored opaque reasoning can improve continuity, but its server-side context cannot be
        # edited by client retention. The generic capability hook keeps that tradeoff in the
        # transport rather than hard-coding provider names into the agent.
        self._reasoning_continuity = response_id_continuity_enabled(self.cfg)
        self._prev_response_id = None
        self._chain_sent_upto = 0   # # of history items the SERVER already holds (delta watermark)
        self._log_io = self.settings.log_io
        if self._log_io:
            from tycho.serving.llm_client import start_recording, set_system_prompt
            start_recording()
            set_system_prompt(self._sys)       # capture the system prompt in call records

    # ---- resume support (see tycho/harness/resume.py) ----
    # We snapshot ONLY mutable per-game state; the prompt (_sys), config, tool specs, executor and
    # builder are reconstructed by reset() (they hold no recoverable cross-turn state). The actor's
    # `history` IS the conversation and round-trips verbatim — bit-exact in legacy/cache mode (the
    # only modes used; rolling is deprecated). The journal checkpoints the complete workspace in
    # parallel; this actor snapshot carries only in-memory agent state.
    def snapshot_state(self) -> dict:
        return {
            "history": self.history,
            "calls": self.calls, "t": self.t, "level": self.level,
            "turn_in_level": self.turn_in_level, "prev_levels_done": self.prev_levels_done,
            "builder_invocations": self.builder_invocations,
            "token_stats": self.token_stats,
            "budget_level": self._budget_level,
            "budget_level_start_usd": self._budget_level_start_usd,
            "done_reason": self._done_reason,
            "last_action": getattr(self, "_last_action", None),
            "last_committed_control": getattr(self, "_last_committed_control", None),
            "last_row": getattr(self, "_last_row", None),
            "last_col": getattr(self, "_last_col", None),
            "pending_game_over_event": getattr(self, "_pending_game_over_event", None),
            "pending_frame_boundary": getattr(self, "_pending_frame_boundary", None),
            "animation_summary_cache": getattr(self, "_animation_summary_cache", {}),
            "animation_summary_count_this_level": getattr(self, "_animation_summary_count_this_level", 0),
            "animation_summary_last_signature": getattr(self, "_animation_summary_last_signature", None),
            "last_prompt_tokens": getattr(self, "_last_prompt_tokens", 0),
            "last_context_compaction_at": getattr(self, "_last_context_compaction_at", None),
            "last_context_compaction_history_len": getattr(
                self, "_last_context_compaction_history_len", -1
            ),
            "context_compaction_streak": getattr(self, "_context_compaction_streak", 0),
            "show_grid": self.show_grid,
            "show_diff": self.show_diff,
            "diff_mode": self.diff_mode,
            "ws_prev_grid": self.ws.prev_grid,
            "ws_turns": self.ws.turns,
            # trigger-mode dispatcher state (per-level fire counts for the kickoff message); absent in other modes.
            "dispatcher": self.dispatcher.snapshot() if self.dispatcher is not None else None,
        }

    def restore_state(self, s: dict) -> None:
        """Re-seat mutable state AFTER reset() has rebuilt ws/exec/builder against the durable dir."""
        self.history = s["history"]
        self.calls = s["calls"]; self.t = s["t"]; self.level = s["level"]
        self.turn_in_level = s["turn_in_level"]; self.prev_levels_done = s["prev_levels_done"]
        self.builder_invocations = s.get("builder_invocations", 0)
        self.token_stats = s.get("token_stats", self.token_stats)
        self._budget_level = int(s.get("budget_level", self.level))
        self._budget_level_start_usd = float(s.get("budget_level_start_usd", 0.0))
        self._done_reason = s.get("done_reason")
        self._last_action = s.get("last_action")
        self._last_committed_control = s.get("last_committed_control")
        self._last_row = s.get("last_row"); self._last_col = s.get("last_col")
        self._pending_game_over_event = s.get("pending_game_over_event")
        boundary = s.get("pending_frame_boundary")
        self._pending_frame_boundary = (
            boundary if boundary in ("game_start", "level_start", "attempt_restart") else None
        )
        self._animation_summary_cache = s.get("animation_summary_cache", {})
        self._animation_summary_count_this_level = s.get("animation_summary_count_this_level", 0)
        self._animation_summary_last_signature = s.get("animation_summary_last_signature")
        self._last_prompt_tokens = s.get("last_prompt_tokens", 0)
        self._last_context_compaction_at = s.get("last_context_compaction_at")
        self._last_context_compaction_history_len = int(
            s.get("last_context_compaction_history_len", -1)
        )
        self._context_compaction_streak = int(s.get("context_compaction_streak", 0))
        self.show_grid = bool(s.get("show_grid", self.show_grid))
        self.show_diff = bool(s.get("show_diff", self.show_diff))
        self.diff_mode = str(s.get("diff_mode", "on" if self.show_diff else "off"))
        if self.diff_mode not in ("on", "summary", "off"):
            self.diff_mode = "on" if self.show_diff else "off"
        self.show_diff = self.diff_mode != "off"
        self.ws.prev_grid = s.get("ws_prev_grid")
        self.ws.turns = s.get("ws_turns", [])
        if self.dispatcher is not None and s.get("dispatcher") is not None:
            self.dispatcher.restore(s["dispatcher"])
        # Response IDs are ephemeral server state — a resumed game can't reuse the prior chain.
        # Drop it; the next call re-establishes from the restored client history.
        self.reset_reasoning_chain()

    def is_done(self, frames, latest_frame) -> bool:
        """Stop cleanly at a call or inference-cost budget boundary.

        Cost limits are checked between environment actions. This preserves the actor's policy up
        to termination and bounds overshoot to the calls used to choose one action. A level advance
        resets the per-level meter before any next-level summary, builder, or actor call.

        Stop once the whole-game LLM-call budget is spent instead of committing an uninformed
        fallback action. The harness checks `is_done` before spending the next environment action."""
        cost = self.inference_cost_usd()
        levels_done = int(getattr(latest_frame, "levels_completed", self._budget_level) or 0)
        if levels_done > self._budget_level:
            self._budget_level = levels_done
            self._budget_level_start_usd = cost or 0.0

        if self.max_inference_cost_per_game > 0 and cost is not None \
                and cost >= self.max_inference_cost_per_game:
            self._done_reason = "inference_cost_game_limit"
            return True
        level_cost = None if cost is None else max(0.0, cost - self._budget_level_start_usd)
        if self.max_inference_cost_per_level > 0 and level_cost is not None \
                and level_cost >= self.max_inference_cost_per_level:
            self._done_reason = "inference_cost_level_limit"
            return True
        if self.calls >= self.max_calls:
            self._done_reason = "llm_call_limit"
            return True
        self._done_reason = None
        return False

    @property
    def done_reason(self) -> str | None:
        return self._done_reason

    def inference_cost_usd(self) -> float | None:
        return usage_cost_usd(self.token_stats, price=self._inference_price)

    def inference_budget_state(self) -> dict:
        cost = self.inference_cost_usd()
        level_cost = None if cost is None else max(0.0, cost - self._budget_level_start_usd)
        return {
            "price_schedule": PRICE_SCHEDULE,
            "model": self.cfg.model,
            "usage": {key: int(self.token_stats.get(key, 0)) for key in (
                "tokens_in", "tokens_out", "cache_read", "cache_write",
            )},
            "cost_usd": cost,
            "level_cost_usd": level_cost,
            "max_cost_per_game_usd": self.max_inference_cost_per_game,
            "max_cost_per_level_usd": self.max_inference_cost_per_level,
            "stop_reason": self._done_reason,
        }

    def choose_action(self, frames, latest_frame, available_actions) -> ActionChoice:
        grid = latest_frame.frame[-1]
        avail = [a.name for a in available_actions]
        game_avail = [a.name for a in normalize_game_actions(available_actions)]
        self._current_available_actions = list(avail)
        self._current_game_actions = list(game_avail)
        self._active_tools_spec = self._tools_with_available_actions(avail)
        state = str(latest_frame.state).replace("GameState.", "")
        levels_done = latest_frame.levels_completed

        # --- MULTI-ACTION SEQUENCE: play the next queued action without an LLM call while no guard
        # fires; otherwise close the sequence and fall through to a normal turn, which reports it.
        if getattr(self, "_batch", None) is not None:
            stop = self._batch_guard(latest_frame, avail, game_avail)
            if self._batch.get("script") is not None:
                # a run_python script that plays (TYCHO_PY_ACTIONS): its next action comes from the script
                if stop is None:
                    step, stop = self._script_next(latest_frame, avail, available_actions, state)
                    if step is not None:
                        self._batch["queue"] = [step]
                        self._batch["labels"].append(
                            self._action_label(step["action"], row=step.get("row"), col=step.get("col")))
                        return self._continue_batch(latest_frame, available_actions, avail, game_avail, state)
            elif stop is None and self._batch["queue"]:
                return self._continue_batch(latest_frame, available_actions, avail, game_avail, state)
            self._finish_batch(stop, grid)

        if self.before_decision is not None:
            self.before_decision(latest_frame)

        # --- LEVEL TRANSITION: levels_completed rose since last turn → a level was just solved.
        # First consolidate the just-finished level while the old-level transcript is still present;
        # then reset history and present the next level as a fresh decision problem. ---
        level_just_completed = levels_done > self.prev_levels_done
        if level_just_completed:
            self._dispatch(Event.LEVEL_COMPLETE)  # boundary consolidation for the just-completed level
            self._reset_history_for_new_level()   # THEN clear the transcript: each level is faced fresh,
                                                  # carry-forward lives on disk (notes/ + world_model.py),
                                                  # which the fresh-level frame message points the agent to.
            self.level += 1
            self.turn_in_level = 0
            self._reset_animation_summary_level_state()
            self._reset_level_counters()
            self.ws.new_level()               # fresh grid; no cross-level diff
            self.reset_reasoning_chain()      # SERVER chain re-establishes from the (now-empty) history
            self._pending_frame_boundary = "level_start"
        self.prev_levels_done = levels_done

        # action is what PRODUCED this frame; the very first frame of a level had no action
        # (it's the initial/reset state) → record None, not a misleading "RESET" (#6).
        self.ws.record(grid, level=self.level, turn_in_level=self.turn_in_level,
                       action=getattr(self, "_last_action", None) if self.turn_in_level > 0 else None,
                       row=getattr(self, "_last_row", None), col=getattr(self, "_last_col", None),
                       state=state, available=avail)
        self.exec.cur_grid = [list(map(int, row)) for row in grid]  # back-compat; unused since zoom removed
        self.exec.available = game_avail  # world-model auto-feedback expands only frame-declared game actions

        # Feed the new frame as a FRESH USER MESSAGE: the grid INLINE (space-separated,
        # #9/#16) + the within-level diff, plus the rendered PNG. The prior tool-use turn
        # was already closed with its toolResult, so this user message is well-formed.
        # ROLLING mode: collapse the PRIOR turn (its grid/image + tool transcript) down to just
        # the committed action BEFORE adding the new frame, so the persistent prefix is
        # [system, a1..a_{N-1}] — byte-stable across turns → prefix-cache hit; only the new grid
        # below is uncached. (No-op on the first turn / in truncate mode.)
        if self.context_mode == "collapse":
            self._collapse_prev_turn()

        self._turn_animation_summaries = []
        had_game_over_event = bool(getattr(self, "_pending_game_over_event", None))
        content = self._game_over_evidence_parts()
        if not had_game_over_event:
            content.extend(self._animation_summary_parts(
                latest_frame.frame,
                action=getattr(self, "_last_action", None),
                row=getattr(self, "_last_row", None),
                col=getattr(self, "_last_col", None),
                terminal="nonterminal",
                level_just_completed=level_just_completed,
            ))
        frame_boundary = getattr(self, "_pending_frame_boundary", None)
        batch_report = getattr(self, "_batch_report", None)
        self._batch_report = None
        if batch_report is not None:
            content.append({"text": batch_report["text"]})
        content.append({"text": self._frame_message(grid, state, avail, frame_boundary)})
        kept = self.exec.retained_signatures() if getattr(self.exec, "py_retain", False) else []
        if kept:
            content.append({"text": "Functions kept from your earlier run_python calls (call them directly; define one "
                                    "again to change it): " + ", ".join(kept[-25:])})
        self._pending_frame_boundary = None
        png = self.ws.current_png(self.level, self.turn_in_level) if self.vision else None
        if png is not None:
            content.append({"image_png": png})
        # H017 exact-grid channel: attach the merge-proof per-cell embeds (prompt_embeds).
        # Only the latest frame carries it (it's a ~57MB payload; never replay older ones —
        # _truncate_old_grids strips it alongside images). Silently skips if unbuildable here.
        if self.grid_embeds:
            from tycho.workspace.optional_channels import grid_embeds_part
            part = grid_embeds_part(grid)
            if part is not None:
                content.append({"grid_embeds_part": part})
        self.history.append({"role": "user", "content": content})

        # truncate edits OLD turns in place (cache-hostile); tail_evict only mutates old heavy
        # payloads when a threshold is crossed (cache-friendly); collapse already collapsed the
        # prior turn above, so it has nothing per-turn to trim.
        if self.context_mode == "tail_evict":
            self._cache_mode_evict_if_needed()
        elif self.context_mode not in ("collapse", "drain"):  # drain trims only in _drain_history
            self._truncate_old_grids()

        _token_stats_before = dict(self.token_stats)
        self._turn_builder_runs = []   # builder activity THIS turn (trigger fire at turn START +
                                       # any actor-pull during _run_until_action) — drained below
        # TRIGGER-MODE BUILDER — fires at the START of the turn: AFTER the new frame is recorded +
        # shown (so its inaccuracy probe sees the transition the just-arrived frame completes = the
        # PRIOR committed action's outcome), but BEFORE the actor reasons, so the actor decides THIS
        # turn's action on the freshly-corrected model. Guarded turn_in_level>0: turn 0 has no
        # within-level transition to learn from. No rate limit — the inaccuracy probe IS the gate
        # (correct model -> ACCURATE -> no fire). Additive: single/orchestrator have no dispatcher.
        if self.dispatcher is not None and had_game_over_event and self.calls < self.max_calls:
            self._dispatch(Event.GAME_OVER_RESET, {"level": self.level, "game_over_reset": True})
        elif self.dispatcher is not None and self.turn_in_level > 0:
            fire = self.calls < self.max_calls and self.dispatcher.should_fire(self.level, self.exec)
            self._dispatch(Event.ACTION_SUBMITTED, {"level": self.level, "wm_inaccuracy_fire": fire})
        # LEVEL_START builder fire: a NEW level's first frame is now recorded + shown. Fire the
        # builder so it can check the new init frame against the model BEFORE the actor acts — a
        # new level can introduce colors/layout/decode rules the prior-level model can't anticipate,
        # and the mispredict gate can't see that until an action lands (1 wasted turn). It reads
        # notes/level_<L-1>_insights.md (boundary consolidation). Skipped at game start
        # (level 0 turn 0). Placed AFTER the frame append + _turn_builder_runs reset so the report
        # is recorded in the trace and lands in history AFTER the new-level frame.
        elif self.dispatcher is not None and self.turn_in_level == 0 and self.level > 0 \
                and self.calls < self.max_calls:
            self._dispatch(Event.LEVEL_START, {"level": self.level, "at_level_start": True})
        if self._stuck_builder_due():
            self._run_stuck_builder()
        if self._stuck_critic_due():
            self._run_stuck_critic(grid, game_avail)
        self._prepared_batch = None
        self._prepared_script = None
        choice, trace = self._run_until_action(available_actions)
        prepared, self._prepared_batch = self._prepared_batch, None
        script, self._prepared_script = self._prepared_script, None
        if script is not None:
            self._start_script_batch(script[0], script[1], grid, game_avail)
        elif prepared is not None and choice is prepared["first"]:
            self._start_batch(prepared, grid, game_avail)
        self._last_tool_trace = trace
        token_delta = {
            k: int(self.token_stats.get(k, 0)) - int(_token_stats_before.get(k, 0))
            for k in self.token_stats
        }

        choice.reasoning = {**(choice.reasoning or {}),
                            "t": self.t, "calls": self.calls,
                            "level": self.level, "turn_in_level": self.turn_in_level,
                            "token_stats": dict(self.token_stats),
                            "token_delta": token_delta,
                            "tool_trace": trace,
                            "workspace": self.ws.snapshot(self.level, self.turn_in_level)}
        if self._log_io:
            from tycho.serving.llm_client import take_recording
            choice.reasoning["llm_calls"] = take_recording()
        self._last_action = choice.action.name
        self._last_committed_control = choice.action.name
        if choice.action.name == "RESET":
            self._level_resets = getattr(self, "_level_resets", 0) + 1
        # (Trigger-mode builder now fires at the START of the turn — see above — so the actor's
        # _run_until_action already reasoned on the corrected model. Nothing to fire here.)
        # Record this turn's builder activity: which agent ran, why it ran, and its report.
        if self._turn_builder_runs:
            choice.reasoning["builder_runs"] = self._turn_builder_runs
        if self._turn_animation_summaries:
            choice.reasoning["animation_summaries"] = self._turn_animation_summaries
        if batch_report is not None:
            choice.reasoning["batch_report"] = batch_report["data"]
        if self._batch is not None:
            choice.reasoning["batch"] = {"i": 0, "n": self._batch["planned"]}
        # track in (row, col) for the next turn's metadata (engine x/y → row/col)
        r = (choice.reasoning or {}).get("row")
        c = (choice.reasoning or {}).get("col")
        self._last_row = r if r is not None else getattr(choice, "y", None)
        self._last_col = c if c is not None else getattr(choice, "x", None)
        self.t += 1
        self.turn_in_level += 1
        return choice

    # ------------------------------------------------------------------------------------------
    # Event-driven agent graph (T2 3/3). The actor's lifecycle emits EVENTS; auxiliary agents
    # (the level-boundary consolidator; the trigger-mode builder) are AgentSpecs that fire on them. This
    # replaces the two special-cased inline hooks (the boundary consolidation call + the post-commit
    # builder block) with ONE dispatch path, so adding the future simplify/QA or fan-out agents is
    # a new AgentSpec, not new control flow. ADDITIVE: single/orchestrator carry ONLY the scribe
    # spec, whose run is the verbatim _summarize_level, fired on the same LEVEL_COMPLETE condition
    # at the same point — so their behaviour is byte-identical to the inline call.
    def _build_agent_graph(self) -> None:
        """Construct the auxiliary-agent graph. The ACTOR is implicit (it IS this object's
        choose_action); these are the agents triggered by its lifecycle events. When a DECLARATIVE
        graph was supplied (config file orchestration.agents), build from it; otherwise use the
        MODES preset (byte-identical to before)."""
        if self._agent_graph_decls is not None:
            self.agents = self._build_agent_graph_from_decls(self._agent_graph_decls)
            return
        specs: list[AgentSpec] = [
            # the level-boundary consolidator — present in EVERY mode; documents the level before the next.
            AgentSpec(id="scribe", trigger=on_events(Event.LEVEL_COMPLETE), kind=AgentKind.LLM,
                      sees_full_history=True, run=lambda ctx: self._summarize_level()),
        ]
        if self.dispatcher is not None:
            # trigger mode: the builder fires on (a) a committed action whose outcome the model
            # mispredicts (the inaccuracy gate), OR (b) a new level's first frame (LEVEL_START) — a
            # new level may need a model update its prior-level version can't anticipate, and the
            # mispredict gate can't see that until an action lands. ctx carries the level.
            specs.append(AgentSpec(
                id="builder",
                trigger=lambda ev, ctx: (ev is Event.ACTION_SUBMITTED and ctx.get("wm_inaccuracy_fire", False))
                                        or ev is Event.LEVEL_START
                                        or ev is Event.GAME_OVER_RESET,
                kind=AgentKind.LLM, sees_full_history=False,
                run=lambda ctx: self._run_trigger_builder(
                    ctx["level"],
                    at_level_start=ctx.get("at_level_start", False),
                    game_over_reset=ctx.get("game_over_reset", False),
                )))
        self.agents = specs

    def _build_agent_graph_from_decls(self, decls) -> list:
        """Map validated AgentDecls (config file orchestration.agents) to live AgentSpecs. The actor
        decl is implicit (its run IS choose_action), so it's skipped here; builder/scribe decls get
        the SAME run() the presets use — the config controls WHICH agents run and on WHAT trigger,
        not new code paths. Unknown event/type already raised in resolve_agent_graph()."""
        _EV = {"game_start": Event.GAME_START, "level_start": Event.LEVEL_START,
               "level_complete": Event.LEVEL_COMPLETE, "game_complete": Event.GAME_COMPLETE,
               "wm_updated": Event.WM_UPDATED, "action_submitted": Event.ACTION_SUBMITTED,
               "game_over_reset": Event.GAME_OVER_RESET, "agent_done": Event.AGENT_DONE}
        specs: list[AgentSpec] = []
        for d in decls:
            if d.type == "actor":
                continue  # the actor is this object's choose_action loop, not a dispatched agent
            ev = _EV[d.on]
            if d.on == "agent_done":
                trig = on_agent_done(d.after_agent)
            elif d.when == "wm_mispredicts":
                # the dispatcher evaluates the cheap inaccuracy gate once per turn and passes the
                # decision via ctx['wm_inaccuracy_fire'] (same as the preset trigger builder).
                trig = (lambda _ev: lambda ev, ctx: ev is _ev and ctx.get("wm_inaccuracy_fire", False))(ev)
            else:
                trig = on_events(ev)
            if d.type == "builder":
                # A builder ALSO fires on LEVEL_START and GAME_OVER_RESET in addition to its declared
                # trigger — same rationale as the preset path.
                base_trig = trig
                trig = (lambda _bt: lambda ev, ctx: _bt(ev, ctx)
                        or ev is Event.LEVEL_START
                        or ev is Event.GAME_OVER_RESET)(base_trig)
                run = (lambda ctx: self._run_trigger_builder(
                    ctx["level"],
                    at_level_start=ctx.get("at_level_start", False),
                    game_over_reset=ctx.get("game_over_reset", False),
                ))
            else:  # scribe
                run = (lambda ctx: self._summarize_level())
            specs.append(AgentSpec(id=d.id, trigger=trig, kind=AgentKind.LLM,
                                   sees_full_history=d.sees_full_history, run=run))
        return specs

    def _dispatch(self, event: Event, ctx: dict | None = None) -> None:
        """Run every agent whose trigger fires for `event`. Sync (each agent completes before the
        next); single/orchestrator only ever match the scribe/boundary consolidator on LEVEL_COMPLETE."""
        ctx = ctx or {}
        for spec in self.agents:
            if spec.run is not None and spec.trigger(event, ctx):
                spec.run(ctx)

    def _run_trigger_builder(self, level: int, at_level_start: bool = False,
                             game_over_reset: bool = False) -> None:
        """trigger-mode builder agent: fire the builder SYNC, account it, inject its advisory report
        as a user message (not a system note, which some transports fold into the global system
        prompt and thereby lose its per-turn placement) so the actor's next
        turn acts on the refreshed model. The gate (inaccuracy mispredict OR level-start) already
        passed in _dispatch's ctx. at_level_start=True → the new-level kickoff (check the new init
        frame against the model) rather than a mid-level divergence."""
        if game_over_reset:
            reason = self.dispatcher.game_over_reset_reason(level)
        elif at_level_start:
            reason = self.dispatcher.level_start_reason(level)
        else:
            reason = self.dispatcher.trigger_reason(level)
        report = self._invoke_builder(reason, reserve_calls=1)
        self.dispatcher.note_fired(level)
        self.history.append({"role": "user", "content":
            "[auto world-model builder fired — advisory, you decide the action]\n" + report})

    def _summarize_level(self):
        """On level completion, consolidate terminal evidence before the level transcript is reset.

        The completed level's observed terminal lives in level_<L>/terminal.* (written by the outer
        harness before choose_action re-enters here). In single mode the actor may update
        world_model.py; in builder-owned modes it writes model-relevant terminal insight to
        notes/world_model.md for the builder and leaves world_model.py alone."""
        if self.calls >= self.max_calls:
            self._write_level_insight_fallback(self.level, "LLM-call budget exhausted before consolidation")
            return
        if getattr(self, "scribe_mode", "tools") == "verified":
            return self._summarize_level_verified()
        self.history.append({"role": "user",
                              "content": render_prompt(
                                  "scribe.user", level=self.level,
                                  actor_writes_wm=self._mode_spec.actor_writes_wm,
                                  wm_variant=self._mode_spec.wm_variant)})
        # let the model make a couple of tool calls to write the summary, then continue
        for _ in range(self.scribe_max_calls):
            if self.calls >= self.max_calls:
                break
            reply = self._chat("level_summary", max_tokens=self.scribe_max_tokens or None)
            self.calls += 1
            self._note_llm_reply(reply)
            calls = reply.get("tool_calls") or []
            self.history.append({"role": "assistant", "content": reply.get("text", ""),
                                 "tool_calls": calls or None, "raw_reasoning": reply.get("raw_reasoning"),
                                 "reasoning_items": reply.get("reasoning_items")})
            if not calls:
                break
            results = []
            for c in calls:
                # ignore any take_action during the summary step; only run cognitive tools
                if canonical(c["name"]) == "take_action":
                    out = "(skip — finish the summary first)"
                elif self._is_builder_owned_world_model_edit(c):
                    out = self._world_model_edit_block_message(summary=True)
                else:
                    out = self.exec.execute(c["name"], c["input"])
                results.append({"id": c["id"], "name": c["name"], "output": out})
            self.history.append({"role": "tool", "results": results})
        self._write_level_insight_fallback(self.level, "summary pass did not create the requested note")

    def _summarize_level_verified(self) -> None:
        """TYCHO_SCRIBE_MODE=verified: the harness, not a tool call, writes the carry-forward note.

        The model gets the level's action/effect log inline and replies with the summary as text (rule
        table, goal check, unexplained changes, next-level checklist); the harness saves that text. With
        the tool-driven summary, DeepSeek-V4 spent its calls reading files and verifying, and 15 of 16
        single-mode boundaries (5 of 16 lite ones) fell back to the deterministic note."""
        path = f"notes/level_{self.level}_insights.md"
        log = self.ws.level_action_log(self.level)
        self.history.append({"role": "user", "content": render_prompt(
            "scribe_verified.user", level=self.level, action_log=log,
            actions=list(getattr(self, "_current_game_actions", None) or []) or ["the offered actions"])})
        text = ""
        max_tokens = max(self.scribe_max_tokens or self.max_tokens, 8192)
        for attempt in range(2):
            if self.calls >= self.max_calls:
                break
            reply = self._chat("level_summary", max_tokens=max_tokens)
            self.calls += 1
            self._note_llm_reply(reply)
            calls = reply.get("tool_calls") or []
            self.history.append({"role": "assistant", "content": reply.get("text", ""),
                                 "tool_calls": calls or None, "raw_reasoning": reply.get("raw_reasoning"),
                                 "reasoning_items": reply.get("reasoning_items")})
            if calls:
                self.history.append({"role": "tool", "results": [
                    {"id": c["id"], "name": c["name"], "output": "(not run: reply with the summary as plain text)"}
                    for c in calls]})
            text = (reply.get("text") or "").strip()
            if len(text) >= 300:
                break
            self.history.append({"role": "user", "content": (
                "Write the summary now as plain text in the six sections above. Do not call tools.")})
        if len(text) >= 300:
            self.ws.write_file(path, f"# Level {self.level} insights (level summary)\n\n{text}\n")
        else:
            self._write_level_insight_fallback(self.level, "summary pass did not produce a note",
                                               action_log=log)

    def _write_level_insight_fallback(self, level: int, reason: str, action_log: str | None = None) -> None:
        """Ensure level-boundary carry-forward exists if the LLM summary failed to write it. An empty or
        near-empty note (e.g. a write_file without content) counts as missing."""
        if getattr(self, "ws", None) is None:
            return
        path = f"notes/level_{level}_insights.md"
        existing = self.ws.read_file(path, max_bytes=200)
        if not existing.startswith("(no such file") and len(existing.strip()) >= 40:
            return
        if action_log is not None:
            self.ws.write_file(path, (
                f"# Level {level} Insights\n\nDeterministic fallback note: {reason}. The level summary did not "
                f"produce a note; this is the level's observed action log instead.\n\n{action_log}\n"))
            return
        solved = self.ws.read_file(f"level_{level}/solved.json", max_bytes=1000)
        terminal = self.ws.read_file(f"level_{level}/terminal.json", max_bytes=1800)
        content = (
            f"# Level {level} Insights\n\n"
            f"Deterministic fallback note: {reason}. The LLM consolidation step did not write "
            "a carry-forward note before the next level began.\n\n"
            f"solved.json:\n{solved}\n\n"
            f"terminal.json:\n{terminal}\n"
        )
        self.ws.write_file(path, content)

    def _chat(self, call_type, max_tokens=None):
        """chat_tools wrapper implementing bounded Responses-API state for proprietary GPT.

        With a LIVE chain (a prior response id from THIS level), send only the NEW history items
        (history[_chain_sent_upto:]) + previous_response_id — the server holds the rest plus the
        model's full reasoning chain. With NO live chain (level start, or after a break/reset), send
        the FULL client history (store=true so the NEXT call can chain it). Either way we capture the
        new response ID and advance the watermark past the assistant response already held by the
        server. Level boundaries and proactive long-context resets break the chain, after which it is
        re-established from the bounded client history; ordinary client eviction cannot alter server
        state.
        No-op for backends without response-ID continuity."""
        if not self._reasoning_continuity:
            return chat_tools(self.history, getattr(self, "_active_tools_spec", self._tools_spec),
                              self.cfg, system=self._sys,
                              max_tokens=max_tokens or self.max_tokens, effort=self.effort,
                              call_type=call_type, cache_anchor_index=self._cache_anchor_index())
        live = self._prev_response_id is not None
        reply = chat_tools(self.history, getattr(self, "_active_tools_spec", self._tools_spec),
                           self.cfg, system=self._sys,
                           max_tokens=max_tokens or self.max_tokens, effort=self.effort, call_type=call_type,
                           prev_response_id=self._prev_response_id if live else None,
                           since=self._chain_sent_upto if live else 0,
                           cache_anchor_index=self._cache_anchor_index())
        rid = reply.get("response_id")
        if rid:
            self._prev_response_id = rid
            # The server stores BOTH the request and this assistant response. The caller appends one
            # assistant history item immediately after _chat returns, so skip it on the next delta;
            # otherwise function_call/toolUse is duplicated alongside previous_response_id.
            self._chain_sent_upto = len(self.history) + 1
        else:
            # no id back (e.g. transient) → drop the chain; next call full-resends from history
            self._prev_response_id = None
            self._chain_sent_upto = 0
        return reply

    def _reset_history_for_new_level(self):
        """Clear the client conversation at a level boundary so EACH LEVEL IS FACED FRESH. ARC-AGI-3
        has one underlying transition function revealed progressively; what carries forward is the
        agent's DURABLE state on disk — notes/ (the boundary step just wrote notes/level_<L>_insights.md),
        world_model.py, actor_beliefs.md / world_model.md — NOT the prior level's turn-by-turn transcript, which has little
        value and is the thing that grows the prompt past the model window on long games. The
        fresh-level frame message (turn_in_level==0) re-primes the agent and points it
        to notes/, and the agent re-reads world_model.py via its tools — the validated warm-restart
        seed pattern. _sys is sent separately (chat_tools(system=...)), so the system prompt is
        untouched; _truncate_old_grids recomputes from self.history so an empty history is consistent;
        resume is consistent because snapshot_state captures whatever self.history is at snapshot time.
        Called AFTER boundary consolidation (which reads the full just-completed-level history and
        terminal event to write its notes)."""
        self.history = []

    def reset_reasoning_chain(self):
        """Drop the server-side reasoning chain so the next call re-establishes it from the
        (bounded, client-evicted) history. Called at each LEVEL boundary to keep SERVER context from
        growing unboundedly (the server-side analogue of _truncate_old_* — which can't reach it)."""
        self._prev_response_id = None
        self._chain_sent_upto = 0
        self._reasoning_chain_rollover_pending = False

    def _rollover_reasoning_chain(self, reason: str, trace: list[dict]) -> None:
        """Drop only opaque server state, preserving the complete visible client transcript.

        The next call full-sends the already bounded client history and starts a fresh chain. If that
        full prompt is still above the soft limit, the following pre-call check escalates to the
        destructive emergency rescue instead of rolling over repeatedly.
        """
        self.reset_reasoning_chain()
        self._reasoning_chain_rollover_pending = True
        self._last_prompt_tokens = 0
        self.token_stats["reasoning_chain_rollovers"] = int(
            self.token_stats.get("reasoning_chain_rollovers") or 0
        ) + 1
        trace.append({"text": "opaque reasoning chain rolled over", "reason": reason[:500],
                      "visible_history_preserved": True})

    def _cache_anchor_index(self, history: list[dict] | None = None) -> int:
        """Last provider-message index safe for a rolling cache point.

        "Safe" means every message up to this point is expected to remain byte-identical under
        Tycho's future retention policy. Messages that still carry images, prompt-embeds, raw
        reasoning, or elidable heavy text/tool payloads are intentionally outside the cached prefix:
        those payloads may be stripped later, and a cache point beyond them would pay cache-write
        cost for a prefix that Tycho itself is likely to mutate.

        Return -1 to disable the rolling conversation cache point for this call. `chat_tools` treats
        None as "use provider fallback", so the agent passes an explicit negative anchor when its
        own retention policy says no stable message exists.
        """

        hist = self.history if history is None else history
        if not getattr(getattr(self, "_ctx_cfg", None), "prompt_caching", True):
            return -1
        mode = getattr(self, "context_mode", "tail_evict")
        if mode not in ("tail_evict", "drain"):
            return -1
        every = int(getattr(getattr(self, "_ctx_cfg", None), "cache_rolling_every", 1) or 0)
        if every <= 0 or len(hist) < 2:
            return -1

        stable_upto = len(hist) - 2  # keep the newest request delta outside the rolling breakpoint
        # drain never edits a kept message, so nothing before the newest delta is mutable
        first_mutable = None if mode == "drain" else self._first_cache_mutable_message_index(hist)
        if first_mutable is not None:
            stable_upto = min(stable_upto, first_mutable - 1)
        if stable_upto < 0:
            return -1
        if every == 1:
            return stable_upto
        return max(0, (stable_upto // every) * every)

    def _first_cache_mutable_message_index(self, history: list[dict]) -> int | None:
        for i, msg in enumerate(history):
            if _has_image(msg) or _has_grid_embeds(msg) or _has_reasoning(msg):
                return i
            if getattr(self, "cache_text_keep", 0) > 0 and _is_elidable_text_block(msg):
                return i
        return None

    def _reset_animation_summary_level_state(self) -> None:
        self._animation_summary_cache = {}
        self._animation_summary_count_this_level = 0
        self._animation_summary_last_signature = None

    def _animation_summary_parts(self, frames, *, action: str | None, terminal: str,
                                 row: int | None = None,
                                 col: int | None = None,
                                 event_level: int | None = None,
                                 event_turn: int | None = None,
                                 event_root: str | None = None,
                                 level_just_completed: bool = False,
                                 force: bool = False) -> list[dict]:
        """Persist numeric animation evidence and optionally attach a short text summary.

        The contact sheet is sent only to a transient summarizer call over a copied actor history.
        The actor history receives only text, so normal image retention/caps are not affected.
        """

        if level_just_completed and terminal != "game_over":
            return []
        if frames is None or len(frames) <= 1:
            return []
        if not action:
            return []

        try:
            decision = analyze_frame_sequence(frames, self._animation_evidence_config)
        except Exception as e:  # noqa: BLE001
            self._turn_animation_summaries.append({
                "terminal": terminal, "action": action, "error": f"{type(e).__name__}: {e}",
            })
            return []

        action_label = self._action_label(action, row=row, col=col)
        coarse_sig = animation_signature(decision, action=action_label, terminal=terminal)
        frame_hash = self._animation_selected_frame_hash(frames, decision.selected_original_indices)
        sig = tuple(sorted([*coarse_sig, ("selected_frame_hash", frame_hash)]))
        cached = self._animation_summary_cache.get(sig)
        reused = cached is not None
        summary = cached or ""
        skipped = ""
        summary_available = (
            self.animation_summary_enabled
            and self.cache_image_cap > 0
            and (terminal != "game_over" or self.animation_summary_include_game_over)
        )
        should_summarize = (decision.surface or force) and summary_available
        if not should_summarize:
            if not decision.surface and not force:
                skipped = f"deterministic filter classified this as {decision.category}"
            elif not self.animation_summary_enabled:
                skipped = "visual summary disabled or unavailable; frames saved"
            elif self.cache_image_cap <= 0:
                skipped = "image cap leaves no transient summarizer image slot; frames saved"
            elif terminal == "game_over" and not self.animation_summary_include_game_over:
                skipped = "GAME_OVER visual summary disabled; frames saved"
            else:
                skipped = "visual summary not run; frames saved"
        elif not summary:
            if self._animation_summary_count_this_level >= self.animation_summary_max_per_level:
                skipped = "per-level animation-summary cap reached"
            elif self.calls >= self.max_calls - 1:
                skipped = "LLM-call budget reserved for actor"
            else:
                summary = self._summarize_animation_with_llm(
                    frames, decision, action=action_label, terminal=terminal
                )
                if summary:
                    self._animation_summary_cache[sig] = summary
                    self._animation_summary_count_this_level += 1
        if should_summarize and not summary and not skipped:
            skipped = "summary call returned no text"
        event = self._record_animation_evidence(
            frames,
            decision,
            action_label=action_label,
            terminal=terminal,
            row=row,
            col=col,
            event_level=event_level,
            event_turn=event_turn,
            summary=summary,
            reused=reused,
            skipped=skipped,
            signature=sig,
            frame_hash=frame_hash,
            storage_root=event_root,
        )
        event_dir = (
            event.get("workspace_directory", event.get("directory", "")) if event else ""
        )
        trace = {
            "terminal": terminal,
            "action": action_label,
            "row": row,
            "col": col,
            "category": decision.category,
            "reused": reused,
            "skipped": skipped,
            "count_this_level": self._animation_summary_count_this_level,
            "cap_per_level": self.animation_summary_max_per_level,
            "selected_frames": decision.selected_original_indices,
            "surface": decision.surface,
            "coarse_signature": coarse_sig,
            "signature": sig,
            "selected_frame_hash": frame_hash,
            "workspace_dir": event_dir or None,
        }

        self._turn_animation_summaries.append({**trace, **({"summary": summary} if summary else {})})
        if event:
            self._append_animation_evidence_note(
                action_label=action_label,
                terminal=terminal,
                category=decision.category,
                original_frame_count=decision.original_frame_count,
                selected_frames=decision.selected_original_indices,
                reused=reused,
                skipped=skipped,
                summary=summary,
                event_dir=event_dir,
                event_level=self.level if event_level is None else event_level,
                event_turn=self.turn_in_level if event_turn is None else event_turn,
            )
        self._animation_summary_last_signature = sig
        return [{"text": self._animation_evidence_actor_text(
            action_label=action_label,
            terminal=terminal,
            decision=decision,
            reused=reused,
            skipped=skipped,
            summary=summary,
            event_dir=event_dir,
        )}]

    @staticmethod
    def _animation_selected_frame_hash(frames, selected_indices: list[int]) -> str:
        h = hashlib.sha256()
        try:
            import numpy as _np
            for idx in selected_indices:
                if idx < 0 or idx >= len(frames):
                    continue
                arr = _np.asarray(frames[idx], dtype=_np.int16)
                h.update(str(idx).encode("ascii"))
                h.update(str(tuple(arr.shape)).encode("ascii"))
                h.update(arr.tobytes())
            return h.hexdigest()[:20]
        except Exception:  # noqa: BLE001
            return "unavailable"

    def _record_animation_evidence(self, frames, decision, *, action_label: str, terminal: str,
                                   row: int | None, col: int | None,
                                   event_level: int | None, event_turn: int | None,
                                   summary: str, reused: bool, skipped: str,
                                   signature, frame_hash: str,
                                   storage_root: str | None = None) -> dict:
        if not getattr(self, "ws", None):
            return {}
        try:
            return self.ws.record_animation_event(
                level=self.level if event_level is None else event_level,
                turn_in_level=self.turn_in_level if event_turn is None else event_turn,
                action=action_label,
                row=row,
                col=col,
                terminal=terminal,
                frames=frames,
                selected_indices=decision.selected_original_indices,
                decision=decision.to_json(),
                summary=summary,
                reused=reused,
                skipped=skipped,
                signature=signature,
                frame_hash=frame_hash,
                keep_per_level=self.animation_evidence_keep_per_level,
                storage_root=storage_root,
            ) or {}
        except Exception:  # noqa: BLE001 - animation evidence is auxiliary
            return {}

    def _animation_evidence_actor_text(self, *, action_label: str, terminal: str, decision,
                                       reused: bool, skipped: str, summary: str,
                                       event_dir: str) -> str:
        label = "GAME_OVER animation evidence" if terminal == "game_over" else "animation evidence"
        if summary:
            status = "exact cached summary reused for the same selected keyframes" if reused else "new transient summary produced"
        else:
            status = f"no new visual summary ({skipped or 'not summarized'})"
        files = (
            f" Animation frames are saved under {event_dir}; inspect them with "
            "`wmlib.animation_index()` plus `wmlib.animation_grids(...)`, or read the frame_*.txt/png files."
            if event_dir else ""
        )
        if summary:
            text = (
                f"[{label}] Action {action_label} returned {decision.original_frame_count} animation frames; "
                f"selected frames {decision.selected_original_indices} ({decision.category}: "
                f"{'; '.join(decision.reasons)}). {status}.{files}"
            )
            text += (
                "\nHeuristic visual caption generated by a separate LLM from the selected keyframes, "
                "not from every saved frame and not from the game engine. It can miss or mislabel "
                "hazards, goals, deaths, and small object interactions; treat saved frames, "
                "wmlib.animation_index()/animation_grids(), grid/diff, and API state as the evidence:\n"
                f"{summary.strip()[:1400]}\n"
            )
        else:
            text = (
                f"[animation frames] Action {action_label} returned {decision.original_frame_count} frames; "
                f"selected frames {decision.selected_original_indices}. {status}.{files}\n"
            )
        return text

    @staticmethod
    def _action_label(action: str | None, *, row: int | None = None, col: int | None = None) -> str:
        if action == "ACTION6" and row is not None and col is not None:
            return f"ACTION6(row={row},col={col})"
        return action or "?"

    def _append_animation_evidence_note(self, *, action_label: str, terminal: str,
                                        category: str, original_frame_count: int,
                                        selected_frames: list[int],
                                        reused: bool, skipped: str, summary: str,
                                        event_dir: str, event_level: int, event_turn: int) -> None:
        """Persist transient animation captions so builder modes can inspect them later."""
        try:
            path = "notes/animation_evidence.md"
            existing = self.ws.read_file(path, max_bytes=1_000_000)
            if existing.startswith("(no such file:"):
                existing = "# Animation Evidence\n"
            selected = ",".join(str(i) for i in selected_frames)
            reuse = " exact-cache-reused" if reused else ""
            status = (
                f"heuristic selected-keyframe caption, not hard evidence: {summary.strip()[:700]}"
                if summary else f"no summary: {skipped or 'not summarized'}"
            )
            files = f"; frames: {event_dir}" if event_dir else ""
            entry = (
                f"\n- level {event_level}, turn {event_turn}, {terminal}, "
                f"{action_label}, {category}, {original_frame_count} frames, "
                f"summary-selected [{selected}]{reuse}{files}: "
                f"{status}\n"
            )
            self.ws.write_file(path, entry.lstrip() + "\n" + existing.lstrip())
        except Exception:  # noqa: BLE001 - animation notes are auxiliary evidence
            return

    def _animation_summary_system(self) -> str:
        return (
            "You summarize short ARC-AGI-3 grid animations for a game-playing agent. "
            "Return 1-3 concise sentences. Focus on state-relevant dynamics: camera or viewport "
            "motion, newly revealed areas, object interactions, hazards, death/game-over effects, "
            "and whether the intermediate frames explain a jump between playable grids. "
            "Do not speculate beyond the frames and do not give strategic advice. "
            "Do not call tools or choose an action in this transient side task."
        )

    def _animation_summary_user_text(self, decision, *, action: str, terminal: str,
                                     media: str = "contact_sheet",
                                     selected_indices: list[int] | None = None) -> str:
        selected = decision.selected_original_indices if selected_indices is None else selected_indices
        if media == "images":
            visual_note = (
                "The attached rendered images are the selected keyframes in chronological order. "
                "Each image is preceded by its original frame index; this is a keyframe sequence "
                "from the returned animation, not necessarily every original frame when indices are skipped."
            )
        elif media == "video":
            visual_note = (
                "The attached video contains the selected rendered keyframes in chronological order. "
                "It is a keyframe sequence from the returned animation, not necessarily every original "
                "frame when indices are skipped."
            )
        else:
            visual_note = (
                "The contact sheet shows the selected frames in chronological order."
            )
        return (
            f"Summarize this transition animation for the next actor turn.\n"
            f"Action: {action}\n"
            f"Transition type: {terminal}\n"
            f"Selected original frame indices: {selected}\n"
            f"{visual_note} "
            "Use the existing conversation only as context for what the agent already knows."
        )

    def _animation_summary_history(self, user_text: str, media_parts: dict | list[dict],
                                   *, strip_history_media: bool = False) -> list[dict]:
        """Copy actor history unchanged and append the transient animation media.

        The fast path intentionally preserves the actor prefix byte-for-byte: same tools, same
        system prompt at the call site, same prior messages. That lets provider/vLLM prefix caches
        treat the side call as a normal extension of the actor conversation. Multi-image evidence
        can optionally strip old media from this copied side-call history to stay inside Qwen's image
        cap while preserving text context and leaving the real actor history untouched.
        """

        if any(_has_grid_embeds(msg) for msg in self.history):
            raise ValueError("contextual animation summary disabled when grid embeds are present")
        media_list = media_parts if isinstance(media_parts, list) else [media_parts]
        added_images = _content_media_payload_count(media_list)
        if added_images > self.cache_image_cap:
            raise ValueError(
                f"contextual animation summary media exceeds image cap "
                f"({added_images}>{self.cache_image_cap})")
        existing_images = _image_payload_count(self.history)
        if existing_images + added_images > self.cache_image_cap and not strip_history_media:
            raise ValueError(
                f"contextual animation summary would exceed image cap "
                f"({existing_images + added_images}>{self.cache_image_cap})")
        hist = copy.deepcopy(self.history)
        if strip_history_media:
            for msg in hist:
                if _image_payload_count(hist) + added_images <= self.cache_image_cap:
                    break
                _strip_image(msg)
        hist.append({"role": "user", "content": [{"text": user_text}, *media_list]})
        return hist

    def _animation_summary_png(self, frames, selected_indices: list[int], *,
                               action: str, terminal: str, scale: int | None = None) -> bytes:
        title = f"{terminal} {action}"
        return contact_sheet_png_bytes(
            frames,
            selected_indices,
            scale=scale or self._animation_summary_scale,
            title=title,
        )

    def _animation_summary_video_indices(self, frames, selected_indices: list[int]) -> list[int]:
        # Qwen3.6's video preprocessor budget is 25,165,824 total pixels. At Tycho's normal
        # Qwen scale (64*32)^2, that permits exactly six frames; if a future config asks for
        # more, keep endpoints and evenly spaced interior keyframes rather than letting vLLM resize.
        max_video_pixels = 25_165_824
        try:
            shape = getattr(frames[0], "shape", None)
            if shape is not None:
                h, w = int(shape[0]), int(shape[1])
            else:
                h, w = len(frames[0]), len(frames[0][0])
        except Exception:  # noqa: BLE001
            h, w = 64, 64
        pixels_per_frame = max(1, h * w * self._animation_summary_video_scale * self._animation_summary_video_scale)
        max_frames = int(max(2, max_video_pixels // pixels_per_frame))
        return self._sample_animation_indices(selected_indices, max_frames)

    def _animation_summary_image_indices(self, selected_indices: list[int]) -> list[int]:
        # Some multimodal servers use a 4-image prompt cap. Multi-image animation
        # evidence spends that entire transient budget on keyframes, so keep endpoints and sample
        # interiors rather than silently falling back to a low-resolution contact sheet.
        return self._sample_animation_indices(selected_indices, max(1, self.cache_image_cap))

    @staticmethod
    def _sample_animation_indices(selected_indices: list[int], max_frames: int) -> list[int]:
        if not selected_indices:
            return []
        if len(selected_indices) <= max_frames:
            return list(selected_indices)
        if max_frames <= 1:
            return [selected_indices[-1]]
        if max_frames == 2:
            return [selected_indices[0], selected_indices[-1]]
        slots = max_frames - 1
        out = []
        for i in range(max_frames):
            pos = round(i * (len(selected_indices) - 1) / slots)
            idx = selected_indices[pos]
            if not out or out[-1] != idx:
                out.append(idx)
        return out

    def _animation_summary_image_parts(self, frames, selected_indices: list[int]) -> list[dict]:
        pngs = selected_frame_png_bytes(
            frames,
            selected_indices,
            scale=self._animation_summary_video_scale,
        )
        parts: list[dict] = []
        for idx, png in zip(selected_indices, pngs):
            parts.append({"text": f"Selected animation frame {idx}:"})
            parts.append({"image_png": png})
        return parts

    def _animation_summary_video_part(self, frames, selected_indices: list[int]) -> dict:
        meta = {
            "fps": 2,
            "do_sample_frames": False,
            "frames_indices": list(selected_indices),
            "total_num_frames": len(frames),
        }
        if self.animation_summary_video_format == "gif":
            return {
                "video_gif": selected_frame_gif_bytes(
                    frames,
                    selected_indices,
                    scale=self._animation_summary_video_scale,
                ),
                "video_metadata": meta,
            }
        return {
            "video_jpeg_frames": selected_frame_jpeg_bytes(
                frames,
                selected_indices,
                scale=self._animation_summary_video_scale,
            ),
            "video_metadata": {
                **meta,
            },
        }

    @staticmethod
    def _reduced_animation_indices(indices: list[int]) -> list[int]:
        if len(indices) <= 3:
            return list(indices)
        mid = indices[len(indices) // 2]
        return [indices[0], mid, indices[-1]]

    def _summarize_animation_with_llm(self, frames, decision, *, action: str, terminal: str) -> str:
        system = self._animation_summary_system()
        contextual_error = ""

        def _try_contextual(media: str) -> str:
            selected_indices = decision.selected_original_indices
            if media == "video":
                selected_indices = self._animation_summary_video_indices(frames, selected_indices)
            elif media == "images":
                selected_indices = self._animation_summary_image_indices(selected_indices)
            user = (
                self._animation_summary_user_text(
                    decision,
                    action=action,
                    terminal=terminal,
                    media=media,
                    selected_indices=selected_indices,
                )
                + "\nReturn only the summary text. Do not call tools, do not emit JSON, and do not choose an action."
            )
            if media == "video":
                media_parts = self._animation_summary_video_part(frames, selected_indices)
                strip_history_media = False
            elif media == "images":
                media_parts = self._animation_summary_image_parts(frames, selected_indices)
                strip_history_media = True
            else:
                media_parts = {"image_png": self._animation_summary_png(
                    frames,
                    selected_indices,
                    action=action,
                    terminal=terminal,
                )}
                strip_history_media = False
            summary_history = self._animation_summary_history(
                user,
                media_parts,
                strip_history_media=strip_history_media,
            )
            reply = chat_tools(
                summary_history,
                [],
                self.cfg,
                system=system,
                max_tokens=self.animation_summary_max_tokens,
                effort="",
                call_type="animation_summary",
                cache_anchor_index=self._cache_anchor_index(summary_history),
            )
            self.calls += 1
            self._note_llm_reply(reply)
            if reply.get("tool_calls"):
                raise RuntimeError("animation-summary call returned tool calls")
            return (reply.get("text") or "").strip()

        try:
            if self.animation_summary_video_enabled:
                try:
                    if self.animation_summary_video_format == "images":
                        return _try_contextual("images")
                    return _try_contextual("video")
                except Exception as e:  # noqa: BLE001
                    contextual_error = f"{self.animation_summary_video_format} {type(e).__name__}: {e}"
            return _try_contextual("contact_sheet")
        except Exception as e:  # noqa: BLE001
            if contextual_error:
                contextual_error = f"{contextual_error}; contact_sheet {type(e).__name__}: {e}"
            else:
                contextual_error = f"{type(e).__name__}: {e}"

        if self.calls >= self.max_calls - 1:
            self._turn_animation_summaries.append({
                "terminal": terminal,
                "action": action,
                "category": decision.category,
                "error": contextual_error,
                "fallback_skipped": "LLM-call budget reserved for actor",
            })
            return ""

        try:
            fallback_indices = self._reduced_animation_indices(decision.selected_original_indices)
            fallback_png = self._animation_summary_png(
                frames,
                fallback_indices,
                action=action,
                terminal=terminal,
                scale=4,
            )
            fallback_user = (
                self._animation_summary_user_text(decision, action=action, terminal=terminal)
                + "\nThe contextual summary call failed, so this fallback has less history and "
                f"only selected frame indices {fallback_indices}. Return only the summary text."
            )
            reply = chat_tools(
                [{"role": "user", "content": [{"text": fallback_user}, {"image_png": fallback_png}]}],
                [],
                self.cfg,
                system=system,
                max_tokens=self.animation_summary_max_tokens,
                effort="",
                call_type="animation_summary",
            )
            self.calls += 1
            self._note_llm_reply(reply)
            return (reply.get("text") or "").strip()
        except Exception as e:  # noqa: BLE001
            self._turn_animation_summaries.append({
                "terminal": terminal,
                "action": action,
                "category": decision.category,
                "error": contextual_error,
                "fallback_error": f"{type(e).__name__}: {e}",
            })
            return ""

    def note_actor_reset(self) -> None:
        """The actor deliberately committed RESET on a playable frame.

        RESET starts a fresh attempt of the same level. It is not an ordinary game mechanic, and
        the reset frame must not be recorded as the result of a modeled transition.
        """
        self._pending_game_over_event = None
        self._last_action = None
        self._last_row = None
        self._last_col = None
        self._last_committed_control = "RESET"
        self.turn_in_level = 0
        self._pending_frame_boundary = "attempt_restart"
        if getattr(self, "ws", None) is not None and hasattr(self.ws, "reset_level"):
            self.ws.reset_level(self.level, reason="actor_reset")
        self.reset_reasoning_chain()
        if getattr(self, "history", None) is not None:
            self.history.append({"role": "user", "content":
                "[harness reset] You chose RESET. The next playable frame starts a fresh attempt "
                "of this same level and must not be treated as the previous action's modeled "
                "outcome. Reusable discoveries should live in notes or other workspace files."})

    def note_external_reset(self, event: dict | None = None) -> None:
        """The harness had to RESET after GAME_OVER without asking Tycho for an action.

        The next real frame is a fresh level-start state, not the result of the prior lethal action.
        Clear the current level's generated observation files and action metadata so the world-model
        verifier cannot learn a fake "lethal action -> reset frame" transition.
        """
        archived = None
        if getattr(self, "_batch", None) is not None:
            self._finish_batch("game_over", None)
        self._last_action = None
        self._last_row = None
        self._last_col = None
        self.turn_in_level = 0
        self._pending_frame_boundary = "attempt_restart"
        if getattr(self, "ws", None) is not None and hasattr(self.ws, "reset_level"):
            archived = self.ws.reset_level(self.level, reason="game_over")
        if event is not None:
            event = dict(event)
            if archived and archived.get("root"):
                event["attempt_root"] = archived["root"]
        self._pending_game_over_event = event
        self.reset_reasoning_chain()
        if getattr(self, "history", None) is not None:
            self.history.append({"role": "user", "content":
                "[harness reset] The previous action caused GAME_OVER. The framework reset this level; "
                "the next playable frame is a fresh start and must not be treated as the prior action's outcome. "
                "The GAME_OVER evidence will be attached to the next turn."})

    def _game_over_evidence_parts(self) -> list[dict]:
        """Attach one pending API GAME_OVER event to the next playable reset-frame prompt.

        GAME_OVER is terminal evidence, but the harness must RESET before the agent can act again.
        This gives the actor the same kind of explicit boundary evidence solved levels get: the
        pre-fatal frame, the fatal action, the observed GAME_OVER frame, and the compact diff.
        """
        ev = getattr(self, "_pending_game_over_event", None)
        if not ev:
            return []
        self._pending_game_over_event = None
        self._level_deaths = getattr(self, "_level_deaths", 0) + 1
        lvl = int(ev.get("level", self.level))
        turn = int(ev.get("turn", self.turn_in_level))
        action = ev.get("action") or "?"
        rc = ""
        if ev.get("row") is not None and ev.get("col") is not None:
            rc = f" at row={ev.get('row')}, col={ev.get('col')}"
        death_stem = ev.get("stem") or f"death_{turn:03d}"
        stem = f"level_{lvl}/{death_stem}"
        diff = self.ws.read_file(f"{stem}_diff.txt", max_bytes=2500) if getattr(self, "ws", None) else ""
        text = (
            f"[GAME_OVER evidence] Your previous committed action {action}{rc} caused API GAME_OVER "
            f"on level {lvl}, turn {turn}. The framework then RESET this same level automatically; "
            "the current frame below is a fresh playable reset state, NOT the direct outcome of that "
            "fatal action.\n\n"
            f"Evidence files: {stem}.json, {stem}_prev.txt, {stem}_next.txt, {stem}_diff.txt.\n"
            "Treat the modeled post-fatal-action state as outcome(state) == \"game_over\" and avoid "
            "equivalent actions unless deliberately probing.\n"
        )
        if self._mode_spec.actor_writes_wm:
            text += (
                "If this failed attempt revealed stable facts about the deterministic level start, "
                "hidden map, hazards, counters, or off-screen geometry, encode those facts in "
                "notes/world_model.md and in init_state()/level constants before retrying, so the "
                "next attempt can plan from a better initial state instead of rediscovering them.\n"
            )
        if diff and not diff.startswith("(no such file"):
            text += "\nDiff from pre-fatal frame to observed GAME_OVER frame:\n" + diff
        parts: list[dict] = [{"text": text}]
        parts.extend(self._animation_summary_parts(
            ev.get("frames"),
            action=action,
            row=ev.get("row"),
            col=ev.get("col"),
            terminal="game_over",
            event_level=lvl,
            event_turn=turn,
            event_root=ev.get("attempt_root"),
            force=True,
        ))
        # The next actor turn also carries the fresh reset-frame PNG. Attach the two evidence PNGs
        # only when the configured cap can accommodate all three newest images; _cache_mode_evict_if_needed
        # then drops older historical images first so this reset frame and the death evidence survive.
        if self.vision and self.cache_image_cap >= 3 and getattr(self, "ws", None) is not None:
            prev_png = self.ws.dir / f"{stem}_prev.png"
            next_png = self.ws.dir / f"{stem}_next.png"
            if prev_png.exists() and next_png.exists():
                parts.append({"text": "\nGAME_OVER PNG 1 = pre-fatal playable frame; PNG 2 = observed GAME_OVER frame."})
                parts.append({"image_png": prev_png.read_bytes()})
                parts.append({"image_png": next_png.read_bytes()})
        return parts

    def _collapse_prev_turn(self):
        """ROLLING context mode: rewrite the PRIOR turn's messages (its frame user-msg + the
        within-turn tool transcript + the assistant turns) down to a SINGLE compact assistant
        message recording the committed action + a one-line reason. This makes the persistent
        history `[system] + committed actions only`, so the prefix is byte-stable turn-to-turn
        and vLLM's prefix cache reuses it (docs/context-caching-design.md). Lossless wrt what we
        intend to keep: old grids/images live on disk (agent re-reads via tools) and hypotheses
        in notes/actor_beliefs.md / notes/world_model.md. No-op until there's a prior turn to collapse."""
        h = self.history
        # find the start of the previous turn: the last user message carrying a frame (grid+image)
        start = None
        for i in range(len(h) - 1, -1, -1):
            c = h[i].get("content")
            if h[i].get("role") == "user" and isinstance(c, list) and any(
                    ("image_png" in p or "grid_embeds_part" in p or "text" in p) for p in c):
                start = i
                break
        if start is None:
            return  # nothing to collapse yet (first turn)
        # the committed action of that turn = self._last_action (+ row/col, set after the turn ran)
        la = getattr(self, "_last_action", None)
        if la is None:
            return
        rc = ""
        r, c = getattr(self, "_last_row", None), getattr(self, "_last_col", None)
        if la == "ACTION6" and r is not None and c is not None:
            rc = f" (row={r}, col={c})"
        note = f"I observed the frame and committed {la}{rc}."
        # replace the whole span [start:] with one compact assistant message
        del h[start:]
        h.append({"role": "assistant", "content": note})

    def _should_emergency_compact_before_call(self) -> bool:
        if not self.context_emergency_compaction:
            return False
        if self.context_emergency_soft_tokens <= 0:
            return False
        if self._last_prompt_tokens < self.context_emergency_soft_tokens:
            return False
        return (
            self._last_context_compaction_at != (self.level, self.turn_in_level)
            or len(self.history) > self._last_context_compaction_history_len
        )

    def _prepare_context_before_call(self, trace: list[dict]) -> None:
        """Apply the least destructive context safeguard justified by the last usage report."""
        if self.context_mode == "drain":
            self._drain_history(trace)
        if not self.context_emergency_compaction:
            return
        if self.context_emergency_soft_tokens <= 0:
            return
        if self._last_prompt_tokens < self.context_emergency_soft_tokens:
            return
        reason = (
            f"last prompt estimate {self._last_prompt_tokens} >= "
            f"soft limit {self.context_emergency_soft_tokens}"
        )
        # A chained response carries opaque server history that client-side eviction cannot edit.
        # First discard only that chain. The next request sends the complete, already bounded client
        # transcript; emergency compaction is reserved for a full request that is itself too large.
        if (
            self._reasoning_continuity
            and self._prev_response_id is not None
            and not self._reasoning_chain_rollover_pending
        ):
            self._rollover_reasoning_chain(reason, trace)
        elif self._should_emergency_compact_before_call():
            self._compact_context_for_emergency(reason, trace)

    def _drain_history(self, trace: list[dict], force: bool = False) -> bool:
        """Drain mode: drop the oldest whole frame-turns in ONE batch once the prompt is over budget.

        Franzen's harness (M0) trims the same way: the transcript is append-only between drains, so
        every call reuses the prefix cache; a drain invalidates it once and frees ``drain_tokens``
        below the budget, so the next drain is many turns away. A frame-turn is a frame user message
        plus everything up to the next one, so tool-call/result pairs are never split. The newest
        frame-turn (the live decision) always survives. ``force`` (a context-limit error) drains
        regardless of the last usage report, keeping only that newest turn.
        """
        over_tokens = self._last_prompt_tokens >= self.context_drain_budget
        over_images = self.cache_image_cap > 0 and _image_payload_count(self.history) > self.cache_image_cap
        if not (force or over_tokens or over_images):
            return False
        if over_images and not (force or over_tokens) and getattr(self, "drain_images", "drop") == "strip":
            # TYCHO_DRAIN_IMAGES=strip: only the image cap is exceeded, so strip the oldest images down to
            # image_keep and keep their turns' text, tool results and reasoning. Dropping whole turns for
            # it (the default) drained a DeepSeek-V4 run every ~5 decisions on vLLM's 8-image limit (78 of
            # 84 drains in 25 games), wiping the agent's working memory each time. One re-prefill per strip.
            keep = max(1, min(self.cache_image_keep, self.cache_image_cap))
            stripped = 0
            for m in self.history:
                if _image_payload_count(self.history) <= keep:
                    break
                if _has_media_payload(m):
                    _strip_image(m)
                    stripped += 1
            self.token_stats["image_strips"] = int(self.token_stats.get("image_strips", 0)) + 1
            trace.append({"text": "images stripped", "stripped_messages": stripped,
                          "kept_images": _image_payload_count(self.history)})
            return False
        starts = [i for i, m in enumerate(self.history) if self._is_frame_user_message(m)]
        if len(starts) < 2:
            # one frame-turn holds the whole transcript (a long tool loop): shrink inside it
            return self._shrink_live_turn(trace, force=force)
        # Per-message token estimate (text chars / 3, images at a flat cost), scaled so the whole
        # history matches the server's last prompt count (which also covers system + tools).
        est = [_msg_text_len(m) / 3.0 + _DRAIN_IMAGE_TOKENS * _content_media_payload_count(
            m.get("content") if isinstance(m.get("content"), list) else []) for m in self.history]
        total = sum(est)
        scale = (self._last_prompt_tokens / total) if (total > 0 and self._last_prompt_tokens > total) else 1.0
        target = max(0.0, self.context_drain_budget - self.context_drain_tokens)
        image_target = self.cache_image_cap // 2 if self.cache_image_cap > 0 else None
        cut = starts[-1] if force else None
        if cut is None:
            prompt, images = self._last_prompt_tokens or total * scale, _image_payload_count(self.history)
            for k in range(1, len(starts)):
                dropped = self.history[:starts[k]]
                prompt_k = prompt - scale * sum(est[:starts[k]])
                images_k = images - _image_payload_count(dropped)
                cut = starts[k]
                if prompt_k <= target and (image_target is None or images_k <= image_target):
                    break
        dropped_turns = sum(1 for i in starts if i < cut)
        kept = self.history[cut:]
        note = (f"[context drained: the {dropped_turns} oldest frame-turn(s) of this level were removed to fit "
                f"the context window. This is not a game event. Their frames and diffs are on disk under "
                f"level_{self.level}/; notes/ holds what you wrote.]")
        first = dict(kept[0])
        first["content"] = [{"text": note}, *first["content"]]
        self.history = [first, *kept[1:]]
        self._last_prompt_tokens = 0
        self.token_stats["context_drains"] = int(self.token_stats.get("context_drains", 0)) + 1
        self.reset_reasoning_chain()
        trace.append({"text": "context drained", "dropped_turns": dropped_turns, "kept_messages": len(self.history),
                      "force": force, "over_tokens": over_tokens, "over_images": over_images})
        kept_tokens = scale * sum(_msg_text_len(m) / 3.0 for m in self.history)
        if force or kept_tokens > self.context_drain_budget:
            self._shrink_live_turn(trace, force=force)
        return True

    def _shrink_live_turn(self, trace: list[dict], *, force: bool = False, keep_steps: int = 4) -> bool:
        """Drain mode, when the live frame-turn alone is too big: keep its frame message and its last
        ``keep_steps`` tool steps (each an assistant message with whatever follows it up to the next
        assistant message, so tool calls and their results stay paired) and drop the steps between.
        Always shrinks or returns False, unlike emergency compaction, whose rescue suffix can be the
        whole transcript (51 messages became 52 in the contract probe)."""
        starts = [i for i, m in enumerate(self.history) if self._is_frame_user_message(m)]
        if not starts:
            return False
        f = starts[-1]
        tail = self.history[f + 1:]
        steps = [i for i, m in enumerate(tail) if m.get("role") == "assistant"]
        keep = 1 if force else keep_steps
        if len(steps) <= keep:
            return False
        cut = steps[-keep]
        note = (f"[context drained: {len(steps) - keep} earlier tool step(s) of this turn were removed to fit "
                "the context window; their outputs are not repeated here. Re-run a probe if you need one.]")
        frame = dict(self.history[f])
        frame["content"] = [{"text": note}, *frame["content"]]
        self.history = [*self.history[:f], frame, *tail[cut:]]
        self._last_prompt_tokens = 0
        self.token_stats["context_drains"] = int(self.token_stats.get("context_drains", 0)) + 1
        self.reset_reasoning_chain()
        trace.append({"text": "live turn shrunk", "dropped_steps": len(steps) - keep,
                      "kept_messages": len(self.history), "force": force})
        return True

    @staticmethod
    def _is_context_limit_error(e: Exception) -> bool:
        return is_context_limit_error(e)

    def _recover_from_context_error(self, e: Exception, trace: list[dict]) -> bool:
        if not self.context_emergency_compaction or not self._is_context_limit_error(e):
            return False
        if self.context_mode == "drain" and self._drain_history(trace, force=True):
            return True
        if (
            self._reasoning_continuity
            and self._prev_response_id is not None
            and not self._reasoning_chain_rollover_pending
        ):
            self._rollover_reasoning_chain(
                f"context-limit error in opaque chain: {type(e).__name__}: {e}", trace
            )
            return True
        if (
            self._last_context_compaction_at == (self.level, self.turn_in_level)
            and len(self.history) <= self._last_context_compaction_history_len
        ):
            return False
        self._compact_context_for_emergency(f"context-limit error: {type(e).__name__}: {e}", trace)
        return True

    def _compact_context_for_emergency(self, reason: str, trace: list[dict]) -> None:
        """Replace the live transcript with a self-consistent short seed for long-level recovery.

        This is deliberately not the normal retention policy. It fires only when a configured soft
        prompt-token threshold or a provider context-limit error says the current level is becoming
        unsafe. The durable workspace remains intact; the compact prompt points the model to it and
        preserves a contiguous recent suffix that starts at a real frame boundary. Keeping the suffix
        structurally intact is less confusing than cherry-picking old tool snippets from unrelated
        positions in the transcript.
        """

        same_position = self._last_context_compaction_at == (self.level, self.turn_in_level)
        self._context_compaction_streak = self._context_compaction_streak + 1 if same_position else 1
        keep_turns = max(
            1,
            self.context_emergency_keep_turns // (2 ** (self._context_compaction_streak - 1)),
        )
        suffix = self._context_rescue_suffix(keep_turns)
        if not suffix:
            latest_frame_msg = self._latest_frame_user_message()
            suffix = [latest_frame_msg] if latest_frame_msg is not None else []
        recent_actions = self._recent_action_digest(self.context_emergency_recent_actions)
        ld = f"level_{self.level}"
        useful_files = self._context_reset_useful_files(ld)
        text = (
            "[harness long-level context reset]\n"
            "Earlier transcript messages for this long level were removed to avoid a context-limit "
            "failure. This is not a game event, not an ARC reset, and not an action outcome. The "
            "workspace files, recorded frame files, and the contiguous recent transcript suffix below "
            "are authoritative.\n\n"
            f"Reason: {reason[:700]}\n"
            f"Current level/turn: level {self.level}, turn {self.turn_in_level}.\n"
            f"Recent committed actions this level: {recent_actions or '(none recorded)'}\n"
            f"Preserved suffix: last {keep_turns} frame-turn(s), starting at a "
            "valid frame message when available.\n"
            f"Useful files: {useful_files}.\n"
            "Continue from the latest frame in the suffix. If you need older evidence, read it from the "
            "workspace instead of relying on the removed transcript.\n"
        )
        self.history = [{"role": "user", "content": text}, *suffix]
        self._last_context_compaction_at = (self.level, self.turn_in_level)
        self._last_context_compaction_history_len = len(self.history)
        self._last_prompt_tokens = 0
        self.token_stats["context_compactions"] = int(self.token_stats.get("context_compactions", 0)) + 1
        self.reset_reasoning_chain()
        trace.append({"text": "long-level context reset applied", "reason": reason[:500],
                      "kept_messages": len(suffix)})

    def _context_reset_useful_files(self, ld: str) -> str:
        current = (
            f"{ld}/turn_{self.turn_in_level:03d}.txt, {ld}/turn_{self.turn_in_level:03d}.png, "
            f"and recent {ld}/diff_*.txt files"
        )
        if getattr(self._mode_spec, "wm_variant", "single") == "none":
            return f"notes/actor_beliefs.md, {current}"
        return f"notes/world_model.md, world_model.py, verify.py, plan.py, {current}"

    def _context_rescue_suffix(self, keep_turns: int) -> list[dict]:
        frame_indices = [
            i for i, msg in enumerate(self.history)
            if self._is_frame_user_message(msg)
        ]
        if not frame_indices:
            return []
        keep = max(1, int(keep_turns))
        start = frame_indices[-keep] if len(frame_indices) >= keep else frame_indices[0]
        return copy.deepcopy(self.history[start:])

    @staticmethod
    def _is_frame_user_message(msg: dict) -> bool:
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        return any(
            isinstance(part, dict) and str(part.get("text", "")).startswith("=== Level ")
            for part in content
        )

    def _latest_frame_user_message(self) -> dict | None:
        for msg in reversed(self.history):
            if self._is_frame_user_message(msg):
                return copy.deepcopy(msg)
        return None

    def _recent_action_digest(self, limit: int) -> str:
        turns = [
            t for t in (getattr(self.ws, "turns", []) or [])
            if int(t.get("level", -1)) == self.level and t.get("action")
        ][-max(1, limit):]
        bits = []
        for t in turns:
            action = t.get("action")
            if action == "ACTION6" and t.get("row") is not None and t.get("col") is not None:
                bits.append(f"t{int(t.get('turn', 0)):03d}:{action}(row={t.get('row')},col={t.get('col')})")
            else:
                bits.append(f"t{int(t.get('turn', 0)):03d}:{action}")
        return ", ".join(bits)

    def _recent_tool_result_digest(self, limit: int) -> str:
        if limit <= 0:
            return ""
        chunks: list[str] = []
        for msg in reversed(self.history):
            if msg.get("role") != "tool":
                continue
            for r in reversed(msg.get("results") or []):
                name = r.get("name") or "tool"
                out = _clip_trace(str(r.get("output", "")), cap=1200)
                chunks.append(f"- {name}: {out}")
                if len(chunks) >= limit:
                    return "\n".join(reversed(chunks))
        return "\n".join(reversed(chunks))

    def _run_until_action(self, available_actions):
        trace = []
        steps = 0
        extra_redecide = False
        defer_extra_used = False
        while steps < self.max_tool_steps or extra_redecide:
            if extra_redecide:
                extra_redecide = False
            else:
                steps += 1
            if self.calls >= self.max_calls:
                break
            # Optional aggressive trimming before each intra-turn call. This bounds long tool loops,
            # but can also remove probe context before the model reasons over it.
            if self.context_mode == "tail_evict" and self.evict_each_llm_call:
                self._cache_mode_evict_if_needed()
            self._prepare_context_before_call(trace)
            try:
                reply = self._chat("freeform")
            except Exception as e:  # noqa: BLE001
                if self._recover_from_context_error(e, trace):
                    continue
                raise
            self.calls += 1
            self._note_llm_reply(reply)
            calls = reply.get("tool_calls") or []
            if (not calls and self._is_length_stop(reply)
                    and self.retry_max_tokens > self.max_tokens
                    and self.calls < self.max_calls):
                trace.append({"text": "length stop without tool call; retrying with larger max_tokens",
                              "max_tokens": self.retry_max_tokens})
                # Retry the original visible prompt rather than chaining from an incomplete
                # response with no newly appended input. Some Responses gateways reject an empty
                # continuation; others merely continue the truncated answer instead of re-deciding.
                self.reset_reasoning_chain()
                try:
                    reply = self._chat("freeform_retry", max_tokens=self.retry_max_tokens)
                except Exception as e:  # noqa: BLE001
                    if self._recover_from_context_error(e, trace):
                        continue
                    raise
                self.calls += 1
                self._note_llm_reply(reply)
                calls = reply.get("tool_calls") or []
            if not calls:
                # model emitted only text — nudge it to use a tool / commit an action
                self.history.append({"role": "assistant", "content": reply.get("text", ""),
                                     "raw_reasoning": reply.get("raw_reasoning"),
                                     "reasoning_items": reply.get("reasoning_items")})
                self.history.append({"role": "user", "content":
                    "Use a tool to investigate, or commit a move with take_action."})
                trace.append({"text": _clip_trace(reply.get("text", ""))})
                continue
            self.history.append({"role": "assistant", "content": reply.get("text", ""),
                                 "tool_calls": calls, "raw_reasoning": reply.get("raw_reasoning"),
                                 "reasoning_items": reply.get("reasoning_items")})
            # Execute ALL calls and answer EVERY toolUse id in one toolResult message
            # (parallel tool use is common — leaving a sibling id unanswered desyncs the
            # API). take_action gets a synthetic ack and ENDS the turn. Names are flavored
            # per model, so match on the CANONICAL name.
            act_call = next((c for c in calls if canonical(c["name"]) == "take_action"), None)
            # If the SAME batch also invokes the builder, the builder's report can't be allowed to
            # influence this action (it's already chosen). Native-parallel-tool models emit both at
            # once; committing the stale same-batch action would blunt orchestrator mode (observed:
            # builder recommended ACTION2, committed action was the same-batch ACTION1). So when both
            # are present, run the builder, return its report, and DEFER the action — the actor
            # re-decides next iteration WITH the report in hand. Same for world_model edits and
            # planner runs: their feedback is precisely what should inform the next action.
            builder_in_batch = self._orchestrator and any(canonical(c["name"]) == "invoke_builder" for c in calls)
            wm_edit_in_batch = any(self._is_world_model_edit(c) for c in calls)
            planner_in_batch = any(self._is_planner_run(c) for c in calls)
            action_choice = None
            action_trace_args = None
            invalid_action_msg = ""
            if act_call is not None:
                action_choice, action_trace_args, invalid_action_msg = self._prepare_take_action(
                    act_call.get("input", {}), available_actions
                )
            defer_action = (
                act_call is not None
                and not invalid_action_msg
                and (builder_in_batch or wm_edit_in_batch or planner_in_batch)
            )
            results = []
            script_started = None   # (session, choice, step) once a run_python script played (TYCHO_PY_ACTIONS)
            for c in calls:
                cn = canonical(c["name"])
                if cn == "run_python" and getattr(self, "py_actions", False) and script_started is None:
                    out, script_started = self._run_python_acting(c.get("input") or {}, available_actions)
                    results.append({"id": c["id"], "name": c["name"], "output": out})
                    trace.append({"tool": cn, "args": c["input"], "result": _clip_trace(out),
                                  **({"acting": True} if script_started else {})})
                    continue
                if cn == "take_action" and script_started is not None:
                    results.append({"id": c["id"], "name": c["name"], "output": (
                        "NOT executed: a run_python script in this reply is playing instead.")})
                    trace.append({"tool": "take_action", "args": c.get("input", {}), "skipped": "script_acting"})
                    continue
                if cn == "take_action" and self.strict_receipts and c is not act_call:
                    # Only the first take_action of a reply runs. Saying "committed" for the others
                    # made the model believe moves were played that never happened.
                    results.append({"id": c["id"], "name": c["name"], "output": (
                        "NOT executed: only the first take_action in a reply runs. To play several "
                        "moves, pass them as one take_action actions=[...] sequence."
                        if self.batch_max > 0 else
                        "NOT executed: only the first take_action in a reply runs; commit one action per turn.")})
                    trace.append({"tool": "take_action", "args": c.get("input", {}), "skipped": "extra_take_action"})
                elif cn == "take_action":
                    if invalid_action_msg:
                        out = invalid_action_msg
                    elif builder_in_batch:
                        why = "You invoked the builder in the SAME batch as this action"
                    elif wm_edit_in_batch:
                        why = "You attempted to edit world_model.py in the SAME batch as this action"
                    else:
                        why = "You ran the planner in the SAME batch as this action"
                    results.append({"id": c["id"], "name": c["name"],
                                    "output": (
                                        out if invalid_action_msg else
                                        (f"{why}, so the action is NOT committed — read the "
                                         "tool feedback below, then take_action again."
                                         if defer_action else
                                         "Action committed. The resulting frame is shown next.")
                                    )})
                elif cn == "set_verbosity":
                    # agent-steered context control: it picks full/diff/none for the frame text
                    results.append({"id": c["id"], "name": c["name"], "output": self._set_verbosity(c["input"])})
                    trace.append({"tool": cn, "args": c["input"]})
                elif cn == "invoke_builder":
                    # orchestrator mode: hand off to the world-model builder subagent, return
                    # its report. (In single mode invoke_builder isn't in the tool set.)
                    if self._orchestrator:
                        out = self._invoke_builder(c.get("input", {}).get("reason", ""), reserve_calls=1)
                    else:
                        out = "(invoke_builder unavailable — this mode does not expose actor-pull builder calls)"
                    results.append({"id": c["id"], "name": c["name"], "output": out})
                    trace.append({"tool": cn, "args": c.get("input", {}), "result": _clip_trace(out)})
                elif self._is_builder_owned_world_model_edit(c):
                    out = self._world_model_edit_block_message(summary=False)
                    results.append({"id": c["id"], "name": c["name"], "output": out})
                    blocked_reason = (
                        "world_model_unavailable"
                        if self._mode_spec.wm_variant == "none"
                        else "builder_owned_world_model"
                    )
                    trace.append({"tool": cn, "args": c["input"], "blocked": blocked_reason})
                else:
                    out = self.exec.execute(c["name"], c["input"])
                    results.append({"id": c["id"], "name": c["name"], "output": out})
                    trace.append({"tool": cn, "args": c["input"], "result": _clip_trace(out)})
            if script_started is not None:
                session, s_choice, s_step = script_started
                if act_call is not None:   # a take_action earlier in the same reply: the script plays instead
                    for r in results:
                        if r["id"] == act_call["id"]:
                            r["output"] = "NOT executed: a run_python script in this reply is playing instead."
                self.history.append({"role": "tool", "results": results})
                self._prepared_batch = None
                self._prepared_script = (session, s_step)
                trace.append({"tool": "take_action", "args": s_step, "committed": True, "src": "script"})
                return s_choice, trace
            self.history.append({"role": "tool", "results": results})  # CLOSE the turn, always
            if act_call is not None and invalid_action_msg:
                trace.append({"tool": "take_action", "args": act_call.get("input", {}),
                              "invalid": True, "result": invalid_action_msg})
                continue
            if act_call is not None and not defer_action:
                trace.append({"tool": "take_action", "args": action_trace_args, "committed": True})
                return action_choice, trace
            if defer_action:
                trace.append({"tool": "take_action", "args": act_call["input"], "deferred": True})
                if steps >= self.max_tool_steps and not defer_extra_used and self.calls < self.max_calls:
                    # The deferral only helps if the actor gets to see the builder report and re-decide.
                    # Grant one extra LLM turn at the cap for that re-decision; do not commit the stale
                    # same-batch action we explicitly told the model was not committed.
                    extra_redecide = True
                    defer_extra_used = True
        if self.calls < self.max_calls:
            self.history.append({"role": "user", "content":
                "You have reached the per-turn tool-step cap without committing an action. "
                "Choose now: respond with exactly one take_action call. Do not run analysis tools, "
                "do not edit files, and do not call the planner in this final commit pass."})
            try:
                reply = self._chat("freeform_commit")
            except Exception as e:  # noqa: BLE001
                if self._recover_from_context_error(e, trace):
                    reply = self._chat("freeform_commit")
                else:
                    raise
            self.calls += 1
            self._note_llm_reply(reply)
            calls = reply.get("tool_calls") or []
            self.history.append({"role": "assistant", "content": reply.get("text", ""),
                                 "tool_calls": calls or None,
                                 "raw_reasoning": reply.get("raw_reasoning"),
                                 "reasoning_items": reply.get("reasoning_items")})
            if calls:
                act_call = next((c for c in calls if canonical(c["name"]) == "take_action"), None)
                action_choice = None
                action_trace_args = None
                invalid_action_msg = ""
                if act_call is not None:
                    action_choice, action_trace_args, invalid_action_msg = self._prepare_take_action(
                        act_call.get("input", {}), available_actions
                    )
                results = []
                for c in calls:
                    cn = canonical(c["name"])
                    if c is act_call:
                        results.append({"id": c["id"], "name": c["name"],
                                        "output": (
                                            invalid_action_msg
                                            or "Action committed. The resulting frame is shown next."
                                        )})
                    else:
                        results.append({"id": c["id"], "name": c["name"],
                                        "output": "(skip — final commit pass only accepts take_action)"})
                        trace.append({"tool": cn, "args": c.get("input", {}), "skipped": "final_commit"})
                self.history.append({"role": "tool", "results": results})
                if act_call is not None and invalid_action_msg:
                    trace.append({"tool": "take_action", "args": act_call.get("input", {}),
                                  "invalid": True, "final_commit": True,
                                  "result": invalid_action_msg})
                elif act_call is not None:
                    trace.append({"tool": "take_action", "args": action_trace_args,
                                  "committed": True, "final_commit": True})
                    return action_choice, trace
            else:
                trace.append({"text": "final commit pass produced no tool call"})
        # Cap hit without an action. Prefer a non-RESET, non-click legal action because RESET
        # restarts the attempt and ACTION6 requires a target; only click-only frames keep the
        # historical center fallback.
        non_reset_actions = [x for x in available_actions if x != GameAction.RESET]
        a = next((x for x in non_reset_actions if x != GameAction.ACTION6), None)
        if a is None:
            a = non_reset_actions[0] if non_reset_actions else (
                available_actions[0] if available_actions else GameAction.ACTION1
            )
        if a == GameAction.ACTION6:
            return ActionChoice(a, x=32, y=32, reasoning={"src": "default (click-only tool cap)"}), trace
        return ActionChoice(a, reasoning={"src": "default (tool cap)"}), trace

    def _prepare_take_action(self, args: dict, available_actions) -> tuple[ActionChoice | None, dict | None, str]:
        """Validate a take_action call. A multi-action `actions` list (when enabled) is validated as a
        whole; nothing is committed unless every entry is legal. A valid sequence of two or more is
        parked in _prepared_batch, and choose_action starts it only if its first choice is the one
        _run_until_action actually commits (not a deferred or superseded call)."""
        self._prepared_batch = None
        steps = args.get("actions") if isinstance(args, dict) else None
        if getattr(self, "batch_max", 0) > 0 and isinstance(steps, list) and steps:
            return self._prepare_sequence(steps, available_actions)
        return self._prepare_single_action(args, available_actions)

    def _prepare_sequence(self, steps: list, available_actions) -> tuple[ActionChoice | None, dict | None, str]:
        dropped = max(0, len(steps) - self.batch_max)
        steps = steps[:self.batch_max]
        n = len(steps)
        parsed: list[dict] = []
        trace_steps: list[dict] = []
        first = None
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return None, None, (
                    f"Invalid take_action sequence: entry {i + 1} of {n} is not an object like "
                    "{\"action\": \"ACTION1\"}. No action was committed.")
            if i > 0 and str(step.get("action", "")).upper() == "RESET":
                return None, None, (
                    f"Invalid take_action sequence: RESET at position {i + 1}. RESET may only be the "
                    "first action of a sequence. No action was committed.")
            choice, step_args, err = self._prepare_single_action(step, available_actions)
            if err:
                detail = err.removeprefix("Invalid take_action: ")
                return None, None, (
                    f"Invalid take_action sequence: action {i + 1} of {n}: {detail}")
            if first is None:
                first = choice
            parsed.append(step_args)
            trace_steps.append(step_args)
        if n > 1:
            self._prepared_batch = {"first": first, "steps": parsed, "dropped": dropped}
        return first, (trace_steps[0] if n == 1 else {"actions": trace_steps, "dropped": dropped}), ""

    def _prepare_single_action(self, args: dict, available_actions) -> tuple[ActionChoice | None, dict | None, str]:
        by_name = {a.name: a for a in available_actions}
        requested = str(args.get("action", "")).upper()
        if requested not in by_name:
            avail = ", ".join(a.name for a in available_actions) or "(none)"
            label = requested or "(missing)"
            return None, None, (
                f"Invalid take_action: {label} is not available on this frame. "
                f"Available actions are: {avail}. No action was committed; call take_action again "
                "with one available action."
            )
        if requested == "RESET" and getattr(self, "_last_committed_control", None) == "RESET":
            return None, None, (
                "Invalid take_action: repeated RESET blocked. Your previous committed environment "
                "control was RESET, so the current frame is already the fresh start of this same "
                "level/attempt. A second immediate RESET is redundant. No action was committed; "
                "inspect the reset frame if needed, then choose a non-RESET available action."
            )
        if requested == "ACTION6":
            row, col = args.get("row"), args.get("col")
            if not self._valid_click_coord(row) or not self._valid_click_coord(col):
                return None, None, (
                    "Invalid take_action: ACTION6 requires integer row and col in 0-63. "
                    "No action was committed; call take_action again with ACTION6 row/col "
                    "or another available action."
                )
        choice = self._to_choice(args, available_actions)
        trace_args = {"action": choice.action.name}
        if choice.action == GameAction.ACTION6:
            trace_args["row"] = getattr(choice, "y", None)
            trace_args["col"] = getattr(choice, "x", None)
        return choice, trace_args, ""

    # ---- multi-action sequences (TYCHO_BATCH_ACTIONS) -------------------------------------------
    # The first action of a sequence leaves through the normal turn path. Each later one is returned by
    # _continue_batch on the next choose_action call: the frame is recorded in the workspace as usual
    # (so wmlib.frames()/transitions() and world_model.py see every step), but no user message, image,
    # animation summary or LLM call is spent. The next real turn opens with a report of what ran.

    def _start_batch(self, prepared: dict, grid, game_avail: list[str]) -> None:
        steps = prepared["steps"]
        self._batch = {
            "queue": list(steps[1:]),
            "labels": [self._action_label(s["action"], row=s.get("row"), col=s.get("col")) for s in steps],
            "planned": len(steps),
            "executed": 1,  # the first action is returned by this turn
            "dropped": prepared.get("dropped", 0),
            "start_turn": self.turn_in_level,
            "start_level": self.level,
            "start_grid": [list(map(int, row)) for row in grid],
            "prev_grid": [list(map(int, row)) for row in grid],
            "game_actions": sorted(game_avail),
        }

    def _batch_guard(self, latest_frame, avail: list[str], game_avail: list[str]) -> str | None:
        """Why the running sequence must stop before its next action, or None to continue."""
        b = self._batch
        if int(latest_frame.levels_completed) > int(self.prev_levels_done):
            return "level_completed"
        state = str(latest_frame.state).replace("GameState.", "")
        if state != "NOT_FINISHED":
            return f"state_{state.lower()}"
        if sorted(game_avail) != b["game_actions"]:
            return "available_actions_changed"
        if b["queue"] and b["queue"][0]["action"] not in avail:
            return "next_action_unavailable"
        grid = latest_frame.frame[-1]
        if b.get("script") is not None:
            return None   # a script observes no-ops itself and follows no validated plan
        if (getattr(self, "_last_action", None) not in (None, "RESET")
                and self.level >= self.batch_noop_guard_from_level
                and not _interior_changed(b["prev_grid"], grid, self.batch_border)):
            return "no_op"
        if self._mode_spec.wm_variant != "none":
            hint = self.ws.validated_plan_hint(grid, level=self.level, turn=self.turn_in_level, available=avail)
            if hint and "diverged" in hint:
                return "plan_diverged"
        return None

    def _continue_batch(self, latest_frame, available_actions, avail: list[str], game_avail: list[str],
                        state: str) -> ActionChoice:
        grid = latest_frame.frame[-1]
        self.ws.record(grid, level=self.level, turn_in_level=self.turn_in_level,
                       action=getattr(self, "_last_action", None) if self.turn_in_level > 0 else None,
                       row=getattr(self, "_last_row", None), col=getattr(self, "_last_col", None),
                       state=state, available=avail)
        self.exec.cur_grid = [list(map(int, row)) for row in grid]
        self.exec.available = game_avail
        self.prev_levels_done = latest_frame.levels_completed
        b = self._batch
        b["prev_grid"] = [list(map(int, row)) for row in grid]
        step = b["queue"].pop(0)
        choice = self._to_choice(step, available_actions)
        choice.reasoning = {**(choice.reasoning or {}), "src": "batch",
                            "i": b["executed"], "n": b["planned"], "t": self.t, "calls": self.calls,
                            "level": self.level, "turn_in_level": self.turn_in_level,
                            "token_stats": dict(self.token_stats)}
        b["executed"] += 1
        self._last_action = choice.action.name
        self._last_committed_control = choice.action.name
        r = choice.reasoning.get("row")
        c = choice.reasoning.get("col")
        self._last_row = r if r is not None else getattr(choice, "y", None)
        self._last_col = c if c is not None else getattr(choice, "x", None)
        self.t += 1
        self.turn_in_level += 1
        return choice

    # ---- run_python scripts that play (TYCHO_PY_ACTIONS) ----
    def _run_python_acting(self, args: dict, available_actions) -> tuple[str, tuple | None]:
        """Run a run_python call that may call action(...). Returns (tool output, None) if the script ended
        without playing, or (a 'playing' notice, (session, choice, step)) once it requested a legal action:
        that action ends the decision and the script keeps playing from the next frame."""
        code = args.get("code", "") if isinstance(args, dict) else ""
        if not code.strip() or "action(" not in code:
            return self.exec.execute("run_python", args), None
        try:
            timeout = max(1, min(300, int(args.get("timeout_s", 15))))
        except (TypeError, ValueError):
            timeout = 15
        session = self.exec.start_script(code)
        session.step_timeout = timeout
        while True:
            ev = session.next_event(timeout)
            if "action" not in ev:
                out = session.finish("it ran too long" if ev.get("timeout") else None)
                if not ev.get("timeout") and "Traceback (most recent call last)" not in out:
                    self.exec.retain(code)
                if ev.get("timeout"):
                    out = f"(run_python did not finish within {timeout}s)\n{out}"
                return self.exec._portable_output(out or "(no output)"), None
            step = {k: v for k, v in (("action", ev.get("action")), ("row", ev.get("row")),
                                      ("col", ev.get("col"))) if v is not None}
            choice, step_args, err = self._prepare_single_action(step, available_actions)
            if not err and step_args["action"] == "RESET":
                err = "RESET cannot be played from a script; end the script and use take_action RESET."
            if err:
                session.send({"error": err})
                continue
            label = self._action_label(step_args["action"], row=step_args.get("row"), col=step_args.get("col"))
            return (f"Script is playing: it played {label} and goes on from the next frame (at most "
                    f"{self.py_actions_max} actions). This ends your turn; what it prints and the actions it "
                    "plays are reported with the frame after it ends."), (session, choice, step_args)

    def _start_script_batch(self, session, step: dict, grid, game_avail: list[str]) -> None:
        self._batch = {
            "script": session,
            "queue": [],
            "labels": [self._action_label(step["action"], row=step.get("row"), col=step.get("col"))],
            "planned": self.py_actions_max,
            "executed": 1,  # the first action is returned by this turn
            "dropped": 0,
            "start_turn": self.turn_in_level,
            "start_level": self.level,
            "start_grid": [list(map(int, row)) for row in grid],
            "prev_grid": [list(map(int, row)) for row in grid],
            "game_actions": sorted(game_avail),
        }

    def _script_next(self, latest_frame, avail: list[str], available_actions, state: str):
        """Send the script the frame its last action produced; return (next step, None) or (None, stop)."""
        b = self._batch
        session = b["script"]
        if session.actions >= self.py_actions_max:
            return None, "action_limit"
        grid = [list(map(int, row)) for row in latest_frame.frame[-1]]
        change = diff_text(b["prev_grid"], grid).splitlines()
        changed = change[0] + ("; " + "; ".join(s.strip() for s in change[1:4]) if len(change) > 1 else "")
        if getattr(self, "object_diff", False) and change[0] != "no cells changed":
            changed = (self._object_changes(b["prev_grid"], grid) or "") + " | " + change[0]
        session.send({"obs": {"grid": grid, "state": state, "levels_completed": int(latest_frame.levels_completed),
                              "changed": changed, "available": list(avail)}})
        while True:
            ev = session.next_event(getattr(session, "step_timeout", 15))
            if "action" not in ev:
                return None, ("script_timeout" if ev.get("timeout") else "script_done")
            step = {k: v for k, v in (("action", ev.get("action")), ("row", ev.get("row")),
                                      ("col", ev.get("col"))) if v is not None}
            choice, step_args, err = self._prepare_single_action(step, available_actions)
            if not err and step_args["action"] == "RESET":
                err = "RESET cannot be played from a script; end the script and use take_action RESET."
            if err:
                session.send({"error": err})
                continue
            return step_args, None

    def _object_changes(self, prev, cur) -> str:
        """TYCHO_OBJECT_DIFF: frame_diff_text of two grids on one line ('' when unavailable)."""
        if prev is None or cur is None:
            return ""
        try:
            from tycho.workspace.wmlib_template import frame_diff, frame_diff_text

            text = frame_diff_text(frame_diff(prev, [list(map(int, r)) for r in cur]), max_items=4)
        except Exception:  # noqa: BLE001 - a perception aid must never break a turn
            return ""
        return "" if text == "no object changes" else text.replace("\n", "; ")

    def _object_changes_line(self, prev, cur, span: str) -> str:
        if not getattr(self, "object_diff", False):
            return ""
        objects = self._object_changes(prev, cur)
        return f" Object changes {span}: {objects}." if objects else ""

    def _finish_script_batch(self, b: dict, stop: str | None, grid) -> None:
        k = b["executed"]
        ran = b["labels"][:k]
        last = ran[-1] if ran else "?"
        session = b["script"]
        to_script = {"level_completed": "the level completed", "state_game_over": "the game ended (GAME_OVER)",
                     "action_limit": f"it played the limit of {self.py_actions_max} actions",
                     "available_actions_changed": "the available actions changed",
                     "script_timeout": "it computed too long between actions"}.get(stop)
        if stop is not None and stop.startswith("state_") and to_script is None:
            to_script = f"the game state became {stop.removeprefix('state_').upper()}"
        why = {None: "its code finished", "script_done": "its code finished",
               "level_completed": f"action {k} ({last}) completed the level",
               "state_game_over": f"action {k} ({last}) caused GAME_OVER and the level was reset"}.get(
            stop, f"the harness stopped it: {to_script or stop}")
        output = self.exec._portable_output(session.finish(to_script))
        if "Traceback (most recent call last)" not in output:
            self.exec.retain(getattr(session, "code", ""))
        text = f"[script report] Your run_python script played {k} action(s): {', '.join(ran)}. It ended because {why}."
        text += f"\nWhat it printed:\n{_clip_trace(output, cap=4000)}" if output else "\nIt printed nothing."
        same_attempt = stop not in ("level_completed", "state_game_over") and self.level == b["start_level"]
        if same_attempt and grid is not None and self.turn_in_level > b["start_turn"]:
            ld = f"level_{self.level}"
            text += (f"\nIntermediate frames: {ld}/turn_{b['start_turn'] + 1:03d}.txt to "
                     f"turn_{self.turn_in_level:03d}.txt."
                     + self._object_changes_line(b["start_grid"], grid, "over the whole script")
                     + " Cell changes over the whole script:\n"
                     + _clip_trace(diff_text(b["start_grid"], [list(map(int, row)) for row in grid]), cap=2500))
        self._batch_report = {"text": text, "data": {"planned": b["planned"], "executed": k,
                                                     "stop": stop or "script_done", "script": True}}

    def _finish_batch(self, stop: str | None, grid) -> None:
        """Close the running sequence and queue its report for the next turn's message."""
        b, self._batch = self._batch, None
        if b is None:
            return
        if b.get("script") is not None:
            return self._finish_script_batch(b, stop, grid)
        k, n = b["executed"], b["planned"]
        ran, skipped = b["labels"][:k], b["labels"][k:]
        last = ran[-1] if ran else "?"
        head = f"[sequence report] You committed a sequence of {n} actions."
        if b["dropped"]:
            head += (f" Only the first {self.batch_max} were accepted; the {b['dropped']} after them "
                     "were dropped and never executed.")
        skip_txt = (f" Actions {k + 1}-{n} ({', '.join(skipped)}) were SKIPPED and did not execute."
                    if skipped else "")
        if stop is None:
            body = f" All {n} executed: {', '.join(ran)}."
        elif stop == "level_completed":
            body = (f" Executed {k} of {n}: {', '.join(ran)}. Action {k} ({last}) completed the level."
                    f"{skip_txt}")
        elif stop == "game_over":
            body = (f" Executed {k} of {n}: {', '.join(ran)}. Action {k} ({last}) caused GAME_OVER and "
                    f"the level was reset.{skip_txt}")
        elif stop == "no_op":
            body = (f" Executed {k} of {n}: {', '.join(ran)}. Action {k} ({last}) EXECUTED but produced "
                    f"no change inside the board area; only the outer {self.batch_border} cells, where a "
                    f"timer or step counter usually sits, could have changed.{skip_txt} Do NOT update "
                    "your position estimate by counting submitted moves; re-locate from the current "
                    "frame before planning. Batching is still the right approach: the stop is what keeps "
                    "a wrong movement model from costing the whole sequence.")
        elif stop in ("available_actions_changed", "next_action_unavailable"):
            body = (f" Executed {k} of {n}: {', '.join(ran)}. After action {k} ({last}) the available "
                    f"actions changed, so the sequence stopped.{skip_txt}")
        elif stop == "plan_diverged":
            body = (f" Executed {k} of {n}: {', '.join(ran)}. After action {k} ({last}) the observed "
                    f"frame diverged from the validated plan's trajectory; re-plan from the current "
                    f"frame.{skip_txt}")
        else:
            body = f" Executed {k} of {n}: {', '.join(ran)}. Stopped ({stop}).{skip_txt}"
        text = head + body
        same_attempt = stop not in ("level_completed", "game_over") and self.level == b["start_level"]
        if same_attempt and grid is not None and self.turn_in_level > b["start_turn"]:
            ld = f"level_{self.level}"
            text += (f"\nIntermediate frames: {ld}/turn_{b['start_turn'] + 1:03d}.txt to "
                     f"turn_{self.turn_in_level:03d}.txt."
                     + self._object_changes_line(b["start_grid"], grid, "over the whole sequence")
                     + " Cell changes over the whole sequence:\n"
                     + _clip_trace(diff_text(b["start_grid"], [list(map(int, row)) for row in grid]), cap=2500))
        self._batch_report = {"text": text, "data": {"planned": n, "executed": k, "stop": stop,
                                                     "skipped": len(skipped), "dropped": b["dropped"]}}

    @staticmethod
    def _valid_click_coord(v) -> bool:
        return type(v) is int and 0 <= v <= 63

    @staticmethod
    def _is_world_model_edit(call: dict) -> bool:
        cn = canonical(call.get("name", ""))
        if cn not in ("write_file", "edit_file", "edit_function"):
            return False
        path = str((call.get("input") or {}).get("path", ""))
        return path.replace("\\", "/").split("/")[-1] in ("world_model.py", "worldmodel.py")

    def _is_builder_owned_world_model_edit(self, call: dict) -> bool:
        return not self._mode_spec.actor_writes_wm and self._is_world_model_edit(call)

    def _world_model_edit_block_message(self, *, summary: bool) -> str:
        if self._mode_spec.wm_variant == "none":
            if summary:
                return (
                    "(blocked: this mode does not use executable world-model files. "
                    "Write level insights to notes/level_<N>_insights.md and keep "
                    "direct-reasoning observations in notes/actor_beliefs.md.)"
                )
            return (
                "(blocked: this mode does not use executable world-model files. "
                "Keep observations in notes/actor_beliefs.md and act from observed frames.)"
            )
        if summary:
            return (
                "(blocked: in this mode world_model.py is builder-owned. "
                "Write level insights to notes/level_<N>_insights.md and "
                "model-relevant observations to notes/world_model.md.)"
            )
        return (
            "(blocked: in this mode world_model.py is builder-owned. "
            "Write actor observations to notes/actor_beliefs.md or use the builder report.)"
        )

    @staticmethod
    def _is_planner_run(call: dict) -> bool:
        if canonical(call.get("name", "")) != "run_python":
            return False
        code = str((call.get("input") or {}).get("code", "")).strip()
        if not code:
            return False
        if "\n" in code:
            return "plan.py" in code and "run_path" in code
        try:
            parts = shlex.split(code)
        except ValueError:
            return False
        if parts and parts[0] in ("python", "python3"):
            parts = parts[1:]
        return bool(parts and parts[0].replace("\\", "/").split("/")[-1] == "plan.py")

    def _reset_level_counters(self) -> None:
        """Per-level state of the stuck gate and the builder cap (TYCHO_BUILDER_*)."""
        self._level_tokens_out_start = int(self.token_stats.get("tokens_out", 0))
        self._level_builder_tokens_out = 0
        self._level_builder_runs = 0
        self._level_auto_builder_fires = 0
        self._level_deaths = 0
        self._level_resets = 0
        self._level_critic_runs = 0

    def _level_actor_tokens_out(self) -> int:
        """Tokens the actor generated on this level (builder output excluded, so a build cannot
        bring the next automatic one forward)."""
        return (int(self.token_stats.get("tokens_out", 0)) - getattr(self, "_level_tokens_out_start", 0)
                - getattr(self, "_level_builder_tokens_out", 0))

    def _stuck_builder_due(self) -> bool:
        """TYCHO_BUILDER_GATE=stuck: fire the builder once the level has cost another
        builder_stuck_tokens actor tokens, or another builder_stuck_deaths GAME_OVERs, since the last
        automatic run. On M0 a level not solved within ~60K tokens is solved only 30-40% of the time and
        the model keeps defending its first theory of the rules; a builder in a fresh context re-derives
        them from the recorded transitions instead of from the actor's transcript."""
        if self.builder_gate != "stuck" or self.builder is None or self.turn_in_level == 0:
            return False
        if self.calls >= self.max_calls:
            return False
        if self.builder_max_per_level and getattr(self, "_level_builder_runs", 0) >= self.builder_max_per_level:
            return False
        k = getattr(self, "_level_auto_builder_fires", 0) + 1
        if self.builder_stuck_tokens > 0 and self._level_actor_tokens_out() >= self.builder_stuck_tokens * k:
            return True
        return self.builder_stuck_deaths > 0 and getattr(self, "_level_deaths", 0) >= self.builder_stuck_deaths * k

    def _stuck_critic_due(self) -> bool:
        """TYCHO_STUCK_CRITIC: a review once the level has cost another stuck_critic_tokens actor tokens, 2
        GAME_OVERs or 3 RESETs since the last one, at most stuck_critic_max times per level."""
        if not getattr(self, "stuck_critic", False) or self.turn_in_level == 0 or self.calls >= self.max_calls:
            return False
        k = getattr(self, "_level_critic_runs", 0)
        if k >= self.settings.stuck_critic_max:
            return False
        need = k + 1
        return (self._level_actor_tokens_out() >= self.settings.stuck_critic_tokens * need
                or getattr(self, "_level_deaths", 0) >= 2 * need
                or getattr(self, "_level_resets", 0) >= 3 * need)

    def _run_stuck_critic(self, grid, game_avail: list[str]) -> None:
        """One call in a FRESH context (observations and notes, not the actor's transcript) that proposes
        alternative explanations and probes; its reply goes to the actor as advice. In 7 of 13 DeepSeek-V4
        games that never cleared a level, the agent saw the decisive change and explained it away."""
        self._level_critic_runs = getattr(self, "_level_critic_runs", 0) + 1
        notes = self.ws.notes_digest(
            [f"notes/level_{k}_insights.md" for k in range(max(0, self.level - 2), self.level)]
            + ["notes/actor_beliefs.md"], max_bytes_each=3000)
        prompt = render_prompt(
            "critic.user", level=self.level, tokens_k=self._level_actor_tokens_out() // 1000,
            deaths=getattr(self, "_level_deaths", 0), resets=getattr(self, "_level_resets", 0),
            actions=list(game_avail) or ["(none)"], notes=notes, log=self.ws.attempt_logs(self.level))
        content = [{"text": prompt}]
        png = self.ws.current_png(self.level, self.turn_in_level) if self.vision else None
        if png is not None:
            content.append({"image_png": png})
        else:
            content.append({"text": "Current board:\n" + grid_text([list(map(int, r)) for r in grid])})
        try:
            reply = chat_tools([{"role": "user", "content": content}], [], self.cfg,
                               system=render_prompt("critic.system"),
                               max_tokens=max(self.max_tokens, 8192), effort=self.effort, call_type="critic")
        except Exception as exc:  # noqa: BLE001 - advice is optional; never lose the turn over it
            print(f"[tycho] stuck critic failed: {type(exc).__name__}: {exc}", flush=True)
            return
        self.calls += 1
        usage = reply_usage(reply)
        for key in ("tokens_in", "tokens_out", "cache_read", "cache_write"):
            self.token_stats[key] = int(self.token_stats.get(key, 0)) + int(usage.get(key, 0))
        # not actor tokens: a review must not bring the next one (or the stuck builder) forward
        self._level_builder_tokens_out = getattr(self, "_level_builder_tokens_out", 0) + int(usage.get("tokens_out", 0))
        text = (reply.get("text") or "").strip()
        if text:
            self.history.append({"role": "user", "content": (
                "[fresh-eyes review: an independent reviewer saw only the recorded observations and your notes, "
                "not your reasoning. Advisory: you decide.]\n" + text)})

    def _run_stuck_builder(self) -> None:
        self._level_auto_builder_fires = getattr(self, "_level_auto_builder_fires", 0) + 1
        tokens_k = self._level_actor_tokens_out() // 1000
        reason = (
            f"[automatic: level {self.level} is stuck — {tokens_k}K actor tokens and "
            f"{getattr(self, '_level_deaths', 0)} GAME_OVER(s) on it without completing it.] Take a fresh "
            "look. Re-derive the mechanics from the recorded transitions, not from the actor's beliefs, "
            "and above all test the WIN CONDITION: does outcome() agree with every observed frame "
            "(no level_complete before the level ended)? Name one experiment that would tell the actor's "
            "current theory apart from the most plausible alternative, and put it in your report.")
        report = self._invoke_builder(reason, reserve_calls=1)
        self.history.append({"role": "user", "content":
            "[auto world-model builder fired — advisory, you decide the action]\n" + report})

    def _wm_status_line(self) -> str | None:
        """TYCHO_WM_STATUS_LINE: one line of exact-replay verification for the newest frame message.
        One sandboxed probe per turn (CPU only, no tokens); the line sits in the new message, so the
        cached prefix is untouched."""
        if not self.wm_status_line or self._mode_spec.wm_variant == "none" or self.turn_in_level == 0:
            return None
        if getattr(self, "exec", None) is None or getattr(self, "ws", None) is None:
            return None
        from tycho.agent.builder import _SEED_MARKER
        if _SEED_MARKER in self.ws.read_file("world_model.py", max_bytes=200000):
            return None  # still the seed: "reproduces 0/N" every turn would be noise
        from tycho.agent.wm_signal import wm_signal
        try:
            sig = wm_signal(self.exec)
        except Exception:  # noqa: BLE001 — a probe failure must never cost the turn
            return None
        stats = sig.get("stats")
        if stats is None:
            return None
        matched, evaluated, first = stats
        line = f"[world model] reproduces {matched}/{evaluated} observed changing transitions exactly"
        if first:
            line += f"; first divergence: {first}"
        outcome = sig.get("outcome_divergence")
        if outcome:
            line += f"; outcome() check: {outcome}"
        return line + "."

    def _invoke_builder(self, reason: str, *, reserve_calls: int = 0) -> str:
        """Orchestrator mode: run the world-model builder subagent (it builds/refines
        world_model.py and returns a compact advisory report). The builder's LLM passes count
        toward the whole-game ceiling (`self.calls`) and are tracked separately for orchestration
        analysis. Returns advisory report text; the actor still decides what to do."""
        if self.builder is None:
            return "(invoke_builder unavailable — not in orchestrator mode)"
        runs = getattr(self, "_level_builder_runs", 0)
        cap = getattr(self, "builder_max_per_level", 0)
        if cap and runs >= cap:
            return (f"World-model builder skipped: it already ran {runs} time(s) on this level (cap "
                    f"{cap}). Its latest report is in notes/world_model_report.md; "
                    "continue from it and the current world_model.py.")
        self._level_builder_runs = runs + 1
        remaining = self.max_calls - self.calls - max(0, reserve_calls)
        if remaining <= 0:
            return ("World-model builder skipped: no LLM-call budget remains after reserving "
                    f"{reserve_calls} actor call(s).")
        self.builder_invocations += 1
        try:
            report, btrace = self.builder.build(
                self.level,
                hint=reason,
                max_calls=remaining,
                available_actions=getattr(self, "_current_available_actions", None),
            )
        finally:
            self.calls += self.builder.calls  # account builder passes against the game ceiling
            self.builder.calls = 0
            _bstats = self.builder.take_token_stats()
            self._level_builder_tokens_out = (getattr(self, "_level_builder_tokens_out", 0)
                                              + int(_bstats.get("tokens_out", 0)))
            self._merge_token_usage(_bstats)
        self._last_builder_trace = btrace
        # Record this builder run with its trigger reason, tool trace, and report.
        self._turn_builder_runs.append({"reason": reason, "report": report, "trace": btrace})
        return ("World-model builder report (advisory — you decide the action):\n" + report)

    def _set_verbosity(self, args) -> str:
        """Agent-steered context control (the meta-tool): independent controls for what the
        harness inlines each turn — grid (full board text) and diff (exact, header summary, or off).
        Mutates self.show_grid / self.show_diff / self.diff_mode, which _frame_message reads. Either arg may be omitted
        (only the given one changes). The agent overrides the TYCHO_TEXT_GRID starting default per
        its read of the game (static game → grid off; busy game → both on)."""
        msgs = []
        if "grid" in args and args["grid"] is not None:
            v = str(args["grid"]).lower()
            if v not in ("on", "off"):
                return f"(set_verbosity: grid must be 'on' or 'off', got {args['grid']!r})"
            self.show_grid = v == "on"
            msgs.append(f"grid={v}")
        if "diff" in args and args["diff"] is not None:
            v = str(args["diff"]).lower()
            if v not in ("on", "summary", "off"):
                return (f"(set_verbosity: diff must be 'on', 'summary', or 'off', "
                        f"got {args['diff']!r})")
            self.diff_mode = v
            self.show_diff = v != "off"
            msgs.append(f"diff={v}")
        if not msgs:
            return "(set_verbosity: pass grid='on'|'off' and/or diff='on'|'summary'|'off')"
        return (f"verbosity updated ({', '.join(msgs)}). Now: full grid inlined="
                f"{self.show_grid}, diff mode={self.diff_mode}. The full grid and exact diff are always on "
                "disk / readable via run_python.")

    def _tools_with_available_actions(self, available: list[str]) -> list[dict]:
        """Constrain take_action to controls legal on the current frame.

        The turn header already exposes this set. Encoding the same fact in the tool schema prevents
        malformed free-form action strings from consuming another model call before validation.
        """
        if not available:
            return self._tools_spec
        specs = copy.deepcopy(self._tools_spec)
        for spec in specs:
            if canonical(spec.get("name", "")) == "take_action":
                props = spec["schema"]["properties"]
                props["action"]["enum"] = list(available)
                if "actions" in props:
                    props["actions"]["items"]["properties"]["action"]["enum"] = list(available)
                break
        return specs

    def _to_choice(self, args, available_actions) -> ActionChoice:
        by_name = {a.name: a for a in available_actions}
        act = by_name.get(str(args.get("action", "")).upper())
        if act is None:
            act = available_actions[0] if available_actions else GameAction.ACTION1
        if act == GameAction.ACTION6:
            # THE ONE translation point: the agent speaks (row, col); the engine's
            # ActionChoice needs x=col, y=row. Nowhere else is x/y agent-facing.
            rv, cv = args.get("row"), args.get("col")
            row = int(rv) if isinstance(rv, (int, float)) else 32
            col = int(cv) if isinstance(cv, (int, float)) else 32
            crow, ccol = max(0, min(63, row)), max(0, min(63, col))
            return ActionChoice(act, x=ccol, y=crow,
                                reasoning={"src": "take_action", "row": crow, "col": ccol})
        return ActionChoice(act, reasoning={"src": "take_action"})

    def _frame_message(self, grid, state, avail, frame_boundary: str | None) -> str:
        gt = grid_text([list(map(int, row)) for row in grid])  # inline, space-separated (#9/#16)
        ld = f"level_{self.level}"
        header = (
            f"=== Level {self.level}, turn {self.turn_in_level} "
            f"(state={state}; available actions={avail}) ===")
        # turn 0 has no within-level diff; later turns read the harness diff (capped). The turn-0
        # intro text is rendered from the explicit game/level/attempt boundary kind.
        if self.turn_in_level == 0:
            diff = ""
        else:
            diff = self.ws.read_file(
                f"{ld}/diff_{self.turn_in_level-1:03d}_{self.turn_in_level:03d}.txt", max_bytes=2500)
        # No-op feedback (#1): the harness diff is the literal "no cells changed" when nothing moved;
        # the template surfaces that as an explicit one-liner (always, even with diff off) so the agent
        # doesn't keep firing a no-op action; the verbose cell-change list respects show_diff.
        no_op = bool(diff) and diff.strip().startswith("no cells changed")
        prompt_diff = diff
        if diff and not no_op and getattr(self, "diff_mode", "on") == "summary":
            prompt_diff = diff.splitlines()[0]
        if getattr(self, "object_diff", False) and diff and not no_op:
            # TYCHO_OBJECT_DIFF: the change as objects (moved/rotated, recolored, resized, appeared), as in
            # Franzen's frame_diff, ahead of the exact cell changes
            objects = self._object_changes(self.ws.level_grid(self.level, self.turn_in_level - 1), grid)
            if objects:
                prompt_diff = f"Objects: {objects}\nCells: {prompt_diff}"
        plan_hint = None
        if self._mode_spec.wm_variant != "none":
            plan_hint = self.ws.validated_plan_hint(
                grid,
                level=self.level,
                turn=self.turn_in_level,
                available=avail,
                batch_max=self.batch_max if self.plan_batch_hint else 0,
            )
            status = self._wm_status_line()
            if status:
                plan_hint = f"{status}\n\n{plan_hint}" if plan_hint else status
        # TYCHO_LEVEL_START_INLINE: turn 0 (game start, new level, restarted attempt) shows the grid
        # even when text_grid=diff, and a new level inlines the carry-forward notes of the last three
        # levels, so the first decision does not spend tool calls fetching them (DeepSeek-V4 spent 41%
        # of its output tokens on level-start decisions that re-read notes, turn files and the grid).
        inline_start = getattr(self, "level_start_inline", False) and self.turn_in_level == 0
        carry_forward = None
        if inline_start and frame_boundary == "level_start" and getattr(self, "ws", None) is not None:
            notes = [f"notes/level_{k}_insights.md" for k in range(max(0, self.level - 3), self.level)]
            carry_forward = "\n\n".join(filter(None, [
                self.ws.notes_digest(notes, max_bytes_each=6000),
                self.ws.notes_digest(["notes/actor_beliefs.md"], max_bytes_each=3000)])) or None
            if getattr(self, "level_start_probe", False) and self.level > 0:
                # TYCHO_LEVEL_START_PROBE: in 4 of 12 games stuck on level 2 the carried rules were right but
                # the new level needed a mechanic the agent never found; list what is new on the board.
                from tycho.workspace.workspace import board_novelty

                prev = self.ws.level_grid(self.level - 1)
                novelty = board_novelty(prev, [list(map(int, r)) for r in grid]) if prev else []
                head = f"### New on this board (compared with level {self.level - 1}'s first board)\n"
                body = ("\n".join(f"- {line}" for line in novelty) if novelty else
                        "- no new colors or small shapes; expect the same mechanics, possibly at a larger scale")
                probe = (head + body + "\nBefore reusing the earlier rules, probe what is new here and the actions "
                         "the notes mark untested: one batch that tries each once costs a single turn. A level "
                         "that adds objects usually adds a mechanic that the earlier goal now depends on.")
                carry_forward = f"{carry_forward}\n\n{probe}" if carry_forward else probe
        # Assemble from actor.frame.j2 — the grid/diff/no-op/vision conditionals are {% if %} blocks
        # in the template now (were inline string-building here). turn-0 intro is handled in-template.
        return render_prompt(
            "actor.user", header=header, level=self.level, turn_in_level=self.turn_in_level,
            frame_boundary=frame_boundary, ld=ld, grid_text=gt,
            wm_variant=self._mode_spec.wm_variant,
            show_grid=self.show_grid or inline_start, show_diff=self.show_diff, vision=self.vision,
            diff=prompt_diff, no_op=no_op, last_action=getattr(self, "_last_action", None),
            plan_hint=plan_hint, carry_forward=carry_forward,
            turn3=f"{self.turn_in_level:03d}")

    def _truncate_old_grids(self):
        """Bound context so long games don't overflow the model window (HTTP 400 /
        'input length exceeds maximum context'). Two heavy, accumulating payloads:
          - the 64x64 hex GRID text (~4KB) we inject each turn,
          - the rendered PNG IMAGE (~280 vision tokens each — invisible to a text-length
            check, the bug that overflowed Gemma at ~turn 100).
        Keep both only for the most recent `grid_keep` turns; elide older ones to a
        text breadcrumb (the frames stay on disk — the agent re-reads them by tool).
        Images are the dominant cost, so we elide by IMAGE recency, not text size."""
        # The H017 grid-embeds block is ~57MB; keeping >1 in the conversation blows past
        # vLLM's request-body limit ("Invalid HTTP request"). Strip it from EVERY turn but
        # the most recent, regardless of grid_keep (the grid is on disk + in the text channel).
        emb_idx = [i for i, m in enumerate(self.history) if _has_grid_embeds(m)]
        for i in emb_idx[:-1]:
            _strip_grid_embeds(self.history[i])
        img_idx = [i for i, m in enumerate(self.history) if _has_image(m)]
        for i in img_idx[:-self.grid_keep] if len(img_idx) > self.grid_keep else []:
            _strip_image(self.history[i])
        # also elide large hex-text frame/diff messages older than grid_keep
        big = [i for i, m in enumerate(self.history) if _is_elidable_text_block(m)]
        for i in big[:-self.grid_keep] if len(big) > self.grid_keep else []:
            _elide_text(self.history[i])
        self._truncate_old_reasoning()

    def _truncate_old_reasoning(self):
        """Bound replayable provider reasoning carried in the client transcript.

        Anthropic ``raw_reasoning`` measured 76-88% of retained tokens in long games; GPT's opaque
        ``reasoning_items`` are similarly replayed on every stateless Responses call. Keep either form
        only on the most recent ``reasoning_keep`` assistant turns. ``keep>=1`` preserves the active
        tool cycle's signature/item-ordering requirement.
        """
        keep = max(1, self.reasoning_keep)
        idx = [i for i, m in enumerate(self.history) if _has_reasoning(m)]
        for i in (idx[:-keep] if len(idx) > keep else []):
            _strip_reasoning(self.history[i])

    def _note_llm_reply(self, reply: dict) -> None:
        usage = reply_usage(reply)
        out = usage["tokens_out"]
        fresh_in = usage["tokens_in"]
        cache_read = usage["cache_read"]
        cache_write = usage["cache_write"]
        prompt_total = fresh_in + cache_read + cache_write
        self._last_prompt_tokens = prompt_total
        if (
            self._reasoning_chain_rollover_pending
            and (
                self.context_emergency_soft_tokens <= 0
                or prompt_total < self.context_emergency_soft_tokens
            )
        ):
            # The fresh full-history request fits comfortably; subsequent calls may extend its new
            # opaque chain until the soft limit is reached again.
            self._reasoning_chain_rollover_pending = False
        self._merge_token_usage(usage)
        self.token_stats["max_tokens_out"] = max(self.token_stats["max_tokens_out"], out)
        self.token_stats["max_prompt_tokens"] = max(
            int(self.token_stats.get("max_prompt_tokens") or 0),
            prompt_total,
        )
        if self._is_length_stop(reply):
            self.token_stats["length_stops"] += 1

    def _merge_token_usage(self, usage: dict[str, int]) -> None:
        for key in ("tokens_in", "tokens_out", "cache_read", "cache_write", "latency_ms"):
            self.token_stats[key] = int(self.token_stats.get(key, 0)) + int(usage.get(key, 0))

    @staticmethod
    def _is_length_stop(reply: dict) -> bool:
        stop = str(reply.get("stop") or "").lower()
        return "length" in stop or "max_token" in stop

    def _cache_mode_evict_if_needed(self):
        """Cache-preserving context mode.

        Keep text/tool history append-only so vLLM can reuse the stable prefix. Only heavy
        image/prompt-embed payloads are evicted, and only in batches once the count crosses a
        threshold. That still causes an occasional re-prefill from the eviction point, but avoids
        legacy's every-turn prefix mutation.
        """
        keep = max(0, self.cache_image_keep)
        threshold = max(keep + 1, keep * self.cache_image_evict_mult + 1)

        # Grid prompt-embeds are huge request-body payloads; keep at most the latest one even in
        # cache mode. Production configurations normally leave TYCHO_GRID_EMBEDS off.
        emb_idx = [i for i, m in enumerate(self.history) if _has_grid_embeds(m)]
        for i in emb_idx[:-1]:
            _strip_grid_embeds(self.history[i])

        img_idx = [i for i, m in enumerate(self.history) if _has_image(m)]
        if len(img_idx) >= threshold:
            for i in img_idx[:-keep] if keep else img_idx:
                _strip_image(self.history[i])

        # HARD CAP backstop: batched eviction fires only at `threshold` (which can exceed the server's
        # per-prompt image limit), so after it runs, force the surviving IMAGE PAYLOAD count down to
        # `cache_image_cap` by stripping the oldest image-bearing messages. Count payloads, not
        # messages: GAME_OVER and animation evidence can carry multiple PNGs in one user message.
        # Without this the request exceeds --limit-mm-per-prompt image and vLLM returns a 400.
        cap = max(0, self.cache_image_cap)
        while _image_payload_count(self.history) > cap:
            stripped = False
            for m in self.history:
                if _has_media_payload(m):
                    _strip_image(m)
                    stripped = True
                    break
            if not stripped:
                break

        # HARD CAP for verbose text/tool round-trips. In Qwen single-mode dev runs, assistant
        # tool-call inputs plus tool outputs were 89% of retained text and drove 262k-context 400s.
        # Keep the latest blocks intact as working memory; older details live on disk or can be
        # re-derived with tools, so replace them with breadcrumbs.
        text_cap = max(0, self.cache_text_keep)
        if text_cap > 0:
            big = [i for i, m in enumerate(self.history) if _is_elidable_text_block(m)]
            if len(big) > text_cap:
                for i in big[:len(big) - text_cap]:
                    _elide_text(self.history[i])

        # Provider reasoning payloads are not prefix-cache-stable and dominate long contexts. Evict
        # Anthropic raw_reasoning and GPT reasoning_items on the same policy in every context mode.
        self._truncate_old_reasoning()


# drain-mode estimate of one image's prompt cost (a 640 px board is ~400 visual tokens on Qwen3.8)
_DRAIN_IMAGE_TOKENS = 400


def _interior_changed(prev, cur, border: int) -> bool:
    """Whether any cell more than `border` cells from the edge differs between two grids.

    A timer or step-counter bar along the edge changes on every action, including moves into a wall;
    the interior test separates those no-ops from real gameplay changes (Franzen's
    `_interior_gameplay_changed`). A shape change counts as a change."""
    rows_prev, rows_cur = list(prev), list(cur)
    if len(rows_prev) != len(rows_cur):
        return True
    n = len(rows_cur)
    for r in range(border, n - border):
        a, b = list(rows_prev[r]), list(rows_cur[r])
        if len(a) != len(b):
            return True
        if [int(x) for x in a[border:len(a) - border]] != [int(x) for x in b[border:len(b) - border]]:
            return True
    return False


def _is_heavy(p) -> bool:
    """A heavy attachment block on a user turn: rendered media OR the H017 grid-embeds
    block (~57MB — the single largest payload; replaying it across turns is catastrophic)."""
    return (
        "image_png" in p
        or "video_jpeg_frames" in p
        or "video_gif" in p
        or p.get("type") in ("image_url", "video_url")
        or "grid_embeds_part" in p
    )


def _supports_animation_video(cfg) -> bool:
    """Use video only on served models/stacks where Tycho can send vLLM video_url parts."""

    return cfg.backend == "openai" and "qwen" in (cfg.model or "").lower()


def _has_grid_embeds(m) -> bool:
    c = m.get("content")
    return isinstance(c, list) and any("grid_embeds_part" in p for p in c)


def _strip_grid_embeds(m):
    """Drop ONLY the ~57MB grid-embeds block from a turn (keep its text + image). A breadcrumb
    isn't needed: the exact grid is on disk and in the text channel for that turn."""
    c = m.get("content")
    if isinstance(c, list):
        m["content"] = [p for p in c if "grid_embeds_part" not in p]


def _has_image(m) -> bool:
    c = m.get("content")
    if isinstance(c, list) and any(_is_heavy(p) for p in c):
        return True
    return any(r.get("image_png") is not None for r in m.get("results", []) or [])


def _has_media_payload(m) -> bool:
    c = m.get("content")
    if isinstance(c, list) and _content_media_payload_count(c) > 0:
        return True
    return any(r.get("image_png") is not None for r in m.get("results", []) or [])


def _image_payload_count(history: list[dict]) -> int:
    n = 0
    for m in history:
        c = m.get("content")
        if isinstance(c, list):
            n += _content_media_payload_count(c)
        n += sum(1 for r in m.get("results", []) or [] if r.get("image_png") is not None)
    return n


def _content_media_payload_count(parts: list[dict]) -> int:
    return sum(
        1 for p in parts
        if isinstance(p, dict) and (
            "image_png" in p
            or "video_jpeg_frames" in p
            or "video_gif" in p
            or p.get("type") in ("image_url", "video_url")
        )
    )


def _strip_image(m):
    """Drop the heavy attachment block(s) — PNG image and/or the H017 grid-embeds block —
    from an old turn, leaving a text breadcrumb. Never emit an empty text block because some
    provider content schemas reject it."""
    note = "[older frame image/grid dropped to bound context — read frames/turn_NNNN.png/.txt if needed]"
    c = m.get("content")
    if isinstance(c, list):
        kept = [p for p in c if not _is_heavy(p)]
        has_text = any(("text" in p and (p.get("text") or "").strip()) for p in kept)
        if not has_text:                       # only add the breadcrumb if no real text remains
            kept = [p for p in kept if (p.get("text") or "").strip() or "text" not in p]
            kept.append({"text": note})
        m["content"] = kept
    for r in m.get("results", []) or []:
        if r.get("image_png") is not None:
            r["image_png"] = None


def _has_reasoning(m) -> bool:
    return m.get("role") == "assistant" and bool(
        m.get("raw_reasoning") or m.get("reasoning_items")
    )


def _strip_reasoning(m):
    """Drop provider reasoning state from an OLD, closed assistant turn.

    Anthropic exposes signed ``raw_reasoning`` blocks; stateless GPT Responses exposes opaque
    ``reasoning_items``. Both are large replay payloads and follow the same recency policy. Keeping at
    least the current tool-use cycle preserves the provider's ordering/signature contract, while the
    visible transcript and durable workspace remain the agent's long-term memory.
    """
    m["raw_reasoning"] = None
    m["reasoning_items"] = None


def _msg_text_len(m) -> int:
    n = 0
    c = m.get("content")
    if isinstance(c, str):
        n += len(c)
    elif isinstance(c, list):
        n += sum(len(p.get("text", "")) for p in c)
    if m.get("results"):
        n += sum(len(str(r.get("output", ""))) for r in m["results"])
    if m.get("tool_calls"):
        for tc in m.get("tool_calls") or []:
            n += _nested_text_len(tc.get("input"))
    return n


def _is_elidable_text_block(m) -> bool:
    """Whether this message should count against TYCHO_TEXT_KEEP.

    Tool-heavy Qwen runs accumulate many medium-sized code/output blocks, so this cannot be based
    only on a large per-message threshold. Any closed assistant/tool round-trip is eligible; user
    frame text still needs the larger frame threshold.
    """
    role = m.get("role")
    if role == "assistant" and m.get("tool_calls"):
        return _msg_text_len(m) > _ELIDE_TOOL_TEXT_TRIGGER
    if role == "tool" and m.get("results"):
        return _msg_text_len(m) > _ELIDE_TOOL_TEXT_TRIGGER
    if role == "user":
        return _msg_text_len(m) > _ELIDE_FRAME_TEXT_TRIGGER
    return False


def _elide_text(m):
    note = "[older frame/tool text elided to bound context — re-read workspace files or rerun tools if needed]"
    if isinstance(m.get("content"), str):
        m["content"] = note
    elif isinstance(m.get("content"), list):
        m["content"] = [p for p in m["content"] if "image_png" in p or p.get("type") == "image_url"] + [{"text": note}]
    elif m.get("results"):
        for r in m["results"]:
            if len(str(r.get("output", ""))) > _ELIDE_TOOL_TEXT_TRIGGER:
                r["output"] = note
    if m.get("tool_calls"):
        for tc in m.get("tool_calls") or []:
            if isinstance(tc.get("input"), dict):
                tc["input"] = _elide_long_strings(tc["input"], note)


def _nested_text_len(v) -> int:
    if isinstance(v, str):
        return len(v)
    if isinstance(v, dict):
        return sum(_nested_text_len(x) for x in v.values())
    if isinstance(v, list):
        return sum(_nested_text_len(x) for x in v)
    return 0


def _elide_long_strings(v, note: str):
    if isinstance(v, str):
        return note if len(v) > _ELIDE_TOOL_TEXT_TRIGGER else v
    if isinstance(v, dict):
        return {k: _elide_long_strings(x, note) for k, x in v.items()}
    if isinstance(v, list):
        return [_elide_long_strings(x, note) for x in v]
    return v


def build_agent() -> Agent:
    return TychoAgent()
