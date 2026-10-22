"""Tycho run-configuration registry, config-file bridge, and resolved manifest snapshot.

Configuration precedence is explicit environment variable, then YAML/JSON config value, then the
documented subsystem default. Config files fill only variables that the process has not already set.
The resolved snapshot records each effective value and whether it came from the environment, file,
or default; secrets record only whether they were set.

The `orchestration:` section carries multi-agent topology that cannot be represented by scalar
environment variables. YAML is imported lazily, so environment-only and JSON runs do not require it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfigKey:
    """One configuration knob. `env` is the environment variable a subsystem reads; `default` is the
    DOCUMENTED default (must match the subsystem's os.environ.get default — a test asserts this);
    `section` groups keys in the config file and run manifest; `secret` redacts the value in the
    manifest (record set/unset, never the value); `cast` parses a config-file value to the env string."""
    env: str
    section: str
    default: str
    help: str
    secret: bool = False

    def cast(self, v) -> str:
        """Config-file value (may be bool/int/str) -> the string form a subsystem's os.environ.get
        expects. bools -> the canonical '1'/'' the _envbool/`in (...)` checks accept."""
        if isinstance(v, bool):
            return "1" if v else "0"
        return str(v)


# THE REGISTRY. Grouped by section. `default` mirrors the subsystem's os.environ.get default so the
# snapshot can mark a field's provenance as 'default' when neither env nor file set it. Keep this in
# sync with the call sites (test_config_registry asserts the env names are all real reads).
CONFIG_KEYS: list[ConfigKey] = [
    # ---- model / backend ----
    ConfigKey("LLM_BACKEND", "model", "openai", "anthropic | openai_responses | openai"),
    ConfigKey("LLM_MODEL", "model", "", "provider model identifier"),
    ConfigKey("LLM_BASE_URL", "model", "", "OpenAI-compatible endpoint (vLLM)"),
    ConfigKey("LLM_API_KEY", "model", "", "API key", secret=True),
    ConfigKey("ANTHROPIC_API_KEY", "model", "", "Anthropic key (anthropic backend)", secret=True),
    ConfigKey("OPENAI_API_KEY", "model", "", "OpenAI key (openai_responses backend)", secret=True),
    ConfigKey("ANTHROPIC_THINKING", "model", "off", "off | enabled | adaptive"),
    ConfigKey("ANTHROPIC_CACHE_TTL", "model", "", "optional Anthropic cache TTL"),
    ConfigKey("LLM_EFFORT", "model", "", "reasoning effort (backend-level default)"),
    ConfigKey("LLM_HTTP_TIMEOUT", "model", "0", "per-call HTTP timeout s (0 = backend default)"),
    ConfigKey("LLM_RETRY_BUDGET_S", "model", "1200", "total retry/backoff budget per call (s)"),
    ConfigKey("OPENAI_REASONING_CONTINUITY", "model", "", "Responses previous_response_id chaining"),
    ConfigKey("OPENAI_STATELESS_REASONING", "model", "", "client-held encrypted-reasoning replay"),
    # ---- reasoning / budgets ----
    ConfigKey("TYCHO_EFFORT", "reasoning", "medium", "Tycho effort (overrides LLM_EFFORT)"),
    ConfigKey(
        "TYCHO_MAX_TOKENS",
        "reasoning",
        "24000",
        "base output-token budget; OpenAI Responses adds reasoning headroom",
    ),
    ConfigKey("TYCHO_MAX_TOKENS_RETRY", "reasoning", "0", "retry max_tokens (0 = same)"),
    ConfigKey("TYCHO_MAX_LLM_CALLS", "reasoning", "1100", "whole-game LLM-call ceiling"),
    ConfigKey("TYCHO_MAX_INFERENCE_COST_PER_GAME", "reasoning", "0", "public-list USD-equivalent inference budget per game (0 = off)"),
    ConfigKey("TYCHO_MAX_INFERENCE_COST_PER_LEVEL", "reasoning", "0", "public-list USD-equivalent inference budget per level (0 = off)"),
    ConfigKey("TYCHO_MAX_TOOL_STEPS", "reasoning", "25", "cognitive tool calls/turn before forced action"),
    # ---- multi-action commits (take_action `actions` list, executed with early-stop guards) ----
    ConfigKey("TYCHO_BATCH_ACTIONS", "actions", "0", "take_action accepts an `actions` list executed without further LLM calls"),
    ConfigKey("TYCHO_BATCH_MAX", "actions", "24", "most actions one take_action may commit"),
    ConfigKey("TYCHO_BATCH_NOOP_GUARD_FROM_LEVEL", "actions", "1", "0-based level from which a no-op action stops a batch"),
    ConfigKey("TYCHO_BATCH_BORDER", "actions", "4", "outer cells ignored by the batch no-op guard (timer/step bars)"),
    ConfigKey("TYCHO_STRICT_RECEIPTS", "actions", "0", "answer a 2nd take_action in one reply as NOT executed instead of acknowledging it as committed"),
    ConfigKey("TYCHO_WALLCLOCK_BUDGET_SECONDS", "runtime", "0", "whole-run wall-clock guard"),
    ConfigKey("TYCHO_PER_GAME_WALLCLOCK_BUDGET_SECONDS", "runtime", "0", "per-game wall-clock guard"),
    ConfigKey("TYCHO_SANDBOX_RUNTIME", "runtime", "auto", "auto | docker | finch"),
    ConfigKey("TYCHO_SANDBOX_IMAGE", "runtime", "tycho-python-sandbox:0.2", "workspace Python image"),
    # ---- orchestration ----
    ConfigKey("TYCHO_MODE", "orchestration", "single", "single | orchestrator | trigger | no_world_model"),
    ConfigKey("TYCHO_AGENT_GRAPH", "orchestration", "", "declarative agent-graph JSON (set from orchestration.agents); recorded for provenance"),
    ConfigKey("TYCHO_EDIT_FUNC", "orchestration", "", "add edit_function to world-model actors; builders always have it"),
    ConfigKey("TYCHO_BUILDER_MAX_STEPS", "orchestration", "40", "builder subagent step budget"),
    ConfigKey("TYCHO_BUILDER_GATE", "orchestration", "", "'' | stuck: also fire the builder automatically when a level is stuck (orchestrator)"),
    ConfigKey("TYCHO_BUILDER_STUCK_TOKENS", "orchestration", "40000", "stuck gate: generated tokens on the level per automatic builder run"),
    ConfigKey("TYCHO_BUILDER_STUCK_DEATHS", "orchestration", "2", "stuck gate: GAME_OVERs on the level per automatic builder run"),
    ConfigKey("TYCHO_BUILDER_MAX_PER_LEVEL", "orchestration", "0", "builder runs per level, actor-pulled or automatic (0 = unlimited)"),
    ConfigKey("TYCHO_BUILDER_MAX_TOKENS", "orchestration", "0", "output-token cap per builder call (0 = TYCHO_MAX_TOKENS)"),
    ConfigKey("TYCHO_WM_STATUS_LINE", "orchestration", "0", "one world-model verify line on each frame message"),
    ConfigKey("TYCHO_PLAN_BATCH_HINT", "orchestration", "0", "offer the remaining validated plan as one take_action sequence"),
    ConfigKey("TYCHO_BUILDER_STARTER", "orchestration", "", "add a first-model recipe to the builder kickoff while world_model.py is the seed"),
    ConfigKey("TYCHO_SCRIBE_MAX_CALLS", "orchestration", "3", "LLM calls the level-boundary summary may make"),
    ConfigKey("TYCHO_SCRIBE_MAX_TOKENS", "orchestration", "0", "output-token cap per level-summary call (0 = TYCHO_MAX_TOKENS)"),
    ConfigKey("TYCHO_WM_MIN_COVERAGE", "orchestration", "0.75", "trigger-mode minimum prediction_coverage before dynamics are considered clean"),
    # ---- vision / perception ----
    ConfigKey("TYCHO_VISION", "vision", "1", "attach current-frame PNG each turn"),
    ConfigKey("TYCHO_RENDER_SCALE", "vision", "", "px/cell override (empty = per-model profile)"),
    ConfigKey("TYCHO_TEXT_GRID", "vision", "full", "full | diff | none (in-prompt grid text)"),
    ConfigKey("TYCHO_GRID_FORMAT", "vision", "spaced_grid8", "grid text rendering format"),
    ConfigKey("TYCHO_TOOL_FLAVOR", "vision", "", "tool naming: claude|gemini|generic (auto if empty)"),
    ConfigKey("TYCHO_ANIMATION_SUMMARY", "vision", "1", "summarize selected information-rich transition animations"),
    ConfigKey("TYCHO_ANIMATION_SUMMARY_VIDEO", "vision", "1", "use video input for animation summaries when the served model supports it"),
    ConfigKey("TYCHO_ANIMATION_SUMMARY_VIDEO_FORMAT", "vision", "images", "animation visual transport: images | jpeg_frames | gif"),
    ConfigKey("TYCHO_ANIMATION_SUMMARY_MAX_PER_LEVEL", "vision", "5", "max new animation-summary LLM calls per level"),
    ConfigKey("TYCHO_ANIMATION_EVIDENCE_KEEP_PER_LEVEL", "vision", "50", "recent animation-frame records kept on disk per level"),
    ConfigKey("TYCHO_ANIMATION_SUMMARY_MAX_KEYFRAMES", "vision", "6", "max selected keyframes in transient contact sheet"),
    ConfigKey("TYCHO_ANIMATION_SUMMARY_INCLUDE_GAME_OVER", "vision", "1", "summarize selected GAME_OVER transition animations"),
    ConfigKey("TYCHO_ANIMATION_SUMMARY_MAX_TOKENS", "vision", "700", "max output tokens for transient animation summaries"),
    # ---- context / caching ----
    ConfigKey("TYCHO_PROMPT_CACHING", "context", "1", "send Anthropic cache markers"),
    ConfigKey("TYCHO_HISTORY", "context", "tail_evict", "tail_evict | truncate | collapse | drain"),
    ConfigKey("TYCHO_CONTEXT_BUDGET_TOKENS", "context", "118272", "drain mode: prompt tokens that trigger a batch drain of the oldest frame-turns"),
    ConfigKey("TYCHO_CONTEXT_DRAIN_TOKENS", "context", "59392", "drain mode: tokens freed below the budget by one drain"),
    ConfigKey("TYCHO_GRID_KEEP", "context", "3", "turns of grid/image kept in context"),
    ConfigKey("TYCHO_IMAGE_KEEP", "context", "3", "images retained before tail-eviction"),
    ConfigKey("TYCHO_IMAGE_EVICT_MULT", "context", "2", "batched-eviction threshold multiplier"),
    ConfigKey("TYCHO_IMAGE_CAP", "context", "4", "HARD images/request cap (<= server limit)"),
    ConfigKey("TYCHO_TEXT_KEEP", "context", "0", "heavy text/tool blocks retained (0 = off for append-only history)"),
    ConfigKey("TYCHO_EVICT_EACH_LLM_CALL", "context", "0", "trim tail-evict history before every intra-turn LLM call"),
    ConfigKey("TYCHO_REASONING_KEEP", "context", "3", "raw_reasoning turns kept (Opus)"),
    ConfigKey("TYCHO_CACHE_ROLLING_EVERY", "context", "1", "advance rolling cache point every N messages (0 = off)"),
    ConfigKey("TYCHO_CONTEXT_EMERGENCY_COMPACTION", "context", "1", "compact history and retry on context-limit errors"),
    ConfigKey("TYCHO_CONTEXT_EMERGENCY_SOFT_TOKENS", "context", "0", "proactive emergency compaction threshold; 0 = off"),
    ConfigKey("TYCHO_CONTEXT_EMERGENCY_KEEP_TURNS", "context", "6", "contiguous recent frame-turns retained in emergency compaction seed"),
    ConfigKey("TYCHO_CONTEXT_EMERGENCY_RECENT_ACTIONS", "context", "24", "recent actions retained in emergency compaction seed"),
    ConfigKey("TYCHO_CONTEXT_EMERGENCY_TOOL_RESULTS", "context", "4", "deprecated; context rescue now preserves a contiguous suffix instead of scattered tool snippets"),
    # ---- dev / misc ----
    ConfigKey("TYCHO_META_REFLECT", "dev", "", "solicit harness/prompt feedback (dev only)"),
    ConfigKey("RUN_HARDWARE", "dev", "", "operator-local runtime label"),
    ConfigKey("LLM_LOG", "dev", "", "capture per-call I/O (set by --viz)"),
]

PUBLIC_CONFIG_KEYS = tuple(CONFIG_KEYS)

# Deployments can extend the registry without changing the core configuration keys.
try:
    from tycho.config._config_extension import CONFIG_KEYS as _EXTENSION_CONFIG_KEYS
except ModuleNotFoundError:
    _EXTENSION_CONFIG_KEYS = []
# Extension keys are prepended so an explicitly installed extension can retain its established names.
CONFIG_KEYS[:0] = _EXTENSION_CONFIG_KEYS

_BY_ENV = {k.env: k for k in CONFIG_KEYS}
_BY_SECTION: dict[str, list[ConfigKey]] = {}
for _k in CONFIG_KEYS:
    _BY_SECTION.setdefault(_k.section, []).append(_k)


def redact(key: ConfigKey, value: str) -> str:
    """Never put a secret's value in the manifest — record only whether it was set."""
    if key.secret:
        return "<set>" if value else "<unset>"
    return value


# --------------------------------------------------------------------------------------------------
# config FILE -> env bridge
# --------------------------------------------------------------------------------------------------
def _load_file(path: Path) -> dict:
    """Parse a YAML or JSON config file to a nested dict. YAML is imported lazily (optional dep); a
    .json file uses the stdlib. A missing pyyaml on a .yaml file is a clear, actionable error."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # optional dep — only needed when a YAML config is actually used
        except ModuleNotFoundError as e:
            raise SystemExit(
                f"--config {path} is YAML but pyyaml is not installed. `pip install pyyaml`, or "
                f"convert the file to JSON (same structure).") from e
        return yaml.safe_load(text) or {}
    if path.suffix == ".json":
        return json.loads(text)
    raise SystemExit(f"--config {path}: unsupported extension (use .yaml/.yml/.json)")


def _flatten(cfg: dict) -> dict[str, object]:
    """A run-config dict maps section -> {key: value}. Flatten to {ENV_NAME: value} using the
    registry: a section's keys are matched to ConfigKeys whose env is TYCHO_<SECTION?>_<KEY> OR whose
    `section` == this section and whose env's trailing token matches the key (case-insensitive). To
    avoid guessing, the file may ALSO use full env names directly under a top-level `env:` block."""
    out: dict[str, object] = {}
    # 1. explicit env passthrough (escape hatch): env: {TYCHO_RENDER_SCALE: 32, ...}
    for name, v in (cfg.get("env") or {}).items():
        if name in _BY_ENV:
            out[name] = v
        else:
            raise SystemExit(f"--config env.{name}: not a known Tycho config key (see tycho.config.CONFIG_KEYS)")
    # 2. orchestration block: its SCALAR settings map to the orchestration env keys (so the file
    # actually DRIVES the run — previously it was parsed but silently ignored, a provenance trap).
    # The declarative `agents:` graph (a multi-agent topology) is NOT flattened to env (it can't
    # round-trip) — it's read separately by resolve_orchestration() and consumed by the agent's
    # _build_agent_graph.
    orch = cfg.get("orchestration") or {}
    if isinstance(orch, dict):
        sec_keys = _BY_SECTION.get("orchestration", [])
        for short, v in orch.items():
            if short == "agents":
                continue
            if isinstance(v, (dict, list)):
                continue
            ck = _match_key("orchestration", short, sec_keys)
            if ck is None:
                raise SystemExit(
                    f"--config orchestration.{short}: no matching config key. Known orchestration keys: "
                    + ", ".join(sorted(k.env for k in sec_keys)) + " (or use an `env:` block with full names)")
            out[ck.env] = v
    # 3. section blocks: {context: {history: tail_evict, grid_keep: 3}, vision: {render_scale: 32}}
    for section, keys in (cfg or {}).items():
        if section in ("env", "orchestration"):  # handled above
            continue
        if not isinstance(keys, dict):
            continue
        sec_keys = _BY_SECTION.get(section, [])
        for short, v in keys.items():
            ck = _match_key(section, short, sec_keys)
            if ck is None:
                raise SystemExit(
                    f"--config {section}.{short}: no matching config key. Known {section} keys: "
                    + ", ".join(sorted(k.env for k in sec_keys)) + " (or use an `env:` block with full names)")
            out[ck.env] = v
    return out


def _match_key(section: str, short: str, sec_keys: list[ConfigKey]) -> ConfigKey | None:
    """Map a short config-file key (e.g. 'render_scale' in section 'vision') to its ConfigKey by
    comparing the short name to the env's tokens after the TYCHO_/LLM_ prefix."""
    want = short.lower().replace("-", "_")
    for ck in sec_keys:
        toks = ck.env.lower().split("_", 1)[-1]  # drop the TYCHO_/LLM_ prefix
        if toks == want or ck.env.lower() == want or ck.env.lower().endswith("_" + want):
            return ck
    return None


def apply_config_file(path: str | Path) -> dict[str, str]:
    """Overlay a run-config file onto os.environ for keys the environment has NOT already set
    (env wins — existing launch scripts stay authoritative). Returns {ENV_NAME: applied_value} for
    the keys this file actually set, so the caller can log exactly what the file contributed."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--config {p}: file not found")
    flat = _flatten(_load_file(p))
    applied = {}
    for name, v in flat.items():
        if name in os.environ:
            continue  # env wins — do not override a value the operator set explicitly
        s = _BY_ENV[name].cast(v)
        os.environ[name] = s
        applied[name] = s
    return applied


def resolve_orchestration(path: str | Path | None) -> dict | None:
    """Return the file's `orchestration:` block (the declarative multi-agent graph / mode preset),
    or None if absent. Kept SEPARATE from the env bridge because a multi-agent graph can't round-trip
    through env vars — tycho/agent/modes.py consumes this directly. (Phase-2 graph wiring lands here.)

    Normalizes the YAML 1.1 `on:` gotcha at THIS boundary: a bare `on:` key parses to the boolean
    True (likewise off/yes/no). We rewrite each agent's trigger `{True: ...}` to `{"on": ...}` HERE so
    the block serializes cleanly to JSON (JSON keys are strings — a True key would become "true" and
    be lost on the round-trip into TYCHO_AGENT_GRAPH)."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    orch = (_load_file(p) or {}).get("orchestration")
    if isinstance(orch, dict):
        for a in (orch.get("agents") or []):
            trig = a.get("trigger") if isinstance(a, dict) else None
            if isinstance(trig, dict) and True in trig and "on" not in trig:
                trig["on"] = trig.pop(True)
    return orch


# --------------------------------------------------------------------------------------------------
# manifest snapshot (provenance per field)
# --------------------------------------------------------------------------------------------------
def _resolved_config(
    keys: list[ConfigKey] | tuple[ConfigKey, ...],
    *,
    config_file: str | Path | None = None,
    file_applied: dict[str, str] | None = None,
) -> dict:
    file_applied = file_applied or {}
    sections: dict[str, dict] = {}
    for ck in keys:
        source_key = ck.env
        if ck.env == "TYCHO_HISTORY" and ck.env not in file_applied and ck.env not in os.environ \
                and "TYCHO_CONTEXT_MODE" in os.environ:
            from tycho.agent.context_config import resolve_context_config
            src = "env"
            source_key = "TYCHO_CONTEXT_MODE"
            value = resolve_context_config().history
        elif ck.env == "TYCHO_PROMPT_CACHING" and ck.env not in file_applied and ck.env not in os.environ \
                and "LLM_PROMPT_CACHE" in os.environ:
            from tycho.agent.context_config import prompt_caching_on
            src = "env"
            source_key = "LLM_PROMPT_CACHE"
            value = "1" if prompt_caching_on() else "0"
        elif ck.env in file_applied:
            src = "file"
            value = os.environ.get(ck.env, ck.default)
        elif ck.env in os.environ:
            src = "env"
            value = os.environ.get(ck.env, ck.default)
        else:
            src = "default"
            value = ck.default
        entry = {"value": redact(ck, value), "source": src}
        if source_key != ck.env:
            entry["source_key"] = source_key
        sections.setdefault(ck.section, {})[ck.env] = entry
    return {"config_file": str(config_file) if config_file else None, "by_section": sections}


def resolved_config(config_file: str | Path | None = None,
                    file_applied: dict[str, str] | None = None) -> dict:
    """Resolve every installed configuration key for local execution and resume checks."""
    return _resolved_config(
        CONFIG_KEYS,
        config_file=config_file,
        file_applied=file_applied,
    )


_RECORDED_CONFIG_EXCLUDE = {
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "LLM_HTTP_TIMEOUT",
    "LLM_RETRY_BUDGET_S",
    "RUN_HARDWARE",
    "LLM_LOG",
}


def recorded_config(
    *,
    file_applied: dict[str, str] | None = None,
    api_protocol: str,
    model: str,
) -> dict:
    """Resolve the stable, provider-neutral configuration recorded with a run."""
    snapshot = _resolved_config(PUBLIC_CONFIG_KEYS, file_applied=file_applied)
    sections = snapshot["by_section"]
    for section in list(sections):
        entries = sections[section]
        for key in list(entries):
            if key in _RECORDED_CONFIG_EXCLUDE:
                del entries[key]
        if not entries:
            del sections[section]
    model_section = sections.setdefault("model", {})
    model_section["LLM_BACKEND"] = {"value": api_protocol, "source": "resolved"}
    model_section["LLM_MODEL"] = {"value": model, "source": "resolved"}
    return {"by_section": sections}
