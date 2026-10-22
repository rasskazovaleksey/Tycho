"""Single source of truth for Tycho's context-management + prompt-caching config.

WHY THIS EXISTS (the bug class it kills): caching was governed by TWO independent, badly-named env
vars that had to BOTH be set for frontier-model caching to actually work, and both defaulted to the
NON-caching value:
  - TYCHO_CONTEXT_MODE=legacy  (history STRUCTURE — 'legacy' mutates old turns each turn → cache-hostile)
  - LLM_PROMPT_CACHE=off       (whether provider cache-control markers are sent at all)
'legacy' was an overloaded catch-all: falling back to it silently changed history structure AND killed
caching AND (via 'rolling') could select a deprecated path — several side effects behind one word. A run
could "have caching on" in one sense and off in another, invisibly. The $6K no-cache H025 surprise was
exactly this: the run defaulted to legacy/no-cache and nothing surfaced it.

THE FIX:
  1. ORTHOGONAL, self-describing switches — history structure, prompt caching, and image retention are
     independent fields, no catch-all string that bundles them.
  2. CACHE-FRIENDLY DEFAULTS — prompt_caching ON + history=tail_evict (the prefix-preserving design from
     docs/context-caching-design.md), so caching is the default you'd have to deliberately turn OFF.
  3. RESOLVED CONFIG IS LOGGED + STAMPED (startup log + manifest) — a silent revert becomes immediately
     visible because the resolved values are printed and saved with every run.
  4. BACK-COMPAT — old env names still work (mapped here), so the other agent's existing invocations and
     the pinned eval clones don't break; the new names take precedence.

vLLM (Qwen/Gemma) auto-caches by prefix and ignores the Anthropic markers, so prompt_caching=on is
harmless there and tail_evict helps every backend (stable prefix → KV/prefix-cache reuse). The markers
specifically help providers with explicit prompt-cache controls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict

# ---- defaults (cache-friendly; the whole point) -------------------------------------------------
DEFAULT_PROMPT_CACHING = True          # send cache-control markers where the provider supports them
DEFAULT_HISTORY = "tail_evict"         # append-only, evict heavy payloads at the TAIL in batches
DEFAULT_CACHE_ROLLING_EVERY = 1        # 1 = historical "write a rolling cache point every call"

# history structures (the renamed TYCHO_CONTEXT_MODE). Names describe what they DO:
#   tail_evict — append-only transcript; evict old images/heavy text at the tail in batches. Prefix is
#                never mutated → cache-friendly. (was "cache")
#   truncate   — keep only the latest N grids/images/text by editing OLD turns every turn. Bounds
#                context but mutates the prefix → cache-HOSTILE. (was "legacy")
#   collapse   — DEPRECATED: collapse prior turns to just the committed action. Known RHAE regression.
#                (was "rolling") — kept only so old invocations resolve, not recommended.
#   drain      — append-only transcript that is never edited turn to turn (no image/reasoning/text
#                stripping). Once the prompt passes TYCHO_CONTEXT_BUDGET_TOKENS (or the image count
#                passes image_cap) the oldest whole frame-turns are dropped in ONE batch, down to
#                budget - TYCHO_CONTEXT_DRAIN_TOKENS. The prefix-cache discipline of Franzen's harness
#                (M0: 94% prompt-cache reuse vs 50-68% for tail_evict on Qwen, whose image cap stripped
#                an image every turn).
_HISTORY_VALUES = {"tail_evict", "truncate", "collapse", "drain"}

# drain-mode defaults: Franzen's harness on the same server (131072 window - 12288 reply - 512 margin;
# ARC3_CONTEXT_DRAIN_TOKENS).
DEFAULT_DRAIN_BUDGET_TOKENS = 118272
DEFAULT_DRAIN_TOKENS = 59392
_OLD_HISTORY_MAP = {"cache": "tail_evict", "legacy": "truncate", "rolling": "collapse"}


def _envbool(name: str):
    v = os.environ.get(name)
    if v is None:
        return None
    return v.lower() in ("1", "on", "true", "yes")


def prompt_caching_on() -> bool:
    """Whether to send prompt-cache markers. New name TYCHO_PROMPT_CACHING; falls back to the old
    LLM_PROMPT_CACHE; default ON. Single source — both the agent and llm_client call THIS so they
    can never disagree (the old split let the structure say 'cache' while markers stayed off)."""
    v = _envbool("TYCHO_PROMPT_CACHING")
    if v is not None:
        return v
    v = _envbool("LLM_PROMPT_CACHE")
    if v is not None:
        return v
    return DEFAULT_PROMPT_CACHING


def cache_rolling_every() -> int:
    """How often the rolling conversation cache point advances.

    1 preserves the historical behaviour: mark the latest stable prefix every call. Larger values
    bucket the rolling breakpoint by message index, which keeps the visible prompt identical but
    writes large cache prefixes less often. 0 disables the rolling conversation breakpoint while
    retaining tool/system cache markers.
    """
    raw = os.environ.get("TYCHO_CACHE_ROLLING_EVERY", str(DEFAULT_CACHE_ROLLING_EVERY))
    try:
        return max(0, int(raw))
    except ValueError as e:
        raise ValueError(f"TYCHO_CACHE_ROLLING_EVERY={raw!r} must be a non-negative integer") from e


def _resolve_history() -> str:
    """New name TYCHO_HISTORY (tail_evict|truncate|collapse|drain); falls back to the old
    TYCHO_CONTEXT_MODE (cache|legacy|rolling); default tail_evict. Unknown value → raise (no silent
    degrade to a catch-all)."""
    raw = os.environ.get("TYCHO_HISTORY")
    if raw:
        h = raw.lower()
        if h not in _HISTORY_VALUES:
            raise ValueError(f"TYCHO_HISTORY={raw!r} not in {sorted(_HISTORY_VALUES)}")
        return h
    old = os.environ.get("TYCHO_CONTEXT_MODE")
    if old:
        o = old.lower()
        if o not in _OLD_HISTORY_MAP:
            raise ValueError(f"TYCHO_CONTEXT_MODE={old!r} not in {sorted(_OLD_HISTORY_MAP)}")
        return _OLD_HISTORY_MAP[o]
    return DEFAULT_HISTORY


@dataclass(frozen=True)
class ContextConfig:
    """Fully-resolved context/caching config for one run. Every field is explicit — no mode string
    that bundles behaviours. Log .resolved() at startup + stamp it in the manifest so a silent
    revert is visible."""
    prompt_caching: bool
    history: str                 # tail_evict | truncate | collapse | drain
    image_keep: int              # images retained before tail-eviction kicks in
    image_evict_mult: int        # batched-eviction threshold multiplier
    image_cap: int               # HARD ceiling on images/request (<= server --limit-mm-per-prompt)
    text_keep: int               # heavy text/tool blocks retained (0 = off; unsafe for long Qwen runs)
    evict_each_llm_call: bool     # also trim before each intra-turn LLM call
    grid_keep: int               # truncate-mode: latest N grids/images kept
    reasoning_keep: int          # raw_reasoning turns kept (Opus context control)
    cache_rolling_every: int      # rolling cache point advances once per N messages; 0 disables it
    emergency_compaction: bool    # on ctx-limit failure, compact to a self-consistent short seed
    emergency_soft_tokens: int    # proactive compaction threshold; 0 = reactive-only
    emergency_keep_turns: int     # contiguous frame-turn suffix retained in emergency seed
    emergency_recent_actions: int # action trace retained in emergency seed
    emergency_tool_results: int   # recent tool outputs retained in emergency seed
    drain_budget_tokens: int      # drain mode: prompt tokens that trigger a drain
    drain_tokens: int             # drain mode: tokens freed below the budget by one drain

    def resolved(self) -> dict:
        return asdict(self)

    def warnings(self) -> list[str]:
        """Non-fatal config smells worth surfacing in the startup log."""
        w = []
        if self.prompt_caching and self.history == "truncate":
            w.append("prompt_caching=on with history=truncate: markers are wasted — truncate mutates "
                     "the prefix every turn, so the cache can't hit. Use history=tail_evict to benefit.")
        if not self.prompt_caching and self.history == "tail_evict":
            w.append("history=tail_evict but prompt_caching=off: you keep an append-only prefix but "
                     "send no explicit cache markers; automatic prefix caches may still apply.")
        if self.history == "drain" and self.drain_tokens >= self.drain_budget_tokens:
            w.append(f"drain_tokens={self.drain_tokens} >= drain_budget_tokens={self.drain_budget_tokens}: "
                     "every drain empties the level transcript down to the current turn.")
        if self.history == "collapse":
            w.append("history=collapse is DEPRECATED (known RHAE regression) — prefer tail_evict.")
        if self.image_keep > self.image_cap:
            w.append(f"image_keep={self.image_keep} > image_cap={self.image_cap}: the cap wins; "
                     f"the keep value above the cap has no effect.")
        if self.cache_rolling_every == 0:
            w.append("cache_rolling_every=0 disables the rolling conversation cache point; tools/system "
                     "may still be cached, but long actor prefixes will not be.")
        elif self.cache_rolling_every > 1:
            w.append(f"cache_rolling_every={self.cache_rolling_every}: visible prompt is unchanged, "
                     "but rolling cache writes advance in buckets to reduce cache-write spend.")
        return w


def resolve_context_config(grid_keep_default: int = 3) -> ContextConfig:
    """Resolve the full context config from the environment (new names, old-name fallback, defaults).
    grid_keep_default lets the caller pass the agent's TYCHO_GRID_KEEP-derived value."""
    grid_keep = int(os.environ.get("TYCHO_GRID_KEEP", str(grid_keep_default)))
    return ContextConfig(
        prompt_caching=prompt_caching_on(),
        history=_resolve_history(),
        # image retention: new TYCHO_IMAGE_* names already exist + are clear; keep them.
        image_keep=int(os.environ.get("TYCHO_IMAGE_KEEP", str(grid_keep))),
        image_evict_mult=max(1, int(os.environ.get("TYCHO_IMAGE_EVICT_MULT", "2"))),
        image_cap=int(os.environ.get("TYCHO_IMAGE_CAP", "4")),
        text_keep=int(os.environ.get("TYCHO_TEXT_KEEP", "0")),
        evict_each_llm_call=_envbool("TYCHO_EVICT_EACH_LLM_CALL") or False,
        grid_keep=grid_keep,
        reasoning_keep=int(os.environ.get("TYCHO_REASONING_KEEP", str(grid_keep))),
        cache_rolling_every=cache_rolling_every(),
        emergency_compaction=_envbool("TYCHO_CONTEXT_EMERGENCY_COMPACTION")
        if _envbool("TYCHO_CONTEXT_EMERGENCY_COMPACTION") is not None else True,
        emergency_soft_tokens=max(0, int(os.environ.get("TYCHO_CONTEXT_EMERGENCY_SOFT_TOKENS", "0") or 0)),
        emergency_keep_turns=max(1, int(os.environ.get("TYCHO_CONTEXT_EMERGENCY_KEEP_TURNS", "6") or 6)),
        emergency_recent_actions=max(1, int(os.environ.get("TYCHO_CONTEXT_EMERGENCY_RECENT_ACTIONS", "24") or 24)),
        emergency_tool_results=max(0, int(os.environ.get("TYCHO_CONTEXT_EMERGENCY_TOOL_RESULTS", "4") or 4)),
        drain_budget_tokens=max(1, int(os.environ.get("TYCHO_CONTEXT_BUDGET_TOKENS", str(DEFAULT_DRAIN_BUDGET_TOKENS))
                                       or DEFAULT_DRAIN_BUDGET_TOKENS)),
        drain_tokens=max(0, int(os.environ.get("TYCHO_CONTEXT_DRAIN_TOKENS", str(DEFAULT_DRAIN_TOKENS))
                                or DEFAULT_DRAIN_TOKENS)),
    )
