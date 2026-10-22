"""Focused world-model builder used by orchestrator and trigger policies.

The builder owns `world_model.py`, reads grids and transitions from the shared workspace, receives
automatic verification feedback after edits, and writes an advisory report for the actor. Its
conversation is bounded and independent of the actor's durable history.
"""

from __future__ import annotations

import os

from tycho.serving.llm_client import (
    chat_tools,
    is_context_limit_error,
    response_id_continuity_enabled,
)
from tycho.serving.pricing import reply_usage
from tycho.workspace.agent_tools import canonical
from tycho.agent.colors import COLORMAP   # shared with the actor so they speak the same language
from tycho.prompts.render import render_prompt  # builder.system.j2 + builder.user.j2

REPORT_FILE = "notes/world_model_report.md"
# a line only the untouched seed world_model.py contains (templates/seed_world_model.py.tmpl)
_SEED_MARKER = '# e.g. if action["action"] == "ACTION1": <move something in s>'

# meta-reflection on the builder is gated by the same TYCHO_META_REFLECT flag as the actor (dev aid:
# the builder is where world-model-API friction surfaces, so its feedback is especially valuable).
_META_REFLECT = os.environ.get("TYCHO_META_REFLECT", "") in ("1", "on", "true")
_BUILDER_SYS = render_prompt("builder.system", colormap=COLORMAP, report_file=REPORT_FILE,
                             meta_reflect=_META_REFLECT)


# AUTO-ATTACHED VERIFY STATE — runs at the start of every build pass and is injected as a user
# message before the LLM's first reasoning step. Saves 2–4 tool calls per fire: every builder would
# otherwise begin by running wmlib.verify + reading the divergent frames + diffing them. Prints the
# verify summary, per-level accuracy, and the graded cells the model gets wrong at the first divergence.
_VERIFY_STATE_PROBE = r'''
import sys, os, glob
sys.dont_write_bytecode = True
for _pyc in glob.glob("__pycache__/world_model*.pyc") + glob.glob("__pycache__/wmlib*.pyc"):
    try: os.remove(_pyc)
    except OSError: pass
try:
    import wmlib, world_model
except Exception as e:
    print(f"[verify state] no usable world_model.py yet ({type(e).__name__}: {e}) — inspect "
          f"frames/transitions via run_python (frames, transitions, np, wmlib are preloaded) and "
          f"write a first model."); sys.exit(0)
try:
    import numpy as np
    def _cov_status(cov):
        if cov is None:
            return "unobserved"
        if cov <= 0:
            return "vacuous"
        if cov >= 0.999999:
            return "complete"
        return "partial"
    def _cov_desc(cov, status):
        cov_s = "n/a" if cov is None else f"{cov:.2f}"
        if status == "complete":
            return f"{cov_s} complete: render() claimed every cell"
        if status == "partial":
            return f"{cov_s} partial: UNKNOWN=-1 abstained on unclaimed cells"
        if status == "vacuous":
            return f"{cov_s} vacuous: render() claimed no cells; not reliable evidence"
        return f"{cov_s} unobserved"
    def _misfire_desc(kind):
        return {
            "false-level-complete": "decision-state false positive: outcome() says level_complete before the level ended",
            "missed-level-complete": "post-winning-action missed terminal: outcome() did not say level_complete",
            "false-game-over": "decision-state false positive: outcome() says game_over before API GAME_OVER",
            "missed-game-over": "post-fatal-action missed terminal: outcome() did not say game_over",
            "terminal-render": "terminal render mismatch: post-winning state does not render like observed terminal",
        }.get(kind, kind)
    def _level_stats(v, lvl):
        if lvl is None:
            return None
        per = v.get("per_level") or {}
        return per.get(lvl) or per.get(str(lvl))
    v = wmlib.verify(world_model)
    nt = v.get("n_total", 0); nc = v.get("n_changing", 0)
    sim = v.get("simulation_accuracy")
    init_acc = v.get("initial_render_accuracy")
    init_strict = v.get("initial_render_strict_accuracy")
    print(f"[verify state] initial render: accepted_level_start_accuracy="
          f"{'n/a' if init_acc is None else f'{init_acc:.2f}'}; strict_level_start_accuracy="
          f"{'n/a' if init_strict is None else f'{init_strict:.2f}'} over "
          f"{v.get('n_initial_render', 0)} level start(s)")
    _cur_key = wmlib.current_frame_key()
    _cur_level = _cur_key[0] if _cur_key is not None else None
    _cur_init = _level_stats(v, _cur_level)
    if _cur_init and not _cur_init.get("initial_render_ok"):
        print(f"  current L{_cur_level} render(init_state()) mismatch: "
              f"{_cur_init.get('initial_render_first_mismatch')}")
    if not nc:
        # No graded transitions to validate transition() against. Do NOT exit — outcome()
        # may still be falsifiable (a one-action solved level has a recorded winning action + a true
        # terminal), so fall through to the outcome channel below; a wrong outcome() must not hide here.
        print(f"[verify state] no graded transitions yet ({nt} observed, all no-op) — "
              "transition() not yet validatable; checking the outcome channel.")
        if v.get("archived_attempts"):
            print(f"  archived RESET evidence not replay-scored: attempts={v.get('archived_attempts', 0)}, "
                  f"transitions={v.get('archived_transitions', 0)}, "
                  f"observed-changing={v.get('archived_observed_changed', 0)}")
    else:
        sim_s = "n/a" if sim is None else f"{sim:.2f}"
        cell = v.get("cell_accuracy")
        cell_s = "n/a" if cell is None else f"{cell:.3f}"
        print(f"[verify state] simulation_accuracy={sim_s}; dynamics replay "
              f"matched={v.get('n_matched', 0)}/evaluated={nc}/observed={nt} "
              "current-attempt transitions")
        if v.get("n_joint_noop"):
            print(f"  omitted jointly-unchanged no-ops={v.get('n_joint_noop', 0)}; "
                  f"false model changes on observed no-ops={v.get('n_false_change_on_noop', 0)}")
        if v.get("archived_attempts"):
            print(f"  archived RESET evidence not replay-scored: attempts={v.get('archived_attempts', 0)}, "
                  f"transitions={v.get('archived_transitions', 0)}, "
                  f"observed-changing={v.get('archived_observed_changed', 0)}")
        if v.get("variant_used"):
            strict = v.get("strict_simulation_accuracy")
            strict_s = "n/a" if strict is None else f"{strict:.2f}"
            print(f"  strict_simulation_accuracy={strict_s}; observation_variants accepted "
                  f"{v.get('variant_used')} transition(s) (limit {v.get('variant_limit')})")
        if v.get("unknown_used") or (v.get("prediction_coverage") is not None and v.get("prediction_coverage") < 0.999):
            cov = v.get("prediction_coverage")
            known = v.get("known_cell_accuracy")
            cov_status = v.get("prediction_coverage_status") or _cov_status(cov)
            known_s = "n/a" if known is None else f"{known:.2f}"
            print(f"  prediction_coverage={_cov_desc(cov, cov_status)}; known_cell_accuracy={known_s}; "
                  f"UNKNOWN used on {v.get('unknown_used', 0)} transition(s)")
        if cell is not None and (sim is not None and sim < 0.999 or v.get("variant_used") or v.get("unknown_used")):
            print(f"  cell_accuracy={cell_s} strict full-grid diagnostic (UNKNOWN counts as non-matching)")
        pl = {k: lv.get("simulation_accuracy") for k, lv in (v.get("per_level") or {}).items()}
        if pl:
            cells = ", ".join(f"L{k}={'n/a' if x is None else f'{x:.2f}'}" for k, x in sorted(pl.items()))
            print(f"  per-level: {cells}")
        cur_key = wmlib.current_frame_key()
        cur_level = cur_key[0] if cur_key is not None else None
        cur = _level_stats(v, cur_level)
        if cur_level is not None and cur:
            cur_sim = cur.get("simulation_accuracy")
            cur_s = "n/a" if cur_sim is None else f"{cur_sim:.2f}"
            print(f"  current level L{cur_level}: simulation_accuracy={cur_s}; "
                  f"matched={cur.get('n_matched', 0)}/evaluated={cur.get('n_changing', 0)}/"
                  f"observed={cur.get('n_total', 0)}")
        for e in (v.get("variant_errors") or [])[:3]:
            print(f"  observation_variants issue: {e}")
    # OUTCOME channel — printed BEFORE the dynamics-divergence detail (and before the perfect-dynamics
    # early-exit below), so a model with sim_accuracy=1.0 but a wrong outcome() still surfaces here.
    # level_complete false positives steer the planner to phantom wins; false negatives make it miss
    # solved terminals. A perfect transition() with a wrong outcome() scores poorly.
    try:
        g = wmlib.verify_outcome(world_model)
        lc_t, lc_nt = g.get("level_complete_on_terminal"), g.get("level_complete_on_nonterminal")
        go_t, go_nt = g.get("game_over_on_death"), g.get("game_over_on_nonterminal")
        if not g.get("outcome_observable"):
            print("  outcome: no level_complete or GAME_OVER terminal observed yet — "
                  "transition dynamics may verify before the goal/hazard rule is known; "
                  "outcome() is not fully falsifiable yet (keep hypotheses).")
        else:
            lc_t_s = "n/a" if lc_t is None else f"{lc_t:.2f}"
            lc_nt_s = "n/a" if lc_nt is None else f"{lc_nt:.2f}"
            go_t_s = "n/a" if go_t is None else f"{go_t:.2f}"
            go_nt_s = "n/a" if go_nt is None else f"{go_nt:.2f}"
            pred_ok = bool(g.get("outcome_ok", g.get("ok")))
            render_ok = g.get("terminal_render_ok")
            if pred_ok and render_ok is not False:
                verdict = "OK"
            else:
                parts = []
                if (lc_nt or 0) > 0:
                    parts.append("LEVEL_COMPLETE TOO LOOSE (labels a decision state as a win)")
                if lc_t is not None and lc_t < 0.999:
                    parts.append("LEVEL_COMPLETE TOO TIGHT (misses a real level-end)")
                if (go_nt or 0) > 0:
                    parts.append("GAME_OVER TOO LOOSE (labels a decision state as game_over)")
                if go_t is not None and go_t < 0.999:
                    parts.append("GAME_OVER MISSED (misses observed API GAME_OVER)")
                if render_ok is False:
                    parts.append("TERMINAL RENDER MISMATCH (win state pixels do not match the observed terminal)")
                verdict = " + ".join(parts) or "WRONG"
            print(f"  outcome channel: post-winning-action level_complete={lc_t_s} (want 1.00), "
                  f"decision-state false level_complete={lc_nt_s} (want 0.00), "
                  f"post-fatal-action game_over={go_t_s} (want 1.00 when observed), "
                  f"decision-state false game_over={go_nt_s} (want 0.00) — {verdict}")
            # render bridge: modeled terminal must reproduce the OBSERVED winning frame (terminal event).
            tre = g.get("terminal_render_exact")
            if g.get("terminal_render_variant_used") or g.get("terminal_render_unknown_used"):
                strict = g.get("terminal_render_strict_exact")
                strict_s = "n/a" if strict is None else f"{strict:.2f}"
                trc = g.get("terminal_render_cell")
                trc_s = "n/a" if trc is None else f"{trc:.2f}"
                cov = g.get("terminal_render_prediction_coverage")
                cov_status = g.get("terminal_render_coverage_status") or _cov_status(cov)
                bits = []
                if g.get("terminal_render_variant_used"):
                    bits.append(f"observation_variants on {g.get('terminal_render_variant_used')} terminal(s)")
                if g.get("terminal_render_unknown_used"):
                    bits.append(f"UNKNOWN on {g.get('terminal_render_unknown_used')} terminal(s)")
                print(f"  terminal render: accepted with {' and '.join(bits)}; "
                      f"strict_exact={strict_s} coverage={_cov_desc(cov, cov_status)} "
                      f"canonical_cell={trc_s}.")
            if tre is not None and (tre < 0.999 or (g.get("terminal_render_coverage_status") == "vacuous")):
                trc = g.get("terminal_render_cell")
                trc_s = "n/a" if trc is None else f"{trc:.2f}"
                cov = g.get("terminal_render_prediction_coverage")
                cov_status = g.get("terminal_render_coverage_status") or _cov_status(cov)
                print(f"  terminal render: exact={tre:.2f} coverage={_cov_desc(cov, cov_status)} "
                      f"cell={trc_s} — your win state does not reproduce "
                      f"the OBSERVED winning frame; fix transition()/render() at the winning action.")
            for e in (g.get("variant_errors") or [])[:3]:
                print(f"  outcome observation_variants issue: {e}")
            for kind, lvl, tn in (g.get("misfires") or [])[:5]:
                print(f"    {_misfire_desc(kind)}: L{lvl} turn {tn}")
    except Exception as e:
        print(f"  (outcome channel check raised {type(e).__name__}: {e})")
    try:
        deaths = wmlib.death_events()
        if deaths:
            print(f"  GAME_OVER evidence: {len(deaths)} fatal action(s) observed outside normal transitions")
            for ev in deaths[-3:]:
                rc = "" if ev.get("row") is None else f" at ({ev.get('row')},{ev.get('col')})"
                print(f"    L{ev.get('level')} turn {ev.get('turn')} {ev.get('action')}{rc} -> GAME_OVER; "
                      "treat as a negative planning constraint")
    except Exception as e:
        print(f"  (GAME_OVER evidence check raised {type(e).__name__}: {e})")
    try:
        print(wmlib.format_game_over_action_report(wmlib.game_over_action_report(world_model)))
    except Exception as e:
        print(f"MODEL_GAME_OVER_ACTIONS: check skipped ({type(e).__name__}: {e})")
    errs = v.get("errors") or []
    if errs:
        for e in errs[:3]: print(f"  ⚠ model raised: {e}")
        if len(errs) > 3: print(f"    … +{len(errs)-3} more")
    fd = v.get("first_divergence")
    if not fd:
        if sim is not None and sim >= 0.999:
            print("  model reproduces every observed transition. Refine outcome() if open hypotheses, "
                  "else 'done'.")
        sys.exit(0)
    print(f"  first divergence: L{fd['level']} turn {fd['turn']} action {fd['action']} — {fd['diff']}")
    # show the cells your model gets wrong on this transition (every cell is graded — no HUD mask)
    fr = wmlib.frames(); ts = wmlib.transitions()
    tr = next((t for t in ts if t["level"]==fd["level"] and t["turn"]==fd["turn"]), None)
    if tr is None: sys.exit(0)
    prev, nxt = tr["prev"], tr["next"]
    try:
        state = world_model.init_state(fr[(fd["level"], min(t for (l,t) in fr if l==fd["level"]))], fd["level"])
        # thread forward to this transition's pre-state
        for t in [x for x in ts if x["level"]==fd["level"] and x["turn"] <= fd["turn"]-1]:
            state = world_model.transition(state, {"action": t["action"], "row": t["row"], "col": t["col"]})
        state = world_model.transition(state, {"action": tr["action"], "row": tr["row"], "col": tr["col"]})
        pred = np.asarray(world_model.render(state))
    except Exception as e:
        print(f"  (transition/render at this turn raised {type(e).__name__}: {e})")
        sys.exit(0)
    # Two compact, lossless diffs (rectangular run-length encoding via wmlib.diff_text):
    print("  OBSERVED delta (prev -> observed next):")
    for line in wmlib.diff_text(prev, nxt).splitlines():
        print(f"    {line}")
    print("  PREDICTED delta (prev -> your model's next):")
    for line in wmlib.diff_text(prev, pred).splitlines():
        print(f"    {line}")
except Exception as e:
    print(f"[verify state] probe crashed: {type(e).__name__}: {e}")
'''


def _verify_state(executor) -> tuple[str, tuple[int, int] | None]:
    """Run the auto-attached verify-state probe in the workspace.
    Returns (text, (divergence_level, divergence_turn) | None).
    The coords let the caller attach the prev+next PNGs for that transition. None when there's
    no divergence to point at (model accurate, no model yet, no transitions, or probe crashed)."""
    out = executor._run_python(_VERIFY_STATE_PROBE, timeout=30)
    text = (out or "[verify state] probe produced no output").strip()
    # parse the structured "first divergence: L<n> turn <t> action <a>" line out of the text
    # output, so the builder caller can read frame PNGs for that transition without re-running
    # verify (the probe just spent compute on it).
    coords = None
    import re
    m = re.search(r"first divergence: L(\d+) turn (\d+)", text)
    if m:
        coords = (int(m.group(1)), int(m.group(2)))
    return text, coords


class WorldModelBuilder:
    """Runs a bounded tool loop against the SAME workspace as the actor, building world_model.py
    and writing the report. Shares the actor's ToolExecutor/workspace; keeps its OWN short
    conversation (not the actor's history) so its prompt stays lean."""

    def __init__(self, cfg, executor, tools_spec, *, effort, max_tokens, max_steps=None):
        self.cfg = cfg
        self.exec = executor
        self.tools_spec = tools_spec
        self.effort = effort
        self.max_tokens = max_tokens
        # Generous default: the builder spends passes UNDERSTANDING (read frames, diff them,
        # confirm the actor's hypothesis) before it can edit world_model.py, so a tight cap can
        # be exhausted before the model is ever written. Frontier models don't loop forever.
        self.max_steps = max_steps or int(os.environ.get("TYCHO_BUILDER_MAX_STEPS", "40"))
        self.calls = 0
        self.token_stats = {
            "tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cache_write": 0,
            "latency_ms": 0,
        }

    def _note_reply(self, reply: dict) -> None:
        for key, value in reply_usage(reply).items():
            self.token_stats[key] = int(self.token_stats.get(key, 0)) + int(value)

    def take_token_stats(self) -> dict[str, int]:
        """Return and clear usage accumulated since the actor last consumed it."""
        out = dict(self.token_stats)
        for key in self.token_stats:
            self.token_stats[key] = 0
        return out

    def build(self, level: int, hint: str = "", max_calls: int | None = None,
              available_actions: list[str] | None = None) -> tuple[str, list]:
        """Run one build/refine pass. `hint` is the actor's reason for invoking (optional).
        Returns (report_text, trace). The report is also persisted to REPORT_FILE."""
        call_limit = self.max_steps if max_calls is None else max(0, min(self.max_steps, int(max_calls)))
        if call_limit <= 0:
            return ("confidence: none\nmodel: skipped — no LLM-call budget remained for the builder\n"
                    "dynamics: n/a\noutcome: unknown\noutcome_verified: no\n"
                    "plan: start=current; validated=no; length=none; assumptions=unknown\n"
                    "recommended_action: none   subgoal: none\n"
                    "note: continue from the current observations; the builder did not run."), []
        ws = self.exec.ws
        available_actions = list(available_actions or [])
        # TYCHO_BUILDER_STARTER: a first-model recipe while world_model.py is still the seed. In the 2 h
        # Qwen run 21 of 24 orchestrator workspaces still held the seed transition() at the end.
        starter = (os.environ.get("TYCHO_BUILDER_STARTER", "") in ("1", "on", "true", "yes")
                   and _SEED_MARKER in ws.read_file("world_model.py", max_bytes=200000))
        kickoff = render_prompt("builder.user", level=level, hint=hint,
                                available_actions=available_actions, report_file=REPORT_FILE,
                                starter=starter)
        # Auto-attach the current verify state up front so the builder doesn't have to spend its
        # first 2–4 tool calls running verify + diffing frames itself (it always would). When the
        # verify state names a first_divergence, ALSO attach the two PNGs (prev frame + observed
        # next frame) so the builder can SEE the transition that broke the model — easier to
        # spot block recolors, motion, region changes visually than from text diffs alone.
        verify_state, div_coords = _verify_state(self.exec)
        vs_content: list = [{"text": verify_state}]
        if div_coords is not None:
            div_lvl, div_turn = div_coords
            png_prev = ws.current_png(div_lvl, div_turn - 1) if div_turn >= 1 else None
            png_next = ws.current_png(div_lvl, div_turn)
            if png_prev is not None and png_next is not None:
                vs_content.append({"text":
                    f"\n[divergence frames] PNG 1 = prev (L{div_lvl} turn {div_turn-1}), "
                    f"PNG 2 = observed next (L{div_lvl} turn {div_turn}) — the transition the model gets wrong:"})
                vs_content.append({"image_png": png_prev})
                vs_content.append({"image_png": png_next})
        history = [{"role": "user", "content": kickoff},
                   {"role": "user", "content": vs_content}]
        continuity = response_id_continuity_enabled(self.cfg)
        previous_response_id = None
        chain_sent_upto = 0
        continuity_soft_tokens = int(os.environ.get("TYCHO_CONTEXT_EMERGENCY_SOFT_TOKENS", "0") or 0)

        def builder_chat() -> dict:
            """Run one builder call while bounding opaque GPT server-side state.

            Each build invocation owns an independent chain. When the provider reports that the
            retained prompt crossed Tycho's soft limit, the next call re-establishes from the
            visible builder transcript instead of extending an unbounded hidden-reasoning chain.
            """
            nonlocal previous_response_id, chain_sent_upto
            live = continuity and previous_response_id is not None
            kwargs = {
                "system": _BUILDER_SYS,
                "max_tokens": self.max_tokens,
                "effort": self.effort,
                "call_type": "builder",
                "prev_response_id": previous_response_id if live else None,
                "since": chain_sent_upto if live else 0,
            }
            try:
                reply = chat_tools(history, self.tools_spec, self.cfg, **kwargs)
            except Exception as exc:  # noqa: BLE001 - one bounded recovery from opaque server state
                if not live or not is_context_limit_error(exc):
                    raise
                previous_response_id = None
                chain_sent_upto = 0
                kwargs.update(prev_response_id=None, since=0)
                reply = chat_tools(history, self.tools_spec, self.cfg, **kwargs)
            if not continuity:
                return reply
            usage = reply_usage(reply)
            prompt_total = usage["tokens_in"] + usage["cache_read"] + usage["cache_write"]
            rid = reply.get("response_id")
            if rid and not (continuity_soft_tokens and prompt_total >= continuity_soft_tokens):
                previous_response_id = rid
                # The provider stores this response; the caller appends one matching assistant
                # history item immediately after return, so it is not part of the next input delta.
                chain_sent_upto = len(history) + 1
            else:
                previous_response_id = None
                chain_sent_upto = 0
            return reply
        # FRESHNESS: snapshot the report BEFORE this pass. If the builder doesn't rewrite it (crash,
        # step-cap before writing, a pass that only edited world_model.py), the file still holds the
        # PRIOR pass's report — returning that as if fresh injects stale confidence/recommended-action
        # into the actor. We detect "unchanged" below and force a fresh report instead.
        report_before = ws.read_file(REPORT_FILE, max_bytes=4000)
        trace = []
        log_io = os.environ.get("LLM_LOG") == "1"
        if log_io:
            from tycho.serving.llm_client import set_system_prompt
            set_system_prompt(_BUILDER_SYS)  # identify builder calls in the recorded trace
        for _ in range(call_limit):
            reply = builder_chat()
            self.calls += 1
            self._note_reply(reply)
            calls = reply.get("tool_calls") or []
            history.append({"role": "assistant", "content": reply.get("text", ""),
                            "tool_calls": calls or None, "raw_reasoning": reply.get("raw_reasoning"),
                            "reasoning_items": reply.get("reasoning_items")})
            if not calls:
                break  # builder finished (replied 'done' or ran out of things to do)
            results = []
            for c in calls:
                cn = canonical(c["name"])
                if cn == "take_action":
                    # the builder must NOT act — hard block, tell it so
                    out = "(blocked — you are the builder; you do not take game actions)"
                else:
                    out = self.exec.execute(c["name"], c["input"])
                results.append({"id": c["id"], "name": c["name"], "output": out})
                trace.append({"tool": cn, "args": c.get("input", {}), "result": str(out)[:200]})
            history.append({"role": "tool", "results": results})

        # Always return a signal to the actor. If no report exists after the bounded tool loop,
        # spend one final, tool-free pass requesting the compact report.
        report = ws.read_file(REPORT_FILE, max_bytes=4000)
        # STALE GUARD: a report that is missing OR byte-identical to the pre-pass snapshot means THIS
        # pass produced no fresh report — don't hand the actor last pass's text as current. Force one.
        can_spend_report_call = max_calls is None or self.calls < max_calls
        if (report.startswith("(no such file") or report == report_before) and can_spend_report_call:
            history.append({"role": "user", "content":
                "Step budget reached. Do NOT call any more tools. Right now, in your reply, give "
                "the compact report in the required format (confidence/model/dynamics/outcome/"
                "outcome_verified/plan/recommended_action+subgoal/note) summarizing what you learned and the "
                "state of world_model.py — even if it's 'still exploring, model not yet coded'. Be "
                "concrete about what you DID figure out — especially any open outcome hypotheses and the "
                "action that would disambiguate them — so the actor can use it."})
            reply = builder_chat()
            self.calls += 1
            self._note_reply(reply)
            txt = (reply.get("text") or "").strip()
            # Persist it so the actor and trace analysis see the same report.
            report = ws.read_file(REPORT_FILE, max_bytes=4000)
            if report.startswith("(no such file") or report == report_before:
                report = txt or ("confidence: none\nmodel: none — builder did not finish a model "
                                 "in its step budget\ndynamics: n/a\noutcome: unknown\n"
                                 "outcome_verified: no-terminal-yet\n"
                                 "plan: start=current; validated=no; length=none; assumptions=unknown\n"
                                 "recommended_action: none   "
                                 "subgoal: none\nnote: keep exploring; re-invoke with more specific "
                                 "mechanics once you have them")
                # PERSIST the fallback to REPORT_FILE so a later file read / history reset doesn't
                # resurface the STALE prior-pass report — the returned text and the on-disk file agree.
                try:
                    ws.write_file(REPORT_FILE, report)
                except Exception:  # noqa: BLE001 — never fail a build over a report write
                    pass
        elif report.startswith("(no such file") or report == report_before:
            report = ("confidence: none\nmodel: none — builder exhausted its reserved call budget before "
                      "writing a fresh report\ndynamics: n/a\noutcome: unknown\noutcome_verified: no\n"
                      "plan: start=current; validated=no; length=none; assumptions=unknown\n"
                      "recommended_action: none   subgoal: none\n"
                      "note: keep acting from the current evidence; re-invoke only if more budget remains.")
            try:
                ws.write_file(REPORT_FILE, report)
            except Exception:  # noqa: BLE001
                pass
        return report, trace
