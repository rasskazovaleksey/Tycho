"""Cheap world-model-divergence signal for the `trigger` mode builder condition.

The `trigger` mode fires the builder when the current world model is detectably wrong on the observed
trace. "Wrong" has TWO independent channels in the Moore machine M=(St,s0,Acts,δ,Ω,ρ,T):

  1. DYNAMICS — δ∘ρ is not accepted for an observed grid (simulation_accuracy < 1.0). A wrong
     transition or render. (s0/hidden-state errors surface here too, as a downstream render mismatch.)
     simulation_accuracy uses bounded observation_variants() when the model supplies them; strict
     canonical render accuracy remains separately reported as strict_simulation_accuracy. render()
     may use UNKNOWN=-1 for genuinely unclaimed cells, but a very low prediction_coverage is treated
     as inaccurate so trigger mode cannot go quiet on an all-unknown model.
  2. OUTCOME — outcome() disagrees with observed terminal evidence:
       level_complete too_loose : outcome() returns "level_complete" on a state that was NOT a
                   level end. This is PLANNING-CRITICAL: the planner steers toward a phantom terminal.
       level_complete too_tight : outcome() misses a solved level's terminal. Only knowable once a
                   real terminal is observed, i.e. at a level boundary.
       game_over missed/too_loose: outcome() misses an observed API GAME_OVER, or returns "game_over"
                   on an ordinary decision frame.

Historically this probe checked only channel 1, so a model with perfect dynamics but a wrong terminal
predicate could go quiet. Both channels are checked now, in ONE subprocess (verify + verify_outcome
share the model import).

Computing this must be CHEAP — it runs on every committed turn. It is one sandboxed subprocess: the
AGENT-AUTHORED world_model.py must execute under the SAME sandbox as run_python (it can import the
engine otherwise), so we evaluate it via ToolExecutor._run_python (Baseline B sandbox, stale-.pyc
handling). Both verify and verify_outcome are CPU-only numpy over the observed trace — milliseconds; the
subprocess startup dominates, and combining the two channels into one script pays that once.

Conservative default: if world_model.py is absent/unloadable/crashing, dynamics is reported wrong
(there is no model → the builder SHOULD run to bootstrap one); outcome is reported clean (a model that
can't load can't have a committed-but-wrong outcome — let the dynamics arm drive the first build).
"""

from __future__ import annotations


# A self-contained probe script (text). Run in the workspace via the sandboxed _run_python. Prints
# exactly two lines: "DYN:ACCURATE|INACCURATE|NOMODEL" and
# "OUTCOME:ok|level_complete_too_loose|level_complete_too_tight|game_over_missed|
#          game_over_too_loose|terminal_render|invalid|nomodel".
_WM_SIGNAL_PROBE = r'''
import sys, os, glob
sys.dont_write_bytecode = True
for _pyc in glob.glob("__pycache__/world_model*.pyc") + glob.glob("__pycache__/wmlib*.pyc"):
    try: os.remove(_pyc)
    except OSError: pass
try:
    import wmlib, world_model
except Exception:
    print("DYN:NOMODEL"); print("OUTCOME:nomodel"); sys.exit(0)
MIN_COVERAGE = float(os.environ.get("TYCHO_WM_MIN_COVERAGE", "0.75"))
# DYNAMICS — accuracy over ALL observed levels, not just the newest. A model can break a PRIOR
# level (e.g. a transition refactor for the current level regressed an earlier one) while the newest
# level has no graded transitions yet — checking only per[max(per)] would miss that and let the
# trigger go quiet while global verify() is wrong. Use the all-levels simulation_accuracy
# (denominator = graded transitions across all levels); INACCURATE if any level diverges.
dyn_ok = False
try:
    v = wmlib.verify(world_model)
    if v.get("errors"):
        print("DYN:NOMODEL")                          # a crashing model can't predict → needs building
    else:
        _fd = v.get("first_divergence")
        print("STATS:%d/%d|%s" % (int(v.get("n_matched") or 0), int(v.get("n_changing") or 0),
                                  str(_fd)[:200].replace("\n", " ") if _fd else ""))
        if not v.get("n_changing"):
            print("DYN:ACCURATE"); dyn_ok = True       # nothing graded observed anywhere yet
        else:
            acc = v.get("simulation_accuracy")         # over ALL levels' graded transitions
            cov = v.get("prediction_coverage")
            cov_status = v.get("prediction_coverage_status")
            cov_ok = cov is None or (cov_status != "vacuous" and cov >= MIN_COVERAGE)
            dyn_ok = acc is not None and acc >= 0.999 and cov_ok
            print("DYN:ACCURATE" if dyn_ok else "DYN:INACCURATE")
except Exception:
    print("DYN:NOMODEL")
# OUTCOME — only checked when dynamics are already clean: if dynamics mispredict we fire on that
# anyway, so the second forward-simulation (verify_outcome ≈ doubles probe cost) is pure waste.
if not dyn_ok:
    print("OUTCOME:skip")                             # dynamics already firing; outcome pass skipped
else:
    try:
        g = wmlib.verify_outcome(world_model)
        lc_false = g.get("level_complete_on_nonterminal") or 0
        go_false = g.get("game_over_on_nonterminal") or 0
        lc_hit = g.get("level_complete_on_terminal")
        go_hit = g.get("game_over_on_death")
        if g.get("outcome_errors"):
            print("OUTCOME:invalid")
        elif lc_false > 0:
            print("OUTCOME:level_complete_too_loose")
        elif go_false > 0:
            print("OUTCOME:game_over_too_loose")
        elif g.get("level_complete_observable") and lc_hit is not None and lc_hit < 0.999:
            print("OUTCOME:level_complete_too_tight")
        elif g.get("game_over_observable") and go_hit is not None and go_hit < 0.999:
            print("OUTCOME:game_over_missed")
        elif (g.get("terminal_render_exact") is not None
              and (g.get("terminal_render_exact") < 0.999
                   or g.get("terminal_render_coverage_status") == "vacuous"
                   or (g.get("terminal_render_prediction_coverage") is not None
                       and g.get("terminal_render_prediction_coverage") < MIN_COVERAGE))):
            print("OUTCOME:terminal_render")
        else:
            print("OUTCOME:ok")
    except Exception:
        print("OUTCOME:nomodel")
'''


def wm_signal(executor) -> dict:
    """Run the cheap combined divergence probe in the (sandboxed) workspace. Returns
    {"dynamics_inaccurate": bool, "outcome_divergence": <reason>|None}.
    `executor` is the ToolExecutor (its _run_python is sandboxed + workspace-scoped)."""
    out = executor._run_python(_WM_SIGNAL_PROBE, timeout=20)
    dyn = "NOMODEL"
    outcome = None
    stats = None
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("STATS:"):
            counts, _, first = line[6:].partition("|")
            try:
                matched, evaluated = (int(x) for x in counts.split("/"))
                stats = (matched, evaluated, first)
            except ValueError:
                stats = None
        elif line.startswith("DYN:"):
            dyn = line[4:]
        elif line.startswith("OUTCOME:"):
            outcome = line[8:]
    valid_outcome_divergences = {
        "level_complete_too_loose",
        "level_complete_too_tight",
        "game_over_missed",
        "game_over_too_loose",
        "terminal_render",
        "invalid",
    }
    return {
        # ACCURATE only when the probe explicitly says so; anything else (INACCURATE/NOMODEL/garbage
        # / no output) → run, to bootstrap or fix a model.
        "dynamics_inaccurate": dyn != "ACCURATE",
        "outcome_divergence": (outcome if outcome in valid_outcome_divergences else None),
        "stats": stats,  # (matched, evaluated, first divergence) when a model imported and verified
    }


def wm_has_inaccuracy(executor) -> bool:
    """Back-compat shim: True iff the DYNAMICS channel is wrong (or no usable model). Outcome divergence
    is consulted separately via wm_signal()['outcome_divergence']."""
    return wm_signal(executor)["dynamics_inaccurate"]
