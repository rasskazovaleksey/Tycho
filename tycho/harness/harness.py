"""Evaluation harness: drives the local arc-agi engine, enforces the
RHAE action budget, computes scores, and writes a structured result record.

Design notes
- Bulk evaluation runs offline against the local `arc-agi` engine. One online call downloads and caches each game
  (incl. per-level human baselines) to `environment_files/`; afterwards no
  network is needed. This is what makes fully-offline runs viable.
- `available_actions` from a frame are game-action ints; RESET is a separate API control.
- Level transitions are implicit: `levels_completed` increments and the engine
  advances; `state == WIN` means all levels done. We detect a level completion
  by a rise in `levels_completed` and attribute the actions since the last
  completion to that level.
- Budget: per the report, official eval terminates a level after `5 * h`
  actions. We enforce the same so local scores are not optimistically biased.
"""

from __future__ import annotations

import inspect
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from arc_agi import Arcade, OperationMode
from arcengine import FrameData, GameAction, GameState

from .agent import ActionChoice, Agent
from .actions import actor_actions, normalize_game_actions
from .scoring import EnvResult, LevelResult

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # harness/ -> tycho/ -> repo root
ENV_DIR = REPO_ROOT / "environment_files"
RESULTS_DIR = REPO_ROOT / "results"
ACTION_BUDGET_MULTIPLIER = 5  # official eval cutoff: 5 * human_baseline per level

log = logging.getLogger("tycho.harness")


def _frame_key(frame) -> int:
    """Hash the innermost grid of a frame (1xHxW). Used for no-op/loop detection."""
    grid = frame[-1] if frame is not None and len(frame) else []
    return hash(tuple(tuple(row) for row in grid))


# Keys that are large or analysis-only: retained in the trace but never sent to the engine
# (arcengine caps ActionInput.reasoning at 16384 bytes).
_VIZ_ONLY_KEYS = {"llm_calls", "world_model_code", "objects", "plan", "mechanics", "prompt"}


def _engine_reasoning(reasoning: Optional[dict]) -> Optional[dict]:
    """A small, engine-safe subset of the agent's reasoning (drops viz-only/large
    keys; truncates as a backstop) so we never exceed the engine's 16KB cap."""
    if not reasoning:
        return reasoning
    small = {k: v for k, v in reasoning.items() if k not in _VIZ_ONLY_KEYS}
    s = str(small)
    if len(s) > 8000:  # backstop: keep only short scalar fields
        small = {k: v for k, v in small.items() if isinstance(v, (int, float, bool, str)) and len(str(v)) < 300}
    return small


def _clamped_click_coord(value, default: int = 32) -> int:
    v = int(value) if isinstance(value, (int, float)) else default
    return max(0, min(63, v))


def _committed_action_from_reasoning(reasoning: Optional[dict]) -> Optional[dict]:
    if not isinstance(reasoning, dict):
        return None
    committed = None
    for item in reasoning.get("tool_trace") or []:
        if not isinstance(item, dict) or item.get("tool") != "take_action" or not item.get("committed"):
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        action = str(args.get("action", "")).upper()
        if not action:
            continue
        info = {"action": action}
        if action == "ACTION6":
            info["x"] = _clamped_click_coord(args.get("col"))
            info["y"] = _clamped_click_coord(args.get("row"))
        committed = info
    return committed


def _choice_action_info(choice: ActionChoice) -> dict:
    info = {"action": choice.action.name}
    if choice.action == GameAction.ACTION6:
        info["x"] = getattr(choice, "x", None)
        info["y"] = getattr(choice, "y", None)
    return info


def _reasoning_matches_replay_choice(reasoning: Optional[dict], choice: ActionChoice) -> bool:
    committed = _committed_action_from_reasoning(reasoning)
    if committed is None:
        return True
    return committed == _choice_action_info(choice)


def _reasoning_for_replayed_choice(
    choice: ActionChoice,
    prior_reasoning: Optional[dict],
    turn_index: int,
) -> Optional[dict]:
    """Return prior reasoning only if it describes the same action the journal is replaying.

    Resume replay rebuilds frames from the append-only action journal. If a stale or corrupt partial
    record is supplied, its transcript must not be attached to a different action.
    """
    if not prior_reasoning:
        return None
    if _reasoning_matches_replay_choice(prior_reasoning, choice):
        return prior_reasoning
    return {
        "src": "resume_prior_reasoning_omitted",
        "resume_reasoning_omitted": True,
        "reason": "prior reasoning committed a different action than the replay journal",
        "t": turn_index,
        "prior_t": prior_reasoning.get("t") if isinstance(prior_reasoning, dict) else None,
        "prior_committed_action": _committed_action_from_reasoning(prior_reasoning),
        "replay_action": _choice_action_info(choice),
    }


def _grid_to_list(frame) -> Optional[list]:
    """Innermost grid as plain nested lists (JSON-safe; frame rows may be ndarrays). Uses frame[-1] —
    the resulting PLAYABLE state after the action (intermediate animation frames are decorative)."""
    if frame is None or not len(frame):
        return None
    grid = frame[-1]
    return [list(map(int, row)) for row in grid]


def _grid_to_first(frame) -> Optional[list]:
    """frame[0] as plain nested lists. On a level-completing action the engine returns
    [goal_reached_terminal, ...animation..., new_level_init]; frame[0] is the terminal BEFORE the
    advance — the goal-reached configuration. (frame[-1] is the next level, which _grid_to_list returns.)"""
    if frame is None or not len(frame):
        return None
    grid = frame[0]
    return [list(map(int, row)) for row in grid]


class StopRun(Exception):
    """Raised by an agent (or a hook an embedding runner installed on it) to end the run cleanly
    between actions: the stop reason is the agent's ``done_reason``, and no error is recorded."""


def _frame_sequence_to_list(frame) -> list[list[list[int]]]:
    if frame is None:
        return []
    return [[list(map(int, row)) for row in grid] for grid in frame]


@dataclass
class TraceStep:
    turn: int
    action: str
    x: Optional[int]
    y: Optional[int]
    state: str
    levels_completed: int
    frame_changed: bool  # did this action alter the grid? (no-op detection)
    frame_key: int  # hash of the resulting grid (loop / revisit detection)
    frame: Optional[list] = None  # resulting HxW grid — only when keep_frames (viz)
    reasoning: Optional[dict] = None  # agent's per-step internal state, if it attached any


@dataclass
class EnvRunRecord:
    game_id: str
    n_levels: int
    baselines: list[int]
    levels_completed: int
    total_actions: int
    resets: int
    final_state: str
    env_score: float
    level_results: list[dict]
    wall_clock_s: float
    truncated_levels: list[int]  # levels cut by the action budget
    completed_level_actions: int = 0
    unfinished_level_index: Optional[int] = None
    unfinished_level_actions: int = 0
    total_actions_including_unfinished: int = 0
    # True for records produced after RESET accounting was aligned with the official scorecard:
    # the initial env.reset() starts the play for free, while every in-play RESET costs one action.
    reset_actions_counted: bool = True
    # Goal-agnostic behavioural diagnostics.
    noop_actions: int = 0  # scored actions that didn't change the grid
    distinct_frames: int = 0  # unique grid states seen
    revisits: int = 0  # transitions back into an already-seen frame
    actions_before_first_level: Optional[int] = None  # exploration cost proxy
    error: Optional[str] = None  # set if the game ended on an exception (partial record)
    stop_reason: Optional[str] = None
    inference_budget: Optional[dict] = None
    # how many times the world-model builder was invoked this game (actor-pull in orchestrator mode
    # OR harness-fired in trigger mode). None in single mode / when the agent has no builder — so
    # single-mode records are unchanged. Surfaced for orchestration analysis.
    builder_invocations: Optional[int] = None
    trace: list[dict] = field(default_factory=list)
    partial: bool = False  # True for a mid-game level-complete snapshot (live viewing) — NOT a
                           # final result; analysis/manifest aggregation must ignore partial records.


def _accepts(fn, param: str) -> bool:
    """True iff callable fn declares a parameter named `param` (so base Agents without the resume
    kwargs still work)."""
    try:
        return param in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False


def _finish_replay(agent: Agent, snapshot, game_id: str) -> None:
    """Restore the exact actor state after deterministic engine replay."""
    from tycho.harness.resume import ResumeCheckpointError

    if snapshot is None or not hasattr(agent, "restore_state"):
        raise ResumeCheckpointError(
            f"{game_id}: exact actor checkpoint unavailable; warm resume is disabled"
        )
    agent.restore_state(snapshot)
    log.info("%s: resumed — engine replayed and exact actor/workspace checkpoint restored", game_id)


def _load_arcade(api_key: str = "") -> Arcade:
    return Arcade(
        arc_api_key=api_key,
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(ENV_DIR),
    )


def run_env(
    agent: Agent,
    arc: Arcade,
    game_id: str,
    baselines: list[int],
    seed: int = 0,
    scorecard_id: Optional[str] = None,
    keep_trace: bool = True,
    keep_frames: bool = False,
    journal=None,            # tycho.harness.resume.GameJournal — within-game resume (optional)
    ws_root: Optional[str] = None,  # durable workspace base (so files survive a kill)
    on_level_complete=None,  # callback(partial_EnvRunRecord) for durable level-granular progress
    reasoning_source=None,   # prior partial record used to restore reasoning for replayed turns
    status_store=None,       # tycho.harness.run_status.GameStatusStore (optional live telemetry)
) -> EnvRunRecord:
    """Play one environment end-to-end under the action budget; return a record.

    keep_frames: also store the raw grid and agent reasoning per step. Implies keep_trace.

    journal: if given, every committed action is logged AND, when the journal already holds a
    prefix (a prior interrupted run), that prefix is REPLAYED through the deterministic engine to
    fast-forward to where we left off before resuming live play (see tycho/harness/resume.py). The
    replay reuses THIS loop with the action source switched (log vs. agent), so all scoring
    bookkeeping is reproduced by identical code; a per-step frame-hash guard aborts on any engine
    divergence. With journal=None the live path is byte-identical to before.
    """
    keep_trace = keep_trace or keep_frames
    n_levels = len(baselines)
    env = arc.make(game_id, seed=seed, scorecard_id=scorecard_id)
    t0 = time.time()

    # Resume bookkeeping: the contiguous committed-action prefix to replay (empty on a fresh run).
    replay = journal.steps() if (journal is not None and journal.exists()) else []
    resuming = len(replay) > 0
    # Replayed turns rebuild engine state and frames, but the journal stores actions rather than
    # per-turn call traces. Recover available reasoning from the prior partial record.
    prior_reasoning = {}
    if resuming and reasoning_source:
        try:
            import json as _json
            _pd = _json.loads(Path(reasoning_source).read_text())
            for _t in _pd.get("trace", []):
                _r = _t.get("reasoning")
                if isinstance(_r, dict) and _r.get("t") is not None:
                    prior_reasoning[_r["t"]] = _r   # key on the stable global turn index
        except Exception:  # noqa: BLE001 — best-effort; a missing/corrupt partial just leaves it hollow
            prior_reasoning = {}
    # Snapshot decoding has its own narrow compatibility gate. The run supervisor separately checks
    # the immutable score-affecting execution policy before launching this worker.
    from tycho.harness.resume import resume_fingerprint
    fingerprint = resume_fingerprint(agent)
    # History STRUCTURE drives snapshot/restore semantics, so the journal's compat gate must key on
    # the RESOLVED history mode (TYCHO_HISTORY: tail_evict|truncate|collapse), not the raw legacy
    # TYCHO_CONTEXT_MODE env (which is unset on current runs → always "legacy" → all history modes
    # looked identical to the gate, so a resume could restore a snapshot taken under a DIFFERENT
    # history behavior). resolve_context_config() handles both the new and old env names.
    from tycho.agent.context_config import resolve_context_config
    ctx_mode = resolve_context_config().history
    snapshot = None
    checkpoint = None
    if resuming:
        ok, why = journal.compatible(fingerprint=fingerprint, context_mode=ctx_mode)
        if not ok:
            from tycho.harness.resume import ResumeCompatibilityError
            raise ResumeCompatibilityError(f"{game_id}: {why}")
        checkpoint = journal.load_checkpoint()
        snapshot = journal.load_actor_state(checkpoint)
        if ws_root is None:
            from tycho.harness.resume import ResumeCheckpointError
            raise ResumeCheckpointError(f"{game_id}: exact resume requires a durable workspace root")
        journal.restore_workspace(Path(ws_root) / game_id.split("-")[0], checkpoint)

    frame = env.reset()
    frames: list[FrameData] = [frame]
    game_avail = normalize_game_actions(frame.available_actions)
    # Reset the agent against the DURABLE workspace; resume=True preserves the agent-authored
    # world_model.py / notes already on disk. (Base Agent.reset has no ws_root/resume kwargs;
    # only resume-capable agents — TychoAgent — accept them.)
    if _accepts(agent.reset, "ws_root"):
        agent.reset(game_id, game_avail, ws_root=ws_root, resume=resuming)
    else:
        agent.reset(game_id, game_avail)
    if journal is not None:
        journal.init_meta(game_id=game_id, fingerprint=fingerprint, context_mode=ctx_mode)

    level_results: list[LevelResult] = []
    truncated: list[int] = []
    trace: list[TraceStep] = []

    cur_level = 0  # 0-indexed level currently being played
    actions_this_level = 0
    resets = 0
    turn = 0

    # Diagnostic counters (computed even without a full trace).
    noop_actions = 0  # scored actions that left the grid unchanged
    seen_keys: set[int] = set()
    revisits = 0  # transitions into an already-seen frame
    actions_before_first_level: Optional[int] = None
    prev_key = _frame_key(frame.frame)
    seen_keys.add(prev_key)
    run_error: Optional[str] = None  # set if a mid-game exception ends the run early
    stop_reason: Optional[str] = None
    pending_game_over_event: Optional[dict] = None

    while True:
        state = frame.state
        completed_so_far = frame.levels_completed

        if state == GameState.WIN:
            stop_reason = "win"
            break
        if cur_level >= n_levels:
            stop_reason = "all_levels_accounted"
            break
        if agent.is_done(frames, frame):
            stop_reason = getattr(agent, "done_reason", None) or "agent_done"
            break

        # Budget for the current level.
        budget = ACTION_BUDGET_MULTIPLIER * baselines[cur_level]

        # During REPLAY (turn < len(replay)), the action comes from the journal, NOT the agent —
        # no LLM call, deterministic. When the prefix is exhausted, restore the agent snapshot once
        # and hand back to live play.
        in_replay = turn < len(replay)
        if resuming and not in_replay and turn == len(replay):
            _finish_replay(agent, snapshot, game_id)
            resuming = False  # only once

        # A mid-game exception (for example a provider outage or an agent bug)
        # must NOT discard the trace collected so far — break with a partial record
        # instead of propagating. The caller persists what we have (overnight runs
        # otherwise lose hours of progress to a single late failure).
        try:
            if in_replay:
                rec = replay[turn]
                choice = ActionChoice(action=GameAction[rec["action"]],
                                      x=rec.get("x"), y=rec.get("y"))
                if rec["action"] == "RESET":
                    resets += 1
            elif state == GameState.GAME_OVER:
                # Only RESET is valid. The initial env.reset() above starts the play for free, but
                # this in-play recovery RESET is one scored action in the official scorecard.
                if hasattr(agent, "note_external_reset"):
                    try:
                        agent.note_external_reset(pending_game_over_event)
                    except TypeError:
                        agent.note_external_reset()
                pending_game_over_event = None
                choice = ActionChoice(action=GameAction.RESET)
                resets += 1
            else:
                avail = actor_actions(frame.available_actions)
                choice = agent.choose_action(frames, frame, avail)

            next_frame = env.step(
                choice.action,
                data=getattr(choice, "data", None),
                reasoning=_engine_reasoning(choice.reasoning),  # engine-safe subset (16KB cap)
            )
        except StopRun:
            stop_reason = getattr(agent, "done_reason", None) or "agent_done"
            break
        except Exception as e:  # noqa: BLE001 — preserve partial progress on any failure
            run_error = f"{type(e).__name__}: {e}"
            stop_reason = "exception"
            log.warning("%s: run aborted at turn %d: %s", game_id, turn, run_error)
            break

        # Engine-determinism GUARD: a replayed action must reproduce the recorded frame exactly.
        # A mismatch means the engine is non-deterministic across this run (version drift, a seed
        # we missed) — abort loudly rather than resume from a silently-wrong state.
        if in_replay and next_frame is not None:
            got = _frame_key(next_frame.frame)
            want = replay[turn].get("frame_hash")
            if want is not None and got != want:
                raise RuntimeError(
                    f"{game_id}: resume replay diverged at step {turn} "
                    f"(action {replay[turn]['action']}): engine frame hash {got} != logged {want}. "
                    f"Engine is not reproducing the recorded trajectory; refusing to resume.")
        turn += 1
        # Official scorecard accounting charges every in-play environment control, including RESET.
        # The initialization reset is outside this loop and therefore remains uncharged.
        actions_this_level += 1

        if next_frame is None:
            log.warning("%s: engine returned None at turn %d", game_id, turn)
            stop_reason = "engine_returned_none"
            break
        if int(next_frame.levels_completed) < int(completed_so_far):
            run_error = (
                "LevelRegressionError: ARC engine reported levels_completed decrease "
                f"{completed_so_far}->{next_frame.levels_completed} after {choice.action.name}; "
                "refusing to continue from a full-game reset state"
            )
            stop_reason = "level_regression"
            log.warning("%s: %s", game_id, run_error)
            break

        if choice.action == GameAction.RESET and state != GameState.GAME_OVER and not in_replay:
            resets += 1
            if hasattr(agent, "note_actor_reset"):
                try:
                    agent.note_actor_reset()
                except Exception:  # noqa: BLE001 -- reset bookkeeping should not abort a run
                    pass

        if choice.action != GameAction.RESET and next_frame.state == GameState.GAME_OVER:
            ws = getattr(agent, "ws", None)
            game_over_grid = _grid_to_first(next_frame.frame) or _grid_to_list(next_frame.frame)
            prev_grid = _grid_to_list(frame.frame)
            pending_game_over_event = {
                "level": cur_level,
                "turn": actions_this_level,
                "action": choice.action.name,
                "row": getattr(choice, "y", None),
                "col": getattr(choice, "x", None),
                "state": "GAME_OVER",
                "prev": prev_grid,
                "next": game_over_grid,
                "frames": _frame_sequence_to_list(next_frame.frame),
            }
            if not in_replay and ws is not None and hasattr(ws, "record_game_over"):
                try:
                    recorded_event = ws.record_game_over(
                        level=cur_level,
                        turn_in_level=actions_this_level,
                        action=choice.action.name,
                        row=getattr(choice, "y", None),
                        col=getattr(choice, "x", None),
                        prev_grid=prev_grid,
                        game_over_grid=game_over_grid,
                    )
                    if isinstance(recorded_event, dict):
                        pending_game_over_event["stem"] = recorded_event.get("stem")
                except Exception:  # noqa: BLE001 -- bookkeeping must not abort a run
                    pass

        # Most agents observe an action's result on the next choose_action() call. Terminal states do
        # not necessarily have a next decision, so give opt-in agents one final evidence callback.
        # This keeps the base Agent contract unchanged while preventing the winning/fatal action from
        # disappearing from an online learner's exact memory.
        if next_frame.state in (GameState.WIN, GameState.GAME_OVER) and hasattr(agent, "note_final_observation"):
            try:
                agent.note_final_observation(
                    next_frame,
                    normalize_game_actions(next_frame.available_actions),
                )
            except Exception:  # noqa: BLE001 -- evidence bookkeeping must not abort a scored run
                log.exception("%s: terminal observation callback failed at turn %d", game_id, turn)

        new_key = _frame_key(next_frame.frame)
        changed = new_key != prev_key
        if choice.action != GameAction.RESET and not changed:
            noop_actions += 1
        if new_key in seen_keys and changed:
            revisits += 1
        seen_keys.add(new_key)

        if keep_trace:
            # On a REPLAYED turn choice.reasoning is empty (the journal stored only the action). Re-attach
            # available prior reasoning, keyed by global turn index (turn-1 after increment).
            step_reasoning = choice.reasoning
            if in_replay and not step_reasoning and prior_reasoning:
                step_reasoning = _reasoning_for_replayed_choice(
                    choice, prior_reasoning.get(turn - 1), turn - 1
                )
            trace_grid = None
            if keep_frames:
                trace_grid = (
                    (_grid_to_first(next_frame.frame) or _grid_to_list(next_frame.frame))
                    if next_frame.state == GameState.GAME_OVER
                    else _grid_to_list(next_frame.frame)
                )
            trace.append(
                TraceStep(
                    turn=turn,
                    action=choice.action.name,
                    x=getattr(choice, "x", None),
                    y=getattr(choice, "y", None),
                    state=str(next_frame.state),
                    levels_completed=next_frame.levels_completed,
                    frame_changed=changed,
                    frame_key=new_key,
                    frame=trace_grid,
                    reasoning=step_reasoning,
                )
            )
        prev_key = new_key

        # Did a level just complete? (levels_completed rose)
        level_completed = next_frame.levels_completed > completed_so_far
        budget_exhausted = False
        if level_completed:
            # Record the ACTION that completed THIS level, into the just-finished level's workspace
            # dir. The harness is the one place that sees every completion uniformly with `choice` in
            # scope, including the final level where the agent never re-enters choose_action. The
            # terminal frame itself is recorded separately below.
            _ws = getattr(agent, "ws", None)
            if not in_replay and _ws is not None and hasattr(_ws, "record_solved"):
                try:
                    _ws.record_solved(cur_level, choice.action.name,
                                      row=getattr(choice, "y", None), col=getattr(choice, "x", None))
                except Exception:  # noqa: BLE001 — never break the run for a bookkeeping write
                    pass
            # WINNING-TERMINAL EVENT (separate channel, like solved/death evidence). On a
            # level-completing action the engine returns a frame LIST whose FIRST element is the
            # goal-reached terminal (the modeled post-action state of the OLD level) and whose LAST is
            # the next level's init. The terminal is NOT a decision frame (the agent never acts from
            # it), so it is NOT a turn_*.txt; it is the render ground-truth that verify_outcome bridges to:
            # replay the level to the pre-win state, apply the winning action, and require
            # outcome(s_terminal)=="level_complete" AND render(s_terminal)==this observed terminal grid. Recorded
            # uniformly here for EVERY completion (incl. the final level, where the agent never
            # re-enters choose_action).
            if not in_replay and _ws is not None and hasattr(_ws, "record_terminal"):
                fl = next_frame.frame
                if isinstance(fl, list) and len(fl) >= 1:
                    try:
                        _ws.record_terminal(cur_level, _grid_to_first(fl), action=choice.action.name,
                                            row=getattr(choice, "y", None), col=getattr(choice, "x", None))
                    except Exception:  # noqa: BLE001
                        pass
            if not level_results:  # first completion: exploration-cost proxy
                actions_before_first_level = actions_this_level
            level_results.append(
                LevelResult(
                    level_index=cur_level + 1,
                    completed=True,
                    actions_taken=actions_this_level,
                    baseline_actions=baselines[cur_level],
                )
            )
            cur_level += 1
            actions_this_level = 0
            # Hand the caller a level-granular partial record. A callback failure must not break play.
        elif actions_this_level >= budget:
            # Budget exhausted on this level without completing it.
            level_results.append(
                LevelResult(
                    level_index=cur_level + 1,
                    completed=False,
                    actions_taken=actions_this_level,
                    baseline_actions=baselines[cur_level],
                )
            )
            truncated.append(cur_level + 1)
            cur_level += 1
            actions_this_level = 0
            stop_reason = "level_action_budget"
            budget_exhausted = True
            if hasattr(agent, "note_final_observation"):
                try:
                    agent.note_final_observation(
                        next_frame,
                        normalize_game_actions(next_frame.available_actions),
                    )
                except Exception:  # noqa: BLE001 -- evidence bookkeeping must not abort a scored run
                    log.exception("%s: final observation callback failed at action budget", game_id)

        # Commit only after all workspace/scoring effects of the action are durable.  A killed
        # in-flight next turn may have edited files; resume replaces them with this exact manifest.
        if journal is not None and not in_replay:
            snap = agent.snapshot_state() if hasattr(agent, "snapshot_state") else None
            workspace_dir = getattr(getattr(agent, "ws", None), "dir", None)
            if workspace_dir is None:
                from tycho.harness.resume import ResumeCheckpointError
                raise ResumeCheckpointError(f"{game_id}: agent has no checkpointable workspace")
            journal.record_step(
                turn - 1, action=choice.action.name,
                x=getattr(choice, "x", None), y=getattr(choice, "y", None),
                frame_hash=new_key, level=next_frame.levels_completed,
                state=str(next_frame.state), agent_snapshot=snap,
                workspace_dir=workspace_dir)
            if status_store is not None:
                try:
                    current_score = EnvResult(
                        game_id=game_id,
                        n_levels=n_levels,
                        levels=list(level_results),
                        state=str(next_frame.state),
                        resets=resets,
                    ).env_score()
                    status_store.note_action(
                        action_count=turn,
                        action=choice.action.name,
                        level=cur_level,
                        level_action_count=actions_this_level,
                        levels_completed=int(next_frame.levels_completed),
                        rhae=current_score,
                        inference_budget=(
                            agent.inference_budget_state()
                            if hasattr(agent, "inference_budget_state") else None
                        ),
                    )
                except Exception:  # noqa: BLE001 - monitoring must never change game behavior
                    pass

        if level_completed and on_level_complete is not None:
            try:
                on_level_complete(_build_record(
                    game_id, n_levels, baselines, list(level_results), next_frame,
                    resets, time.time() - t0, list(truncated), noop_actions, len(seen_keys),
                    revisits, actions_before_first_level, run_error,
                    [asdict(s) for s in trace] if keep_trace else [], partial=True,
                    current_level_index=cur_level + 1, current_level_actions=actions_this_level,
                    builder_invocations=(getattr(agent, "builder_invocations", None)
                                         if getattr(agent, "builder", None) else None),
                    inference_budget=(agent.inference_budget_state()
                                      if hasattr(agent, "inference_budget_state") else None)))
            except Exception:  # noqa: BLE001
                pass

        frame = next_frame
        frames.append(frame)
        if budget_exhausted:
            # Without env-level seeking we cannot skip ahead; stop this env.
            break

    # Any levels never reached are implicit 0s; scoring handles missing indices.
    return _build_record(
        game_id, n_levels, baselines, level_results, frame, resets, time.time() - t0,
        truncated, noop_actions, len(seen_keys), revisits, actions_before_first_level,
        run_error, [asdict(s) for s in trace] if keep_trace else [], partial=False,
        stop_reason=stop_reason,
        current_level_index=cur_level + 1, current_level_actions=actions_this_level,
        builder_invocations=(getattr(agent, "builder_invocations", None)
                             if getattr(agent, "builder", None) else None),
        inference_budget=(agent.inference_budget_state()
                          if hasattr(agent, "inference_budget_state") else None))


def _build_record(game_id, n_levels, baselines, level_results, frame, resets, elapsed_s,
                  truncated, noop_actions, distinct_frames, revisits, actions_before_first_level,
                  run_error, trace, *, partial, current_level_index=None, current_level_actions=0,
                  stop_reason=None, builder_invocations=None, inference_budget=None) -> EnvRunRecord:
    """Assemble an EnvRunRecord from loop state — used for BOTH the final return and the
    per-level partial snapshots (live viewing). `partial=True` marks a mid-game snapshot."""
    er = EnvResult(game_id=game_id, n_levels=n_levels, levels=level_results,
                   state=str(frame.state), resets=resets)
    recorded_indices = {l.level_index for l in level_results}
    unfinished_level_index = None
    unfinished_level_actions = 0
    if (
        current_level_actions
        and current_level_index is not None
        and 1 <= current_level_index <= n_levels
        and current_level_index not in recorded_indices
        and str(frame.state) != "GameState.WIN"
    ):
        unfinished_level_index = current_level_index
        unfinished_level_actions = current_level_actions
    completed_level_actions = sum(l.actions_taken for l in level_results if l.completed)
    return EnvRunRecord(
        game_id=game_id,
        n_levels=n_levels,
        baselines=baselines,
        levels_completed=er.levels_completed,
        total_actions=er.total_actions,
        completed_level_actions=completed_level_actions,
        unfinished_level_index=unfinished_level_index,
        unfinished_level_actions=unfinished_level_actions,
        total_actions_including_unfinished=er.total_actions + unfinished_level_actions,
        reset_actions_counted=True,
        resets=resets,
        final_state=str(frame.state),
        env_score=er.env_score(),
        level_results=[asdict(l) for l in level_results],
        wall_clock_s=round(elapsed_s, 2),
        truncated_levels=truncated,
        noop_actions=noop_actions,
        distinct_frames=distinct_frames,
        revisits=revisits,
        actions_before_first_level=actions_before_first_level,
        error=run_error,
        stop_reason=stop_reason,
        inference_budget=inference_budget,
        builder_invocations=builder_invocations,
        trace=trace,
        partial=partial,
    )


@dataclass
class RunSummary:
    approach: str
    split: str
    benchmark_score: float
    n_envs: int
    total_levels_completed: int
    total_actions: int
    total_actions_including_unfinished: int
    wall_clock_s: float
    timestamp: str
    config: dict
    envs: list[dict]
    diagnostics: dict = field(default_factory=dict)
    hypothesis: Optional[str] = None  # optional experiment label

    def generalization_note(self) -> str:
        return (
            "Local score is computed on public environments only; it does not establish "
            "performance on private or newly released environments. "
            "Track dev->holdout drop, not just this number."
        )
