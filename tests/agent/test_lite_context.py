"""Tycho-Lite switches: drain history (TYCHO_HISTORY=drain), strict take_action receipts
(TYCHO_STRICT_RECEIPTS) and the capped level summary (TYCHO_SCRIBE_MAX_CALLS / _MAX_TOKENS).

Drain is the prefix-cache discipline of Franzen's harness: the transcript is append-only between
drains, so each request starts with the previous request's messages unchanged, and one drain drops the
oldest whole frame-turns once the prompt passes the budget.
"""

from __future__ import annotations

import copy
import json
import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from arcengine import GameAction

ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]


@pytest.fixture
def lite_env():
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy", "LLM_BACKEND": "openai", "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy", "MPLCONFIGDIR": "/private/tmp", "TYCHO_MODE": "no_world_model",
        "TYCHO_VISION": "0", "TYCHO_TEXT_GRID": "full", "TYCHO_SANDBOX_RUNTIME": "host",
        "TYCHO_ANIMATION_SUMMARY": "0",
    })
    yield os.environ
    os.environ.clear()
    os.environ.update(old)


def _grid(t: int) -> list[list[int]]:
    g = [[0] * 16 for _ in range(16)]
    g[4 + t % 8][4 + (t // 8) % 8] = 9
    return g


def _frame(grid, levels_completed=0):
    return SimpleNamespace(frame=[grid], levels_completed=levels_completed, state="NOT_FINISHED")


def _message_key(m: dict) -> str:
    return json.dumps(m, sort_keys=True, default=lambda v: f"<{type(v).__name__}>")


def _agent(td: str, reply_for):
    """A TychoAgent whose LLM is ``reply_for(agent, call_type)``. Every request's history is recorded,
    and usage reports a prompt size proportional to the transcript (so drains can trigger)."""
    from tycho.agent.agent import TychoAgent

    class StubTycho(TychoAgent):
        def __init__(self):
            super().__init__()
            self.requests: list[tuple[str, list[str], int]] = []

        def _chat(self, call_type, max_tokens=None):
            self.requests.append((call_type, [_message_key(m) for m in self.history],
                                  int(self.token_stats.get("context_drains", 0))))
            self.max_tokens_seen = getattr(self, "max_tokens_seen", []) + [max_tokens]
            reply = reply_for(self, call_type)
            prompt = sum(len(k) for k in self.requests[-1][1]) // 3 + 1000
            return {**reply, "usage": {"in": prompt, "out": 50}}

    agent = StubTycho()
    agent.reset("lite_game", ACTIONS, ws_root=td)
    return agent


def _act(name="ACTION1", n=1):
    def reply(agent, call_type):
        calls = [{"id": f"c{len(agent.requests)}_{i}", "name": "take_action", "input": {"action": name}}
                 for i in range(n)]
        return {"text": "", "tool_calls": calls}
    return reply


def test_drain_keeps_prefix_stable_and_drains_in_batches(lite_env):
    lite_env.update({"TYCHO_HISTORY": "drain", "TYCHO_CONTEXT_BUDGET_TOKENS": "12000",
                     "TYCHO_CONTEXT_DRAIN_TOKENS": "6000"})
    with TemporaryDirectory() as td:
        agent = _agent(td, _act())
        for t in range(40):
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
        drains = int(agent.token_stats.get("context_drains", 0))
        assert drains >= 1
        assert drains <= 40 // 4  # batched: a drain frees many turns, not one per turn
        for (_, prev, d0), (_, cur, d1) in zip(agent.requests, agent.requests[1:]):
            if d1 == d0:  # no drain in between: the previous request is a byte-identical prefix
                assert cur[:len(prev)] == prev
        # after a drain the transcript opens on a frame message carrying the drain note
        first = agent.history[0]
        assert first["role"] == "user" and agent._is_frame_user_message(first)
        assert any("[context drained:" in p.get("text", "") for p in first["content"])


def test_drain_never_edits_kept_messages(lite_env):
    lite_env.update({"TYCHO_HISTORY": "drain", "TYCHO_CONTEXT_BUDGET_TOKENS": "12000",
                     "TYCHO_CONTEXT_DRAIN_TOKENS": "6000"})
    with TemporaryDirectory() as td:
        agent = _agent(td, _act())
        seen: dict[int, tuple[dict, str]] = {}  # holds the message so its id() is never reused
        for t in range(30):
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
            for m in agent.history[1:]:  # [0] carries the drain note after a drain
                seen.setdefault(id(m), (m, _message_key(m)))
                assert _message_key(m) == seen[id(m)][1]


def test_tail_evict_default_unchanged(lite_env):
    with TemporaryDirectory() as td:
        agent = _agent(td, _act())
        assert agent.context_mode == "tail_evict"
        for t in range(5):
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
        assert "context_drains" not in agent.token_stats


def test_forced_drain_on_context_error_keeps_only_live_turn(lite_env):
    lite_env.update({"TYCHO_HISTORY": "drain"})
    with TemporaryDirectory() as td:
        agent = _agent(td, _act())
        for t in range(6):
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
        agent.history.append({"role": "user", "content": [{"text": "=== Level 0, turn 6 (live) ==="}]})
        trace: list[dict] = []
        assert agent._drain_history(trace, force=True)
        assert len(agent.history) == 1 and trace[-1]["force"]


def test_strict_receipts_reject_extra_take_action(lite_env):
    with TemporaryDirectory() as td:
        lite_env["TYCHO_STRICT_RECEIPTS"] = "1"
        agent = _agent(td, _act(n=2))
        choice = agent.choose_action([], _frame(_grid(0)), ACTIONS)
        assert choice.action == GameAction.ACTION1
        outputs = [r["output"] for r in agent.history[-1]["results"]]
        assert outputs[0].startswith("Action committed")
        assert outputs[1].startswith("NOT executed")


def test_receipts_default_is_upstream(lite_env):
    with TemporaryDirectory() as td:
        agent = _agent(td, _act(n=2))
        agent.choose_action([], _frame(_grid(0)), ACTIONS)
        outputs = [r["output"] for r in agent.history[-1]["results"]]
        assert all(o.startswith("Action committed") for o in outputs)


def test_scribe_caps(lite_env):
    lite_env.update({"TYCHO_SCRIBE_MAX_CALLS": "1", "TYCHO_SCRIBE_MAX_TOKENS": "1500"})

    def reply(agent, call_type):
        if call_type == "level_summary":  # keeps calling tools: only the cap stops it
            return {"text": "", "tool_calls": [{"id": f"s{len(agent.requests)}", "name": "ls", "input": {}}]}
        return _act()(agent, call_type)

    with TemporaryDirectory() as td:
        agent = _agent(td, reply)
        agent.choose_action([], _frame(_grid(0)), ACTIONS)
        agent.choose_action([], _frame(_grid(1), levels_completed=1), ACTIONS)
        summary_calls = [i for i, r in enumerate(agent.requests) if r[0] == "level_summary"]
        assert len(summary_calls) == 1
        assert agent.max_tokens_seen[summary_calls[0]] == 1500


def test_diff_text_grid_with_vision_sends_no_grid_text(lite_env):
    """Tycho-Lite's frame message: with the image attached, TYCHO_TEXT_GRID=diff inlines no 64x64 grid
    (Tycho forces it back on only when there is no image to look at)."""
    lite_env.update({"TYCHO_VISION": "1", "TYCHO_TEXT_GRID": "diff"})
    with TemporaryDirectory() as td:
        agent = _agent(td, _act())
        for t in range(2):
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
        frame_msgs = [m for m in agent.history if agent._is_frame_user_message(m)]
        text = "\n".join(p.get("text", "") for p in frame_msgs[-1]["content"])
        assert "Current grid (rows" not in text and "not inlined" in text
        assert any("image_png" in p for p in frame_msgs[-1]["content"])


def test_drain_shrinks_an_oversized_live_turn(lite_env):
    """The contract-probe case: one frame-turn with a 50-step tool loop. Emergency compaction grew it
    (51 -> 52 messages); drain keeps the frame message and the last tool steps, paired."""
    lite_env.update({"TYCHO_HISTORY": "drain", "TYCHO_CONTEXT_BUDGET_TOKENS": "5000"})
    with TemporaryDirectory() as td:
        agent = _agent(td, _act())
        frame = {"role": "user", "content": [{"text": "=== Level 0, turn 0 ==="}]}
        steps = []
        for i in range(25):
            steps.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"t{i}", "name": "run_python", "input": {"code": "x" * 500}}]})
            steps.append({"role": "tool", "results": [{"id": f"t{i}", "name": "run_python", "output": "y" * 1500}]})
        agent.history = [frame, *steps]
        agent._last_prompt_tokens = 20000
        trace: list[dict] = []
        assert agent._drain_history(trace)
        assert len(agent.history) == 1 + 2 * 4 and agent.history[1]["role"] == "assistant"
        assert agent.history[1]["tool_calls"][0]["id"] == "t21" and agent.history[-1]["role"] == "tool"
        assert "earlier tool step(s) of this turn were removed" in agent.history[0]["content"][0]["text"]
        trace.clear()
        assert agent._drain_history(trace, force=True)  # a context-limit error keeps only the last step
        assert len(agent.history) == 3
