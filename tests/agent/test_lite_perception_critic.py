"""Object-level changes (TYCHO_OBJECT_DIFF, wmlib.frame_diff), kept run_python functions (TYCHO_PY_RETAIN),
competing goal hypotheses (TYCHO_GOAL_HYPOTHESES) and the stuck critic (TYCHO_STUCK_CRITIC). All off by default."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import GameAction

ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.RESET]


@pytest.fixture
def env():
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy", "LLM_BACKEND": "openai", "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy", "MPLCONFIGDIR": "/tmp", "TYCHO_MODE": "no_world_model",
        "TYCHO_VISION": "0", "TYCHO_TEXT_GRID": "full", "TYCHO_SANDBOX_RUNTIME": "host",
        "TYCHO_ANIMATION_SUMMARY": "0",
    })
    yield os.environ
    os.environ.clear()
    os.environ.update(old)


def test_frame_diff_names_moves_rotations_recolors_and_bars():
    from tycho.workspace.wmlib_template import frame_diff, frame_diff_text

    a = np.zeros((20, 20), int)
    b = a.copy()
    a[2:4, 2:4] = 5; b[7:9, 2:4] = 5                     # a 2x2 block moves down 5
    a[10:13, 10] = 7; a[12, 11] = 7                       # an L rotates in place
    b[10, 10:13] = 7; b[11, 10] = 7
    a[15:17, 15:17] = 3; b[15:17, 15:17] = 4              # a square changes color
    a[19, 0:10] = 9; b[19, 0:8] = 9                       # a bar shrinks
    b[0, 18:20] = 6                                       # something appears
    d = frame_diff(a, b)
    moved = {m["color"]: m for m in d["moved"]}
    assert moved[5]["delta"] == (5, 0) and "rotated" not in moved[5]
    assert moved[7]["rotated"] == 90
    assert d["changed_color"][0]["before"] == 3 and d["changed_color"][0]["after"] == 4
    assert d["resized"][0]["color"] == 9 and d["resized"][0]["from_size"] == 10 and d["resized"][0]["size"] == 8
    assert d["appeared"][0]["color"] == 6 and not d["disappeared"]
    text = frame_diff_text(d)
    assert "color 5 (4 cells) down 5" in text and "rotated 90 deg clockwise" in text and "3->4" in text


def _agent(td: str, replies: list, critic_replies: list | None = None, monkeypatch=None):
    from tycho.agent import agent as agent_mod
    from tycho.agent.agent import TychoAgent

    critic_calls = []
    if monkeypatch is not None:
        def fake_chat_tools(history, tools, cfg, **kw):
            critic_calls.append({"history": history, "tools": tools, **kw})
            return {"text": (critic_replies or ["review"]).pop(0), "tool_calls": [],
                    "usage": {"in": 900, "out": 30000}}
        monkeypatch.setattr(agent_mod, "chat_tools", fake_chat_tools)

    class StubTycho(TychoAgent):
        def __init__(self):
            super().__init__()
            self.prompts: list[str] = []

        def _chat(self, call_type, max_tokens=None):
            last = self.history[-1]
            content = last.get("content")
            if last.get("role") != "tool":
                self.prompts.append("\n".join(p.get("text", "") for p in content) if isinstance(content, list)
                                    else str(content))
            calls = replies.pop(0) if replies else [{"id": f"a{len(self.prompts)}", "name": "take_action",
                                                     "input": {"action": "ACTION1"}}]
            return {"text": "", "tool_calls": calls, "usage": {"in": 1000, "out": 25000}}

    agent = StubTycho()
    agent.reset("perception_game", ACTIONS, ws_root=td)
    agent.critic_calls = critic_calls
    return agent


def _frame(t: int):
    g = [[0] * 12 for _ in range(12)]
    for r in range(2):
        for c in range(2):
            g[1 + 2 * t + r][2 + c] = 9                   # a 2x2 block moving down 2 per turn
    return SimpleNamespace(frame=[g], levels_completed=0, state="NOT_FINISHED")


def test_object_changes_lead_the_turn_report(env):
    env["TYCHO_OBJECT_DIFF"] = "1"
    with TemporaryDirectory() as td:
        agent = _agent(td, [])
        agent.choose_action([], _frame(0), ACTIONS)
        agent.choose_action([], _frame(1), ACTIONS)
        assert "Objects: moved: color 9 (4 cells) down 2" in agent.prompts[-1] and "Cells: " in agent.prompts[-1]
        desc = next(t["description"] for t in agent._tools_spec if t["name"] == "run_python")
        assert "wmlib.frame_diff(before, after)" in desc
    env["TYCHO_OBJECT_DIFF"] = "0"
    with TemporaryDirectory() as td:
        agent = _agent(td, [])
        agent.choose_action([], _frame(0), ACTIONS)
        agent.choose_action([], _frame(1), ACTIONS)
        assert "Objects:" not in agent.prompts[-1]


def test_functions_are_kept_after_a_clean_run_and_see_the_current_grid(env):
    env["TYCHO_PY_RETAIN"] = "1"
    with TemporaryDirectory() as td:
        agent = _agent(td, [])
        agent.choose_action([], _frame(0), ACTIONS)
        run = lambda code: agent.exec.execute("run_python", {"code": code})  # noqa: E731
        run("import collections\ndef block_row(g):\n    return int(np.argwhere(g == 9)[0][0])\nprint(block_row(grid))\n")
        run("def broken(x):\n    return x\nraise RuntimeError('boom')\n")            # failed run: not kept
        run("def grid(x):\n    return x\n")                                         # shadows a preloaded name: skipped
        assert "1" in run("print(block_row(grid))")
        agent.choose_action([], _frame(1), ACTIONS)                                  # the block moved down 2
        assert "3" in run("print(block_row(grid))")                                  # kept function, current grid
        assert "NameError" in run("broken(1)")
        assert "Functions kept from your earlier run_python calls" in agent.prompts[-1]
        assert "block_row(g)" in agent.prompts[-1] and "grid(" not in agent.prompts[-1].split("calls")[-1]


def test_goal_hypotheses_rule_is_in_the_actor_prompt_only_when_on(env):
    with TemporaryDirectory() as td:
        assert "Treat the goal as unknown" not in _agent(td, [])._sys
    env["TYCHO_GOAL_HYPOTHESES"] = "1"
    with TemporaryDirectory() as td:
        sys_prompt = _agent(td, [])._sys
        assert "Treat the goal as unknown until a level completes" in sys_prompt
        assert "two or three competing goal hypotheses" in sys_prompt


def test_stuck_critic_reviews_in_a_fresh_context(env, monkeypatch):
    env.update({"TYCHO_STUCK_CRITIC": "1", "TYCHO_STUCK_CRITIC_TOKENS": "40000"})
    with TemporaryDirectory() as td:
        agent = _agent(td, [], critic_replies=["1. Overlooked: the HUD glyph changed on turn 1."], monkeypatch=monkeypatch)
        for t in range(4):                                   # 25K actor tokens per decision
            agent.choose_action([], _frame(t), ACTIONS)
        assert len(agent.critic_calls) == 1                  # fired once the level passed 40K actor tokens
        call = agent.critic_calls[0]
        assert len(call["history"]) == 1 and call["tools"] == [] and call["call_type"] == "critic"
        request = call["history"][0]["content"][0]["text"]
        assert "spent" in request and "Observed log of this level" in request and "turn 1: ACTION1" in request
        assert any("[fresh-eyes review" in p and "HUD glyph changed" in p for p in agent.prompts)
        # the review's 30K output tokens do not count as actor tokens on the level
        assert agent._level_actor_tokens_out() == 25000 * len([p for p in agent.prompts])


def test_stuck_critic_off_is_upstream(env, monkeypatch):
    with TemporaryDirectory() as td:
        agent = _agent(td, [], monkeypatch=monkeypatch)
        for t in range(4):
            agent.choose_action([], _frame(t), ACTIONS)
        assert agent.critic_calls == []
