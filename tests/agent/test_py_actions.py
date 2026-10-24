"""run_python scripts that play (TYCHO_PY_ACTIONS): action(...) inside the script, kept alive across game
steps. Off by default; host sandbox only."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from arcengine import GameAction

ACTIONS = [GameAction.ACTION1, GameAction.ACTION2]


@pytest.fixture
def env():
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy", "LLM_BACKEND": "openai", "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy", "MPLCONFIGDIR": "/tmp", "TYCHO_MODE": "no_world_model",
        "TYCHO_VISION": "0", "TYCHO_TEXT_GRID": "full", "TYCHO_SANDBOX_RUNTIME": "host",
        "TYCHO_ANIMATION_SUMMARY": "0", "TYCHO_PY_ACTIONS": "1",
    })
    yield os.environ
    os.environ.clear()
    os.environ.update(old)


def _frame(t: int, levels=0, state="NOT_FINISHED"):
    g = [[0] * 8 for _ in range(8)]
    g[1 + t % 6][2] = 9
    return SimpleNamespace(frame=[g], levels_completed=levels, state=state)


def _agent(td: str, replies: list[list[dict]]):
    from tycho.agent.agent import TychoAgent

    class StubTycho(TychoAgent):
        def __init__(self):
            super().__init__()
            self.prompts: list[str] = []
            self.tool_results: list[list[str]] = []

        def _chat(self, call_type, max_tokens=None):
            last = self.history[-1]
            if last.get("role") == "tool":
                self.tool_results.append([r["output"] for r in last["results"]])
            else:
                content = last.get("content")
                self.prompts.append("\n".join(p.get("text", "") for p in content) if isinstance(content, list)
                                    else str(content))
            calls = replies.pop(0) if replies else [{"id": "dflt", "name": "take_action", "input": {"action": "ACTION2"}}]
            return {"text": "", "tool_calls": calls}

    agent = StubTycho()
    agent.reset("act_game", ACTIONS, ws_root=td)
    return agent


def _tool_output(agent, call_id: str) -> str:
    for m in agent.history:
        for r in m.get("results", []) or []:
            if r["id"] == call_id:
                return r["output"]
    raise AssertionError(f"no tool result for {call_id}")


def _py(code: str, cid="p1"):
    return {"id": cid, "name": "run_python", "input": {"code": code}}


def test_a_script_plays_several_actions_and_reports_its_output(env):
    code = "for a in ['ACTION1', 'ACTION1', 'ACTION2']:\n    o = action(a)\n    print(a, o['changed'].split(';')[0], o['state'])\n"
    with TemporaryDirectory() as td:
        agent = _agent(td, [[_py(code)]])
        played = [agent.choose_action([], _frame(t), ACTIONS).action.name for t in range(4)]
        assert played[:3] == ["ACTION1", "ACTION1", "ACTION2"]       # three actions, one LLM decision
        assert "Script is playing" in _tool_output(agent, "p1")
        report = agent.prompts[-1]                                   # the turn after the script ended
        assert "[script report] Your run_python script played 3 action(s): ACTION1, ACTION1, ACTION2" in report
        assert "its code finished" in report and "ACTION2 2 cells changed NOT_FINISHED" in report
        assert len(agent.prompts) == 2                               # no LLM call during the script


def test_a_level_completion_stops_the_script(env):
    code = "for k in range(10):\n    action('ACTION1')\nprint('never reached')\n"
    with TemporaryDirectory() as td:
        agent = _agent(td, [[_py(code)]])
        agent.choose_action([], _frame(0), ACTIONS)
        agent.choose_action([], _frame(1), ACTIONS)
        agent.ws.record_solved(0, "ACTION1")
        agent.ws.record_terminal(0, _frame(2).frame[0], action="ACTION1")
        agent.choose_action([], _frame(3, levels=1), ACTIONS)
        report = agent.prompts[-1]
        assert "played 2 action(s)" in report and "completed the level" in report
        assert "[script stopped by the harness: the level completed]" in report and "never reached" not in report


def test_illegal_actions_and_reset_are_rejected_inside_the_script(env):
    code = ("for a in ['ACTION9', 'RESET']:\n    try:\n        action(a)\n    except ValueError as e:\n"
            "        print('rejected', a, '-', str(e)[:40])\naction('ACTION2')\n")
    with TemporaryDirectory() as td:
        agent = _agent(td, [[_py(code)]])
        offered = ACTIONS + [GameAction.RESET]
        assert agent.choose_action([], _frame(0), offered).action.name == "ACTION2"
        agent.choose_action([], _frame(1), offered)
        report = agent.prompts[-1]
        assert "rejected ACTION9" in report and "rejected RESET - RESET cannot be played from a script" in report


def test_take_action_in_the_same_reply_is_not_executed(env):
    with TemporaryDirectory() as td:
        agent = _agent(td, [[{"id": "t1", "name": "take_action", "input": {"action": "ACTION2"}},
                             _py("action('ACTION1')\n")]])
        assert agent.choose_action([], _frame(0), ACTIONS).action.name == "ACTION1"
        assert _tool_output(agent, "t1").startswith("NOT executed: a run_python script")


def test_a_script_that_does_not_act_is_a_normal_run_python(env):
    with TemporaryDirectory() as td:
        agent = _agent(td, [[_py("print('just looking', grid.shape)  # no action( here\n")]])
        agent.choose_action([], _frame(0), ACTIONS)
        assert "just looking (8, 8)" in _tool_output(agent, "p1")


def test_py_actions_off_is_upstream(env):
    env["TYCHO_PY_ACTIONS"] = "0"
    with TemporaryDirectory() as td:
        agent = _agent(td, [[_py("action('ACTION1')\n")]])
        assert agent.choose_action([], _frame(0), ACTIONS).action.name == "ACTION2"   # the stub's default
        assert "NameError" in _tool_output(agent, "p1")
        desc = next(t["description"] for t in agent._tools_spec if t["name"] == "run_python")
        assert "PLAY from run_python" not in desc
