"""Tycho-Lite level-start switches: the grid and carry-forward notes inlined on turn 0
(TYCHO_LEVEL_START_INLINE), the auto-imported workspace helpers.py (TYCHO_PY_HELPERS) and the builder
kickoff with its notes inlined (TYCHO_BUILDER_INLINE_NOTES). Each is off by default (upstream)."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from arcengine import GameAction

ACTIONS = [GameAction.ACTION1, GameAction.ACTION2]


@pytest.fixture
def lite_env():
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy", "LLM_BACKEND": "openai", "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy", "MPLCONFIGDIR": "/private/tmp", "TYCHO_MODE": "no_world_model",
        # vision on: with it off, text_grid=diff is forced back to the full grid (no board channel)
        "TYCHO_VISION": "1", "TYCHO_TEXT_GRID": "diff", "TYCHO_SANDBOX_RUNTIME": "host",
        "TYCHO_ANIMATION_SUMMARY": "0",
    })
    yield os.environ
    os.environ.clear()
    os.environ.update(old)


def _frame(grids, levels_completed=0):
    return SimpleNamespace(frame=grids, levels_completed=levels_completed, state="NOT_FINISHED")


def _agent(td: str):
    from tycho.agent.agent import TychoAgent

    class StubTycho(TychoAgent):
        def __init__(self):
            super().__init__()
            self.prompts: list[str] = []

        def _chat(self, call_type, max_tokens=None):
            if call_type == "level_summary":
                return {"text": "done", "tool_calls": []}
            content = self.history[-1].get("content")
            self.prompts.append("\n".join(p.get("text", "") for p in content) if isinstance(content, list)
                                else str(content))
            return {"text": "", "tool_calls": [{"id": f"a{len(self.prompts)}", "name": "take_action",
                                                 "input": {"action": "ACTION1"}}]}

    agent = StubTycho()
    agent.reset("level_start_game", ACTIONS, ws_root=td)
    return agent


def _play_into_level_1(agent, note: str | None):
    start, moved, terminal, next_init = [[0, 1], [1, 0]], [[1, 1], [1, 0]], [[2, 2], [2, 2]], [[3, 3], [4, 4]]
    agent.choose_action([], _frame([start]), ACTIONS)
    agent.choose_action([], _frame([moved]), ACTIONS)
    if note is not None:
        agent.ws.write_file("notes/level_0_insights.md", note)
    agent.ws.record_solved(0, "ACTION1")
    agent.ws.record_terminal(0, terminal, action="ACTION1")
    agent.choose_action([], _frame([terminal, next_init], 1), ACTIONS)


def test_level_start_inline_grid_and_notes(lite_env):
    lite_env["TYCHO_LEVEL_START_INLINE"] = "1"
    with TemporaryDirectory() as td:
        agent = _agent(td)
        _play_into_level_1(agent, "# Level 0 Insights\nGoal: make every panel match its clue.\n")
        game_start, turn_1, level_start = agent.prompts
        assert "Current grid (rows" in game_start and "not inlined here" not in game_start
        assert "not inlined here" in turn_1 and "Current grid (rows" not in turn_1
        assert "Current grid (rows" in level_start
        assert "Goal: make every panel match its clue." in level_start
        assert "inlined below" in level_start and "read it before acting" not in level_start


def test_level_start_inline_off_is_upstream(lite_env):
    with TemporaryDirectory() as td:
        agent = _agent(td)
        _play_into_level_1(agent, "# Level 0 Insights\nGoal: make every panel match its clue.\n")
        assert all("not inlined here" in p for p in agent.prompts)
        assert "If notes/level_0_insights.md exists, read it before acting." in agent.prompts[-1]
        assert "make every panel match" not in agent.prompts[-1]


def test_inlined_fallback_note_drops_the_terminal_grid(lite_env):
    lite_env["TYCHO_LEVEL_START_INLINE"] = "1"
    with TemporaryDirectory() as td:
        agent = _agent(td)
        _play_into_level_1(agent, None)  # the stub scribe writes nothing: the deterministic fallback
        note = agent.ws.read_file("notes/level_0_insights.md")
        assert "Deterministic fallback note" in note and "terminal_grid" in note
        level_start = agent.prompts[-1]
        assert "Deterministic fallback note" in level_start and "ACTION1" in level_start
        assert "terminal_grid" not in level_start and "(terminal frame: level_0/terminal.txt)" in level_start


SUMMARY = ("1. Objects: the 2x2 blue block (player), the grey HUD bar (move budget).\n"
           "2. Actions: ACTION1 moves the block up one cell (2 turns confirm); ACTION2 untested.\n"
           "3. Goal: the block reaches the bottom-right corner; false on turns 0-1, true on the winning frame.\n"
           "4. Unexplained: none.\n5. Level details: ACTION1 twice.\n"
           "6. Next level checklist: probe ACTION2 and any new colors first.")


def _summary_agent(td: str, summary_replies: list[dict]):
    from tycho.agent.agent import TychoAgent

    class StubTycho(TychoAgent):
        def __init__(self):
            super().__init__()
            self.prompts: list[str] = []
            self.summary_prompts: list[str] = []

        def _chat(self, call_type, max_tokens=None):
            content = self.history[-1].get("content")
            text = "\n".join(p.get("text", "") for p in content) if isinstance(content, list) else str(content)
            if call_type == "level_summary":
                self.summary_prompts.append(text)
                return summary_replies.pop(0) if summary_replies else {"text": "", "tool_calls": []}
            self.prompts.append(text)
            return {"text": "", "tool_calls": [{"id": f"a{len(self.prompts)}", "name": "take_action",
                                                 "input": {"action": "ACTION1"}}]}

    agent = StubTycho()
    agent.reset("summary_game", ACTIONS, ws_root=td)
    return agent


def test_verified_summary_is_written_by_the_harness_from_the_reply_text(lite_env):
    lite_env.update({"TYCHO_SCRIBE_MODE": "verified", "TYCHO_LEVEL_START_INLINE": "1"})
    with TemporaryDirectory() as td:
        agent = _summary_agent(td, [{"text": SUMMARY, "tool_calls": []}])
        _play_into_level_1(agent, None)
        prompt = agent.summary_prompts[0]
        assert "Observed log of this level" in prompt and "turn 1: ACTION1 ->" in prompt
        assert "winning action: ACTION1 -> level complete" in prompt and "Do not take an action" in prompt
        note = agent.ws.read_file("notes/level_0_insights.md")
        assert note.startswith("# Level 0 insights (level summary)") and "ACTION2 untested" in note
        assert "ACTION2 untested" in agent.prompts[-1]            # inlined at the next level's start


def test_verified_summary_falls_back_to_the_action_log(lite_env):
    lite_env.update({"TYCHO_SCRIBE_MODE": "verified"})
    tool_only = {"text": "", "tool_calls": [{"id": "r1", "name": "read_file", "input": {"path": "level_0/terminal.json"}}]}
    with TemporaryDirectory() as td:
        agent = _summary_agent(td, [tool_only, dict(tool_only)])
        _play_into_level_1(agent, None)
        assert len(agent.summary_prompts) == 2 and "Do not call tools" in agent.summary_prompts[1]
        note = agent.ws.read_file("notes/level_0_insights.md")
        assert "Deterministic fallback note" in note and "turn 1: ACTION1 ->" in note and "terminal_grid" not in note


def test_an_empty_note_is_replaced_by_the_fallback(lite_env):
    with TemporaryDirectory() as td:
        agent = _summary_agent(td, [{"text": "", "tool_calls": [{"id": "w1", "name": "write_file", "input": {
            "path": "notes/level_0_insights.md", "content": ""}}]}])
        _play_into_level_1(agent, None)
        assert "Deterministic fallback note" in agent.ws.read_file("notes/level_0_insights.md")


def test_level_start_probe_lists_what_is_new(lite_env):
    lite_env.update({"TYCHO_LEVEL_START_INLINE": "1", "TYCHO_LEVEL_START_PROBE": "1"})
    with TemporaryDirectory() as td:
        agent = _summary_agent(td, [{"text": SUMMARY, "tool_calls": []}])
        start, moved, terminal = [[0, 1], [1, 0]], [[1, 1], [1, 0]], [[2, 2], [2, 2]]
        next_init = [[0] * 8 for _ in range(8)]
        next_init[2][2] = next_init[2][3] = 7                    # a color the first board did not have
        agent.choose_action([], _frame([start]), ACTIONS)
        agent.choose_action([], _frame([moved]), ACTIONS)
        agent.ws.record_solved(0, "ACTION1")
        agent.ws.record_terminal(0, terminal, action="ACTION1")
        agent.choose_action([], _frame([terminal, next_init], 1), ACTIONS)
        level_start = agent.prompts[-1]
        assert "### New on this board (compared with level 0's first board)" in level_start
        assert "color 7: new on this board, 1 object(s)" in level_start and "probe what is new" in level_start
        lite_env["TYCHO_LEVEL_START_PROBE"] = "0"
    with TemporaryDirectory() as td:
        agent = _summary_agent(td, [])
        _play_into_level_1(agent, "# Level 0 Insights\nnote\n")
        assert "New on this board" not in agent.prompts[-1]


def test_py_helpers_persist_across_run_python_calls(lite_env):
    from tycho.workspace.agent_tools import HELPERS_FILE, tool_specs

    lite_env["TYCHO_PY_HELPERS"] = "1"
    with TemporaryDirectory() as td:
        agent = _agent(td)
        agent.choose_action([], _frame([[[0, 10], [15, 3]]]), ACTIONS)
        run = lambda code: agent.exec.execute("run_python", {"code": code})  # noqa: E731
        assert "0a" in run("show(grid)") and "f3" in run("show(grid)")
        helpers = agent.ws.dir / HELPERS_FILE
        helpers.write_text(helpers.read_text() + "\n\ndef corner(g):\n    return int(g[0, 0])\n\ngrid = None\n")
        assert run("print(corner(grid), grid.shape)").strip() == "0 (2, 2)"  # helpers never shadow `grid`
        helpers.write_text("def broken(:\n")
        out = run("print('still runs')")
        assert "[helpers.py failed to import (SyntaxError" in out and "still runs" in out
        desc = next(t["description"] for t in agent._tools_spec if t["name"] == "run_python")
        assert "helpers.py (workspace root) is imported into EVERY run_python call" in desc
        plain = next(t["description"] for t in tool_specs("generic") if t["name"] == "run_python")
        assert "You may create reusable Python helper modules with write_file" in plain


def test_py_helpers_off_is_upstream(lite_env):
    with TemporaryDirectory() as td:
        agent = _agent(td)
        agent.choose_action([], _frame([[[0, 1], [1, 0]]]), ACTIONS)
        assert not (agent.ws.dir / "helpers.py").exists()
        assert "NameError" in agent.exec.execute("run_python", {"code": "show(grid)"})


def test_builder_kickoff_inlines_its_notes(lite_env, monkeypatch):
    from tycho.agent import builder as builder_mod

    kickoffs: list[str] = []

    def fake_chat_tools(history, tools, cfg, **kw):
        kickoffs.append(history[0]["content"])
        return {"text": "done", "tool_calls": [], "usage": {"in": 10, "out": 5}}

    monkeypatch.setattr(builder_mod, "chat_tools", fake_chat_tools)
    lite_env["TYCHO_MODE"] = "orchestrator"
    with TemporaryDirectory() as td:
        agent = _agent(td)
        agent.choose_action([], _frame([[[0, 1], [1, 0]]]), ACTIONS)
        agent.ws.write_file("notes/actor_beliefs.md", "the block moves 5 cells per action")
        agent.builder.build(0, hint="stuck")
        assert "First read notes/actor_beliefs.md" in kickoffs[-1] and "5 cells" not in kickoffs[-1]
        lite_env["TYCHO_BUILDER_INLINE_NOTES"] = "1"
        agent.builder.build(0, hint="stuck")
        assert "First read" not in kickoffs[-1] and "inlined below" in kickoffs[-1]
        assert "### notes/actor_beliefs.md\nthe block moves 5 cells per action" in kickoffs[-1]
        assert "### world_model.py" not in kickoffs[-1]  # still the seed: nothing to show
