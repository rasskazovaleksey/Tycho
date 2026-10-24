"""Multi-action commits (TYCHO_BATCH_ACTIONS): take_action with an `actions` sequence.

The sequence's later actions are played without LLM calls; every intermediate frame is still recorded
in the workspace; guards stop the sequence early (no-op inside the board, level completed, GAME_OVER);
the next real turn reports what ran. With the flag off the tool surface is upstream's.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from arcengine import GameAction

ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]


@pytest.fixture
def batch_env():
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy", "LLM_BACKEND": "openai", "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy", "MPLCONFIGDIR": "/private/tmp", "TYCHO_MODE": "single",
        "TYCHO_VISION": "0", "TYCHO_TEXT_GRID": "full", "TYCHO_SANDBOX_RUNTIME": "host",
        "TYCHO_ANIMATION_SUMMARY": "0", "TYCHO_BATCH_ACTIONS": "1", "TYCHO_BATCH_NOOP_GUARD_FROM_LEVEL": "0",
    })
    yield
    os.environ.clear()
    os.environ.update(old)


def _grid(marker: int = 0, border_mark: int = 0) -> list[list[int]]:
    """12x12 board; `marker` moves a cell inside the interior, `border_mark` paints the top-left corner
    (inside the 4-cell border the no-op guard ignores, like a timer bar)."""
    g = [[0] * 12 for _ in range(12)]
    g[4 + marker % 4][4 + marker // 4] = 9
    g[0][0] = border_mark
    return g


def _frame(grid, levels_completed=0, state="NOT_FINISHED"):
    return SimpleNamespace(frame=[grid], levels_completed=levels_completed, state=state)


def _agent(replies: list[dict], td: str):
    from tycho.agent.agent import TychoAgent

    class StubTycho(TychoAgent):
        def __init__(self):
            super().__init__()
            self.chats = 0
            self.prompts: list[str] = []

        def _chat(self, call_type, max_tokens=None):
            self.chats += 1
            last = self.history[-1] if self.history else {}
            content = last.get("content")
            if isinstance(content, list):
                self.prompts.append("\n".join(p.get("text", "") for p in content if isinstance(p, dict)))
            reply = replies[min(self.chats, len(replies)) - 1]
            return {"text": "", "tool_calls": [dict(reply, id=f"c{self.chats}")]}

    agent = StubTycho()
    agent.reset("batch_game", ACTIONS, ws_root=td)
    return agent


def _seq(*names) -> dict:
    return {"name": "take_action", "input": {"actions": [{"action": n} for n in names]}}


def _single(name) -> dict:
    return {"name": "take_action", "input": {"action": name}}


def test_sequence_runs_without_llm_calls_and_reports(batch_env):
    with TemporaryDirectory() as td:
        agent = _agent([_seq("ACTION1", "ACTION2", "ACTION3"), _single("ACTION4")], td)
        first = agent.choose_action([], _frame(_grid(0)), ACTIONS)
        second = agent.choose_action([], _frame(_grid(1)), ACTIONS)
        third = agent.choose_action([], _frame(_grid(2)), ACTIONS)
        assert [first.action, second.action, third.action] == [GameAction.ACTION1, GameAction.ACTION2,
                                                                GameAction.ACTION3]
        assert agent.chats == 1
        assert first.reasoning["batch"] == {"i": 0, "n": 3}
        assert second.reasoning["src"] == "batch" and second.reasoning["i"] == 1
        # intermediate frames are in the workspace like any other turn
        ld = Path(agent.ws.dir) / "level_0"
        assert (ld / "turn_001.txt").is_file() and (ld / "turn_002.txt").is_file()

        fourth = agent.choose_action([], _frame(_grid(3)), ACTIONS)
        assert agent.chats == 2 and fourth.action == GameAction.ACTION4
        assert fourth.reasoning["batch_report"] == {"planned": 3, "executed": 3, "stop": None,
                                                    "skipped": 0, "dropped": 0}
        assert "[sequence report]" in agent.prompts[-1] and "All 3 executed" in agent.prompts[-1]
        assert "Cell changes over the whole sequence" in agent.prompts[-1]


def test_noop_inside_board_stops_the_sequence(batch_env):
    with TemporaryDirectory() as td:
        agent = _agent([_seq("ACTION1", "ACTION2", "ACTION3"), _single("ACTION4")], td)
        agent.choose_action([], _frame(_grid(0)), ACTIONS)
        # ACTION1 only changed the border (a timer tick): the guard stops before ACTION2
        nxt = agent.choose_action([], _frame(_grid(0, border_mark=5)), ACTIONS)
        assert agent.chats == 2 and nxt.action == GameAction.ACTION4
        report = nxt.reasoning["batch_report"]
        assert report["stop"] == "no_op" and report["executed"] == 1 and report["skipped"] == 2
        assert "EXECUTED but produced no change inside the board area" in agent.prompts[-1]
        assert "Actions 2-3 (ACTION2, ACTION3) were SKIPPED" in agent.prompts[-1]


def test_level_completion_stops_the_sequence(batch_env):
    with TemporaryDirectory() as td:
        agent = _agent([_seq("ACTION1", "ACTION2"), _single("ACTION3")], td)
        agent.choose_action([], _frame(_grid(0)), ACTIONS)
        nxt = agent.choose_action([], _frame(_grid(5), levels_completed=1), ACTIONS)
        assert nxt.reasoning["batch_report"]["stop"] == "level_completed"
        assert "completed the level" in agent.prompts[-1]
        assert agent.level == 1


def test_game_over_closes_the_sequence(batch_env):
    with TemporaryDirectory() as td:
        agent = _agent([_seq("ACTION1", "ACTION2"), _single("ACTION3")], td)
        agent.choose_action([], _frame(_grid(0)), ACTIONS)
        agent.note_external_reset()  # the harness RESETs after GAME_OVER without asking the agent
        nxt = agent.choose_action([], _frame(_grid(0)), ACTIONS)
        assert nxt.reasoning["batch_report"]["stop"] == "game_over"
        assert "caused GAME_OVER" in agent.prompts[-1]


def test_invalid_sequence_commits_nothing(batch_env):
    with TemporaryDirectory() as td:
        bad = {"name": "take_action", "input": {"actions": [{"action": "ACTION1"}, {"action": "RESET"}]}}
        agent = _agent([bad, _single("ACTION2")], td)
        choice = agent.choose_action([], _frame(_grid(0)), ACTIONS + [GameAction.RESET])
        assert choice.action == GameAction.ACTION2 and agent.chats == 2 and agent._batch is None
        tool_text = "\n".join(r.get("output", "") for m in agent.history if m.get("role") == "tool"
                              for r in m.get("results", []))
        assert "RESET may only be the first action of a sequence" in tool_text


def test_sequence_is_capped_at_batch_max(batch_env):
    os.environ["TYCHO_BATCH_MAX"] = "2"
    with TemporaryDirectory() as td:
        agent = _agent([_seq("ACTION1", "ACTION2", "ACTION3"), _single("ACTION4")], td)
        agent.choose_action([], _frame(_grid(0)), ACTIONS)
        agent.choose_action([], _frame(_grid(1)), ACTIONS)
        nxt = agent.choose_action([], _frame(_grid(2)), ACTIONS)
        assert nxt.reasoning["batch_report"]["dropped"] == 1
        assert "Only the first 2 were accepted" in agent.prompts[-1]


def test_flag_off_keeps_upstream_tool_surface(batch_env):
    os.environ["TYCHO_BATCH_ACTIONS"] = "0"
    from tycho.agent.agent import TychoAgent

    agent = TychoAgent()
    take = next(t for t in agent._tools_spec if t["name"] == "take_action")
    assert "actions" not in take["schema"]["properties"]
    assert "Committing several actions at once" not in agent._sys

    os.environ["TYCHO_BATCH_ACTIONS"] = "1"
    agent = TychoAgent()
    take = next(t for t in agent._tools_with_available_actions(["ACTION1", "ACTION2"]) if t["name"] == "take_action")
    assert take["schema"]["properties"]["actions"]["items"]["properties"]["action"]["enum"] == ["ACTION1", "ACTION2"]
    assert "Committing several actions at once" in agent._sys


def test_before_decision_runs_at_decisions_not_mid_sequence(batch_env):
    """The hook an embedding runner yields its inference slot in: once before each LLM turn, including the
    turn that closes a finished sequence, and never while a sequence plays out."""
    with TemporaryDirectory() as td:
        agent = _agent([_seq("ACTION1", "ACTION2"), _single("ACTION4")], td)
        seen = []
        agent.before_decision = lambda frame: seen.append(agent.chats)
        agent.choose_action([], _frame(_grid(0)), ACTIONS)  # decision 1 commits the sequence
        agent.choose_action([], _frame(_grid(1)), ACTIONS)  # its second action: no decision
        assert seen == [0]
        agent.choose_action([], _frame(_grid(2)), ACTIONS)  # the sequence is done: decision 2
        assert seen == [0, 1] and agent.chats == 2
