"""Tycho-Lite switches from the review of the 25-game DeepSeek-V4 runs: image-cap drains that strip old
images but keep the turns (TYCHO_DRAIN_IMAGES=strip) and the time-budget actor prompt
(TYCHO_BUDGET_PROMPT). Both are off by default (upstream)."""

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
        "TYCHO_VISION": "1", "TYCHO_TEXT_GRID": "diff", "TYCHO_SANDBOX_RUNTIME": "host",
        "TYCHO_ANIMATION_SUMMARY": "0", "TYCHO_HISTORY": "drain", "TYCHO_IMAGE_CAP": "3", "TYCHO_IMAGE_KEEP": "1",
    })
    yield os.environ
    os.environ.clear()
    os.environ.update(old)


def _grid(t: int) -> list[list[int]]:
    g = [[0] * 16 for _ in range(16)]
    g[4 + t % 8][4 + (t // 8) % 8] = 9
    return g


def _agent(td: str):
    from tycho.agent.agent import TychoAgent, _image_payload_count

    class StubTycho(TychoAgent):
        def __init__(self):
            super().__init__()
            self.images_sent: list[int] = []
            self.frames_sent: list[int] = []

        def _chat(self, call_type, max_tokens=None):
            self.images_sent.append(_image_payload_count(self.history))
            self.frames_sent.append(sum(1 for m in self.history if self._is_frame_user_message(m)))
            return {"text": "", "usage": {"in": 2000, "out": 50}, "tool_calls": [
                {"id": f"a{len(self.images_sent)}", "name": "take_action", "input": {"action": "ACTION1"}}]}

    agent = StubTycho()
    agent.reset("budget_game", ACTIONS, ws_root=td)
    return agent


def _play(agent, turns: int):
    for t in range(turns):
        agent.choose_action([], SimpleNamespace(frame=[_grid(t)], levels_completed=0, state="NOT_FINISHED"), ACTIONS)


def test_image_cap_strips_images_and_keeps_every_turn(lite_env):
    lite_env["TYCHO_DRAIN_IMAGES"] = "strip"
    with TemporaryDirectory() as td:
        agent = _agent(td)
        _play(agent, 10)
        assert max(agent.images_sent) <= 3                       # never over the server's image limit
        assert agent.frames_sent == list(range(1, 11))            # no frame-turn was dropped
        assert int(agent.token_stats.get("context_drains", 0)) == 0
        assert int(agent.token_stats["image_strips"]) >= 3
        from tycho.agent.agent import _image_payload_count

        first = agent.history[0]                                  # the oldest turn: text kept, image gone
        assert "Level 0, turn 0" in str(first["content"]) and _image_payload_count([first]) == 0


def test_image_cap_drop_is_upstream(lite_env):
    with TemporaryDirectory() as td:
        agent = _agent(td)
        _play(agent, 10)
        assert max(agent.images_sent) <= 3
        assert int(agent.token_stats.get("context_drains", 0)) >= 3   # whole turns dropped
        assert agent.frames_sent[-1] < 10
        assert "image_strips" not in agent.token_stats


def test_drain_images_rejects_unknown_values(lite_env):
    from tycho.config.settings import TychoSettings

    lite_env["TYCHO_DRAIN_IMAGES"] = "evict"
    with pytest.raises(ValueError):
        TychoSettings.from_env()


def test_budget_prompt_replaces_the_free_tool_call_cost_model(lite_env):
    lite_env.update({"TYCHO_BATCH_ACTIONS": "1"})
    with TemporaryDirectory() as td:
        plain = _agent(td)._sys
    lite_env["TYCHO_BUDGET_PROMPT"] = "1"
    with TemporaryDirectory() as td:
        budget = _agent(td)._sys
    assert "Tool calls are free for scoring" in plain and "fixed wall-clock budget" not in plain
    assert "Tool calls are free for scoring" not in budget and "fixed wall-clock budget" in budget
    assert "as many levels as possible" in budget and "do not RESET to get a clean look" in budget
    assert "Probes can be batched too" in budget and "Commit single actions when the next outcome" not in budget
    # the paragraph break after the cost-model paragraph survives the inline {% endif %}
    assert "reach the goal.\n\nThe full grid text is not inlined" in budget
