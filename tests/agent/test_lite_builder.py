"""Tycho-Lite world-model switches: the stuck-gated, per-level-capped builder (TYCHO_BUILDER_GATE=stuck,
TYCHO_BUILDER_STUCK_*, TYCHO_BUILDER_MAX_PER_LEVEL), the per-turn verify line (TYCHO_WM_STATUS_LINE) and
the batched plan hint (TYCHO_PLAN_BATCH_HINT)."""

from __future__ import annotations

import hashlib
import json
import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import GameAction

ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]


@pytest.fixture
def orch_env():
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy", "LLM_BACKEND": "openai", "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy", "MPLCONFIGDIR": "/private/tmp", "TYCHO_MODE": "orchestrator",
        "TYCHO_VISION": "0", "TYCHO_TEXT_GRID": "diff", "TYCHO_SANDBOX_RUNTIME": "host",
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


class FakeBuilder:
    """Stands in for WorldModelBuilder: records each build and reports builder output tokens."""

    def __init__(self, tokens_out=5000):
        self.builds: list[str] = []
        self.calls = 0
        self.tokens_out = tokens_out

    def build(self, level, hint="", max_calls=None, available_actions=None):
        self.builds.append(hint)
        self.calls = 2
        return "confidence: low", [{"tool": "fake"}]

    def take_token_stats(self):
        return {"tokens_in": 0, "tokens_out": self.tokens_out, "cache_read": 0, "cache_write": 0, "latency_ms": 0}


def _agent(td: str, *, tokens_per_call=1000, pull_builder=False):
    from tycho.agent.agent import TychoAgent

    class StubTycho(TychoAgent):
        def __init__(self):
            super().__init__()
            self.chats = 0
            self.prompts: list[str] = []

        def _chat(self, call_type, max_tokens=None):
            self.chats += 1
            last = self.history[-1] if self.history else {}
            if isinstance(last.get("content"), list):
                self.prompts.append("\n".join(p.get("text", "") for p in last["content"]))
            if pull_builder and self.chats % 2 == 1:
                call = {"id": f"b{self.chats}", "name": "invoke_builder", "input": {"reason": "help"}}
            else:
                call = {"id": f"a{self.chats}", "name": "take_action", "input": {"action": "ACTION1"}}
            return {"text": "", "tool_calls": [call], "usage": {"in": 2000, "out": tokens_per_call}}

    agent = StubTycho()
    agent.reset("orch_game", ACTIONS, ws_root=td)
    return agent


def test_stuck_gate_fires_per_token_threshold_and_respects_cap(orch_env):
    orch_env.update({"TYCHO_BUILDER_GATE": "stuck", "TYCHO_BUILDER_STUCK_TOKENS": "3000",
                     "TYCHO_BUILDER_STUCK_DEATHS": "0", "TYCHO_BUILDER_MAX_PER_LEVEL": "2"})
    with TemporaryDirectory() as td:
        agent = _agent(td, tokens_per_call=1000)
        agent.builder = FakeBuilder(tokens_out=5000)
        for t in range(12):
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
        # 1K actor tokens per turn: fires once 3K are spent, again at 6K (builder's 5K per run do not
        # count), then the per-level cap of 2 stops it
        assert len(agent.builder.builds) == 2
        assert all("stuck" in h and "WIN CONDITION" in h for h in agent.builder.builds)
        assert any("[auto world-model builder fired" in str(m.get("content")) for m in agent.history)


def test_stuck_gate_counters_reset_at_level_boundary(orch_env):
    orch_env.update({"TYCHO_BUILDER_GATE": "stuck", "TYCHO_BUILDER_STUCK_TOKENS": "3000",
                     "TYCHO_BUILDER_STUCK_DEATHS": "0", "TYCHO_BUILDER_MAX_PER_LEVEL": "1",
                     "TYCHO_SCRIBE_MAX_CALLS": "0"})
    with TemporaryDirectory() as td:
        agent = _agent(td, tokens_per_call=1000)
        agent.builder = FakeBuilder(tokens_out=0)
        for t in range(6):
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
        assert len(agent.builder.builds) == 1
        for t in range(6, 12):
            agent.choose_action([], _frame(_grid(t), levels_completed=1), ACTIONS)
        assert len(agent.builder.builds) == 2  # the new level gets its own run


def test_stuck_gate_fires_on_deaths(orch_env):
    orch_env.update({"TYCHO_BUILDER_GATE": "stuck", "TYCHO_BUILDER_STUCK_TOKENS": "0",
                     "TYCHO_BUILDER_STUCK_DEATHS": "2"})
    with TemporaryDirectory() as td:
        agent = _agent(td, tokens_per_call=10)
        agent.builder = FakeBuilder(tokens_out=0)
        agent.choose_action([], _frame(_grid(0)), ACTIONS)
        for t in (1, 2):
            agent._pending_game_over_event = {"level": 0, "turn": t, "action": "ACTION1"}
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
        assert len(agent.builder.builds) == 1


def test_gate_off_is_upstream(orch_env):
    with TemporaryDirectory() as td:
        agent = _agent(td, tokens_per_call=100000)
        agent.builder = FakeBuilder()
        for t in range(4):
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
        assert agent.builder.builds == []


def test_actor_pulled_builder_capped_per_level(orch_env):
    orch_env["TYCHO_BUILDER_MAX_PER_LEVEL"] = "1"
    with TemporaryDirectory() as td:
        agent = _agent(td, pull_builder=True)
        agent.builder = FakeBuilder(tokens_out=0)
        for t in range(3):
            agent.choose_action([], _frame(_grid(t)), ACTIONS)
        assert len(agent.builder.builds) == 1
        outputs = [r["output"] for m in agent.history if m.get("role") == "tool" for r in m["results"]]
        assert any(o.startswith("World-model builder skipped") for o in outputs)


def _write_plan(ws, grid, actions, level=0, turn=1):
    model = (ws.dir / "world_model.py").read_bytes()
    expected = [hashlib.sha256(np.asarray(_grid(100 + i), dtype=np.int16).tobytes()).hexdigest()
                for i in range(len(actions))]
    artifact = {"status": "validated", "plan_length": len(actions), "actions": actions,
                "expected_grid_sha256": expected, "world_model_sha256": hashlib.sha256(model).hexdigest(),
                "start": {"level": level, "turn": turn,
                          "grid_sha256": hashlib.sha256(np.asarray(grid, dtype=np.int16).tobytes()).hexdigest()}}
    (ws.dir / "notes").mkdir(exist_ok=True)
    (ws.dir / "notes" / "validated_plan.json").write_text(json.dumps(artifact))


def test_plan_batch_hint_lists_the_remaining_plan(orch_env):
    with TemporaryDirectory() as td:
        agent = _agent(td)
        plan = [{"action": "ACTION1"}, {"action": "ACTION6", "row": 3, "col": 4}, {"action": "ACTION2"}]
        _write_plan(agent.ws, _grid(0), plan)
        plain = agent.ws.validated_plan_hint(_grid(0), level=0, turn=1, available=["ACTION1"])
        assert "next action is ACTION1" in plain and "actions=" not in plain
        hint = agent.ws.validated_plan_hint(_grid(0), level=0, turn=1, available=["ACTION1"], batch_max=2)
        listed = json.loads(hint.split("actions=", 1)[1])
        assert listed == plan[:2]


def test_wm_status_line_reports_replay(orch_env):
    orch_env["TYCHO_WM_STATUS_LINE"] = "1"
    with TemporaryDirectory() as td:
        agent = _agent(td)
        agent.choose_action([], _frame(_grid(0)), ACTIONS)
        agent.choose_action([], _frame(_grid(1)), ACTIONS)
        assert not any("[world model]" in p for p in agent.prompts)  # seed model: no line
        from tycho.agent.builder import _SEED_MARKER
        model = agent.ws.dir / "world_model.py"
        model.write_text(model.read_text().replace(_SEED_MARKER, "pass"))
        agent.choose_action([], _frame(_grid(2)), ACTIONS)
        assert "[world model] reproduces" in agent.prompts[-1], agent.prompts[-1][:600]


def test_builder_starter_recipe_only_while_model_is_seed(orch_env, monkeypatch):
    from tycho.agent import builder as builder_mod

    kickoffs: list[str] = []

    def fake_chat_tools(history, tools, cfg, **kw):
        kickoffs.append(history[0]["content"])
        return {"text": "done", "tool_calls": [], "usage": {"in": 10, "out": 5}}

    monkeypatch.setattr(builder_mod, "chat_tools", fake_chat_tools)
    with TemporaryDirectory() as td:
        orch_env["TYCHO_BUILDER_STARTER"] = "1"
        agent = _agent(td)
        agent.choose_action([], _frame(_grid(0)), ACTIONS)
        agent.builder.build(0, hint="stuck")
        assert "still the untouched seed" in kickoffs[-1]
        model = agent.ws.dir / "world_model.py"
        model.write_text(model.read_text().replace(builder_mod._SEED_MARKER, "s.grid[0, 0] = 1"))
        agent.builder.build(0, hint="stuck")
        assert "still the untouched seed" not in kickoffs[-1]
        orch_env["TYCHO_BUILDER_STARTER"] = "0"
        model.write_text(model.read_text().replace("s.grid[0, 0] = 1", builder_mod._SEED_MARKER))
        agent.builder.build(0, hint="stuck")
        assert "still the untouched seed" not in kickoffs[-1]
