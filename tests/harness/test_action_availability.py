import os
import importlib.util
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from arcengine import GameAction
from arcengine import GameState

from tycho.agent.agent import TychoAgent
from tycho.harness.actions import actor_actions, normalize_game_actions
from tycho.harness.agent import ActionChoice
from tycho.harness.harness import run_env
from tycho.workspace.workspace import GameWorkspace


def _set_env() -> dict:
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy",
        "LLM_BACKEND": "openai",
        "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy",
        "MPLCONFIGDIR": "/private/tmp",
        "TYCHO_MODE": "single",
        "TYCHO_VISION": "0",
        "TYCHO_TEXT_GRID": "full",
        "TYCHO_MAX_TOOL_STEPS": "1",
        "TYCHO_MAX_LLM_CALLS": "5",
    })
    return old


def _restore_env(old: dict) -> None:
    os.environ.clear()
    os.environ.update(old)


def _frame():
    return SimpleNamespace(
        frame=[[[0, 0], [1, 1]]],
        levels_completed=0,
        state="NOT_FINISHED",
    )


def test_actor_actions_add_reset_without_polluting_game_actions():
    game = normalize_game_actions([1, GameAction.ACTION2, "ACTION6"])
    assert game == [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6]
    assert GameAction.RESET not in game

    actor = actor_actions(game)
    assert actor == [GameAction.RESET, GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6]
    assert normalize_game_actions(actor) == game


def test_action7_remains_frame_declared_not_global():
    assert actor_actions([1, 7]) == [GameAction.RESET, GameAction.ACTION1, GameAction.ACTION7]
    assert GameAction.ACTION7 not in actor_actions([1])


def test_take_action_accepts_reset_but_not_absent_action7():
    agent = object.__new__(TychoAgent)

    choice, trace_args, msg = agent._prepare_take_action(
        {"action": "RESET"}, actor_actions([GameAction.ACTION1])
    )
    assert msg == ""
    assert choice.action is GameAction.RESET
    assert trace_args == {"action": "RESET"}

    choice, trace_args, msg = agent._prepare_take_action(
        {"action": "ACTION7"}, actor_actions([GameAction.ACTION1])
    )
    assert choice is None
    assert trace_args is None
    assert "ACTION7 is not available" in msg

    choice, trace_args, msg = agent._prepare_take_action(
        {"action": "ACTION7"}, actor_actions([GameAction.ACTION1, GameAction.ACTION7])
    )
    assert msg == ""
    assert choice.action is GameAction.ACTION7
    assert trace_args == {"action": "ACTION7"}


def test_workspace_seed_filters_reset_but_keeps_declared_action7(tmp_path):
    ws = GameWorkspace(
        "action-test",
        root=str(tmp_path),
        render=False,
        available_actions=actor_actions([GameAction.ACTION1, GameAction.ACTION7]),
    )
    text = (ws.dir / "world_model.py").read_text()
    assert '"RESET"' not in text
    assert '"ACTION1"' in text
    assert '"ACTION7"' in text


def test_actor_header_has_reset_but_wm_feedback_actions_do_not():
    old = _set_env()
    try:
        class StubTycho(TychoAgent):
            def _chat(self, call_type, max_tokens=None):
                return {"text": "restart", "tool_calls": [
                    {"id": "a", "name": "take_action", "input": {"action": "RESET"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("action_split", [GameAction.ACTION1, GameAction.ACTION7], ws_root=td)
            choice = agent.choose_action(
                [],
                _frame(),
                actor_actions([GameAction.ACTION1, GameAction.ACTION7]),
            )

        assert choice.action is GameAction.RESET
        assert agent._current_available_actions == ["RESET", "ACTION1", "ACTION7"]
        assert agent._current_game_actions == ["ACTION1", "ACTION7"]
        assert agent.exec.available == ["ACTION1", "ACTION7"]
        header_text = "\n".join(
            part.get("text", "")
            for msg in agent.history
            if isinstance(msg.get("content"), list)
            for part in msg["content"]
            if isinstance(part, dict)
        )
        assert "available actions=['RESET', 'ACTION1', 'ACTION7']" in header_text
    finally:
        _restore_env(old)


def test_tool_cap_fallback_prefers_non_reset_action():
    old = _set_env()
    try:
        class StubTycho(TychoAgent):
            def _chat(self, call_type, max_tokens=None):
                return {"text": "no action yet", "tool_calls": []}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("fallback", [GameAction.ACTION1, GameAction.ACTION6], ws_root=td)
            choice = agent.choose_action(
                [],
                _frame(),
                actor_actions([GameAction.ACTION1, GameAction.ACTION6]),
            )

        assert choice.action is GameAction.ACTION1
        assert choice.reasoning["src"] == "default (tool cap)"
    finally:
        _restore_env(old)


def test_actor_reset_starts_fresh_attempt_without_reset_transition():
    old = _set_env()
    try:
        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.chat_calls = 0

            def _chat(self, call_type, max_tokens=None):
                self.chat_calls += 1
                action = "RESET" if self.chat_calls == 1 else "ACTION1"
                return {"text": action.lower(), "tool_calls": [
                    {"id": f"a{self.chat_calls}", "name": "take_action", "input": {"action": action}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("voluntary_reset", [GameAction.ACTION1], ws_root=td)
            choice = agent.choose_action([], _frame(), actor_actions([GameAction.ACTION1]))
            assert choice.action is GameAction.RESET
            assert agent.turn_in_level == 1

            agent.note_actor_reset()
            assert agent.turn_in_level == 0

            second = agent.choose_action([], _frame(), actor_actions([GameAction.ACTION1]))
            assert second.action is GameAction.ACTION1

            spec = importlib.util.spec_from_file_location("wmlib_under_test", agent.ws.dir / "wmlib.py")
            wmlib = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(wmlib)
            transitions = wmlib.transitions(str(agent.ws.dir))

            assert transitions == []
            assert not (agent.ws.dir / "level_0" / "turn_001.json").exists()
            assert (agent.ws.dir / "level_0" / "turn_000.json").exists()
    finally:
        _restore_env(old)


def test_run_env_routes_actor_reset_through_attempt_boundary():
    class FakeEnv:
        def __init__(self):
            self.actions = []

        def reset(self):
            return SimpleNamespace(
                frame=[[[0]]],
                levels_completed=0,
                state=GameState.NOT_FINISHED,
                available_actions=[1],
            )

        def step(self, action, data=None, reasoning=None):
            self.actions.append(action)
            if action is GameAction.RESET:
                return SimpleNamespace(
                    frame=[[[0]]],
                    levels_completed=0,
                    state=GameState.NOT_FINISHED,
                    available_actions=[1],
                )
            return SimpleNamespace(
                frame=[[[1]]],
                levels_completed=1,
                state=GameState.WIN,
                available_actions=[1],
            )

    class FakeArcade:
        def __init__(self):
            self.env = FakeEnv()

        def make(self, game_id, seed=0, scorecard_id=None):
            return self.env

    class ResetAgent:
        calls = 0
        max_calls = 10
        builder = None

        def __init__(self):
            self.choose_calls = 0
            self.reset_notes = 0
            self.reset_available = None

        def reset(self, game_id, available_actions):
            self.reset_available = list(available_actions)

        def is_done(self, frames, latest_frame):
            return False

        def choose_action(self, frames, latest_frame, available_actions):
            self.choose_calls += 1
            if self.choose_calls == 1:
                assert GameAction.RESET in available_actions
                return ActionChoice(GameAction.RESET)
            assert self.reset_notes == 1
            return ActionChoice(GameAction.ACTION1)

        def note_actor_reset(self):
            self.reset_notes += 1

    arc = FakeArcade()
    agent = ResetAgent()
    rec = run_env(agent, arc, "reset-game", [3], keep_trace=True)

    assert agent.reset_available == [GameAction.ACTION1]
    assert agent.reset_notes == 1
    assert arc.env.actions == [GameAction.RESET, GameAction.ACTION1]
    assert rec.resets == 1
    assert rec.total_actions == 2
    assert rec.reset_actions_counted is True


def test_stop_run_from_choose_action_ends_cleanly_with_the_agents_reason():
    from tycho.harness.harness import StopRun

    class FakeEnv:
        def reset(self):
            return SimpleNamespace(frame=[[[0]]], levels_completed=0, state=GameState.NOT_FINISHED,
                                   available_actions=[1])

        def step(self, action, data=None, reasoning=None):
            return self.reset()

    class FakeArcade:
        def make(self, game_id, seed=0, scorecard_id=None):
            return FakeEnv()

    class StoppingAgent:
        calls = 0
        max_calls = 10
        builder = None
        done_reason = None

        def __init__(self):
            self.choose_calls = 0

        def reset(self, game_id, available_actions):
            pass

        def is_done(self, frames, latest_frame):
            return False

        def choose_action(self, frames, latest_frame, available_actions):
            self.choose_calls += 1
            if self.choose_calls == 2:
                self.done_reason = "deadline"
                raise StopRun("no slot")
            return ActionChoice(GameAction.ACTION1)

    rec = run_env(StoppingAgent(), FakeArcade(), "stop-game", [3], keep_trace=True)

    assert rec.stop_reason == "deadline" and rec.error is None
    assert rec.total_actions_including_unfinished == 1
