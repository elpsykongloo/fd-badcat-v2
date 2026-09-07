"""Production request-capacity and bounded-history contract tests."""
import asyncio
import sys
import threading
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine import ActorEngine, conservative_history_tokens  # noqa: E402
from request_capacity import CONTROL, NORMAL, RequestCapacity  # noqa: E402


class NoVAD:
    def __call__(self, *args, **kwargs):
        return None

    def reset_states(self):
        pass


def engine_with_budget(budget):
    return ActorEngine(
        engine_cfg={"history_token_budget": budget},
        replay_mode="injected", decision_script=lambda *_: {},
        vad_iterator=NoVAD(), llm_fn=lambda _: "", asr_fn=lambda _: "",
        tts_fn=lambda _text, path: path,
        request_capacity=RequestCapacity(),
    )


def history_texts(messages):
    result = []
    for message in messages[1:]:
        content = message["content"]
        result.append(content[0]["text"] if isinstance(content, list) else content)
    return result


def test_history_budget_keeps_newest_complete_pairs_in_original_roles():
    users = ["old-user", "middle-user", "new-user"]
    assistants = ["old-answer", "middle-answer", "new-answer"]
    newest_two = sum(conservative_history_tokens(text) for text in
                     users[1:] + assistants[1:])
    engine = engine_with_budget(newest_two)
    engine.user_history = users
    engine.assistant_history = assistants

    messages = engine.build_messages("system", None, use_history=True, shift_history=False)

    assert history_texts(messages) == [
        "middle-user", "middle-answer", "new-user", "new-answer"]
    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "user", "assistant"]
    assert engine.last_history_window == {
        "available_pairs": 3, "included_pairs": 2,
        "estimated_tokens": newest_two, "budget": newest_two,
    }


def test_history_budget_never_splits_or_skips_the_newest_pair():
    engine = engine_with_budget(50)
    engine.user_history = ["tiny", "新" * 40]
    engine.assistant_history = ["tiny", "答" * 40]
    messages = engine.build_messages("system", None, use_history=False, shift_history=True)
    assert messages == [{"role": "system", "content": "system"}]
    assert engine.last_history_window["available_pairs"] == 2
    assert engine.last_history_window["included_pairs"] == 0


def test_history_disabled_path_remains_history_free():
    engine = engine_with_budget(3000)
    engine.user_history = ["user"]
    engine.assistant_history = ["assistant"]
    messages = engine.build_messages("system", None, use_history=False, shift_history=False)
    assert messages == [{"role": "system", "content": "system"}]


async def wait_until(predicate, timeout=1):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("condition was not met")
        await asyncio.sleep(0.002)


async def test_three_normal_requests_leave_fourth_slot_for_control_and_fifo():
    capacity = RequestCapacity(total_limit=4, normal_limit=3, poll_interval=0.001)
    release = asyncio.Event()
    entered = []

    async def hold(name, request_class):
        async with capacity.slot(request_class):
            entered.append(name)
            await release.wait()

    normals = []
    for index in range(4):
        normals.append(asyncio.create_task(hold(f"normal-{index}", NORMAL)))
        await asyncio.sleep(0.004)
    await wait_until(lambda: capacity.snapshot()["active_normal"] == 3)
    assert entered == ["normal-0", "normal-1", "normal-2"]
    assert capacity.snapshot()["waiting_normal"] == 1

    control = asyncio.create_task(hold("judge", CONTROL))
    await wait_until(lambda: "judge" in entered)
    snap = capacity.snapshot()
    assert snap["active_total"] == snap["peak_total"] == 4
    assert snap["active_normal"] == snap["peak_normal"] == 3
    assert snap["reserved_control"] == 1
    assert "normal-3" not in entered

    release.set()
    await asyncio.gather(*normals, control)
    assert entered[-1] == "normal-3"
    assert capacity.snapshot()["active_total"] == 0


async def test_cancelled_fifo_waiter_does_not_block_following_request():
    capacity = RequestCapacity(total_limit=2, normal_limit=1, poll_interval=0.001)
    release_first = asyncio.Event()
    entered = []

    async def hold(name, release):
        async with capacity.slot(NORMAL):
            entered.append(name)
            await release.wait()

    first = asyncio.create_task(hold("first", release_first))
    await wait_until(lambda: entered == ["first"])
    cancelled = asyncio.create_task(hold("cancelled", asyncio.Event()))
    follower_release = asyncio.Event()
    follower = asyncio.create_task(hold("follower", follower_release))
    await wait_until(lambda: capacity.snapshot()["waiting_normal"] == 2)
    cancelled.cancel()
    await asyncio.gather(cancelled, return_exceptions=True)
    release_first.set()
    await wait_until(lambda: "follower" in entered)
    follower_release.set()
    await asyncio.gather(first, follower)
    assert entered == ["first", "follower"]


async def test_timed_out_caller_does_not_release_still_running_worker_slot():
    capacity = RequestCapacity(total_limit=2, normal_limit=1, poll_interval=0.001)
    engine = engine_with_budget(3000)
    engine.request_capacity = capacity
    release = threading.Event()

    def blocked():
        release.wait(2)
        return "done"

    first = engine._start_capacity_thread_call("response", blocked)
    await wait_until(lambda: capacity.snapshot()["active_normal"] == 1)
    follower = asyncio.create_task(engine._capacity_thread_call("tts", lambda: "next"))
    await wait_until(lambda: capacity.snapshot()["waiting_normal"] == 1)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(first), 0.01)
    assert capacity.snapshot()["active_normal"] == 1
    assert not follower.done()

    control = engine._start_capacity_thread_call("judge", lambda: "switch")
    assert await asyncio.wait_for(asyncio.shield(control), 1) == "switch"
    assert capacity.snapshot()["peak_total"] == 2

    release.set()
    assert await asyncio.wait_for(asyncio.shield(first), 1) == "done"
    assert await asyncio.wait_for(follower, 1) == "next"
    assert capacity.snapshot()["active_total"] == 0


def test_actor_request_classes_and_production_config_are_explicit():
    assert ActorEngine._request_class("judge") == CONTROL
    assert ActorEngine._request_class("interrupt") == CONTROL
    for kind in ("response", "shift", "shift_re", "tts", "tact"):
        assert ActorEngine._request_class(kind) == NORMAL

    production = yaml.safe_load(
        (ROOT / "configs/qwen3_omni_audio_single_gpu.yaml").read_text())
    stages = {stage["stage_id"]: stage for stage in production["stages"]}
    assert stages[0]["max_model_len"] == 4096
    assert {stage["max_num_seqs"] for stage in stages.values()} == {4}

    app = yaml.safe_load((ROOT / "src/config.yaml").read_text())["engine"]
    assert (app["request_total_limit"], app["normal_request_limit"]) == (4, 3)
    assert app["history_token_budget"] == 3000

    launcher = (ROOT / "setup/start_qwen3omni_vllm_omni.sh").read_text()
    assert 'QWEN_SCHEDULING_POLICY:-fcfs' in launcher
    assert '--scheduling-policy "$SCHEDULING_POLICY"' in launcher
