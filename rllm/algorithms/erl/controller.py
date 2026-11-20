"""Backend-independent implementation of the released ERL collection loop."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field

from rllm.algorithms.erl.replay import make_internalization_copy
from rllm.algorithms.erl.upstream import build_updater_state


class UpdaterPromptOverflow(Exception):
    """The updater input is longer than its prompt budget."""


@dataclass
class ERLCollection:
    first: list[dict] = field(default_factory=list)
    additional: list[tuple[int, str, dict]] = field(default_factory=list)
    generated: list[dict] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)


class ERLController:
    """One remembered prompt per FIFO workflow slot, as in AgentWorkflowEngine.

    Slots persist across tasks and epochs. First attempts always use the plain
    agent; memory only conditions the updater. A revised prompt becomes memory
    only when its fresh retry succeeds. There is no candidate search or ACE
    controller. Callbacks bind the unchanged benchmark interaction protocol.
    """

    def __init__(self, pool_size: int):
        if type(pool_size) is not int or pool_size < 1:
            raise ValueError("ERL pool_size must be a positive integer")
        self.memories: list[str | None] = [None] * pool_size
        self.slot_order = list(range(pool_size))

    async def collect(self, count: int, *, attempt, update, succeeded) -> ERLCollection:
        slots = asyncio.Queue()
        for slot in self.slot_order:
            slots.put_nowait(slot)
        result = ERLCollection(first=[None] * count)

        async def episode(index):
            slot = await slots.get()
            try:
                first = await attempt(index, None)
                result.first[index] = first
                result.generated.append(first)
                if succeeded(first):
                    return
                initial_prompt = str(first["chat_completions"][0]["content"])
                current = self.memories[slot] or initial_prompt
                state = build_updater_state(current, first, correct=False)
                try:
                    prompt, updater = await update(index, state, current)
                except UpdaterPromptOverflow:
                    # The release drops an episode whose generation prompt is
                    # too long (an empty episode leaves the batch); memory is
                    # unchanged and nothing is summarized or truncated.
                    result.dropped.append(index)
                    return
                result.generated.append(updater)
                second = await attempt(index, prompt)
                result.generated.append(second)
                # Reference default episode batch_size=1: mean second reward
                # is exactly this retry's raw reward. No delta or success gate.
                updater["trajectory_reward"] = float(second["trajectory_reward"])
                updater["step_rewards"] = [updater["trajectory_reward"]]
                internalized = make_internalization_copy(second, first["prompt_tokens"])
                internalized["chat_completions"][0]["content"] = initial_prompt
                result.additional.extend(
                    [
                        (index, "updater", updater),
                        (index, "second_raw", second),
                        (index, "second_distilled", internalized),
                    ]
                )
                if succeeded(second):
                    self.memories[slot] = prompt
            finally:
                slots.put_nowait(slot)

        tasks = [asyncio.create_task(episode(index)) for index in range(count)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            self.slot_order = [slots.get_nowait() for _ in range(slots.qsize())]
        # Runtime completion order must not change record order or group IDs.
        order = {"updater": 0, "second_raw": 1, "second_distilled": 2}
        result.additional.sort(key=lambda row: (row[0], order[row[1]]))
        return result

    def state_dict(self):
        return {"memories": copy.deepcopy(self.memories), "slot_order": list(self.slot_order)}

    def load_state_dict(self, document):
        if not isinstance(document, dict) or set(document) != {"memories", "slot_order"}:
            raise ValueError("Invalid ERL memory state fields")
        memories, order = document["memories"], document["slot_order"]
        if not isinstance(memories, list) or len(memories) != len(self.memories):
            raise ValueError("ERL checkpoint memory pool size is incompatible")
        if not all(value is None or isinstance(value, str) for value in memories):
            raise ValueError("Invalid ERL remembered prompt")
        if not isinstance(order, list) or any(type(value) is not int for value in order) or sorted(order) != list(range(len(memories))):
            raise ValueError("Invalid ERL available-slot order")
        self.memories = list(memories)
        self.slot_order = list(order)
