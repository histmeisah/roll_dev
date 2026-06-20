import numpy as np
import pytest
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.replay_buffer.group_buffer import GroupReplayBuffer
from roll.pipeline.agentic.replay_buffer.priority_functions import reward_priority


def _make_group_batch(rewards):
    batch_size = len(rewards)
    seq_len = 4
    tensors = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * batch_size, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * batch_size, dtype=torch.bool),
        "position_ids": torch.arange(seq_len).repeat(batch_size, 1),
        "response_mask": torch.tensor([[0, 1, 1, 0]] * batch_size, dtype=torch.bool),
        "prompt_mask": torch.tensor([[1, 0, 0, 0]] * batch_size, dtype=torch.bool),
        "scores": torch.zeros((batch_size, seq_len), dtype=torch.float32),
        "infer_logprobs": torch.zeros((batch_size, seq_len - 1), dtype=torch.float32),
    }
    for i, reward in enumerate(rewards):
        tensors["scores"][i, 2] = reward

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "traj_group_id": np.array(["group-0"] * batch_size, dtype=object),
        },
    )


def test_group_priority_uses_mean_over_group_entries():
    buffer = GroupReplayBuffer(
        capacity=4,
        batch_size=1,
        priority_fn=reward_priority,
        priority_exponent=0.6,
    )

    buffer.push_from_dataproto(_make_group_batch([0.0, 1.0]), global_step=7)

    stored_group = buffer.groups[0]
    assert stored_group is not None
    assert stored_group.group_size == 2
    assert stored_group.priority == pytest.approx((1e-6 + 1.0 + 1e-6) / 2)
