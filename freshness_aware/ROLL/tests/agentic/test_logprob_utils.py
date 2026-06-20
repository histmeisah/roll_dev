from types import SimpleNamespace

import torch

from roll.pipeline.agentic.logprob_utils import (
    attach_replay_slot_indices,
    compute_kl_fresh_priorities,
    preserve_replay_behavior_old_log_probs,
)


def test_preserve_replay_behavior_old_log_probs_keeps_buffer_values():
    behavior_old_log_probs = torch.tensor([[-3.0, -2.0, 0.0]])
    current_actor_log_probs = torch.tensor([[-0.1, -0.2, -0.3]])
    replay_batch = SimpleNamespace(
        batch={
            "attention_mask": torch.tensor([[True, True, True, False]]),
            "old_log_probs": behavior_old_log_probs.clone(),
            "log_probs": current_actor_log_probs,
        }
    )

    preserve_replay_behavior_old_log_probs(replay_batch)

    assert torch.equal(replay_batch.batch["old_log_probs"], behavior_old_log_probs.float())
    assert not torch.equal(replay_batch.batch["old_log_probs"], current_actor_log_probs)


def test_preserve_replay_behavior_old_log_probs_legacy_fallback_is_float():
    replay_batch = SimpleNamespace(
        batch={
            "attention_mask": torch.tensor([[True, True, False, False]]),
        }
    )

    preserve_replay_behavior_old_log_probs(replay_batch)

    assert replay_batch.batch["old_log_probs"].dtype == torch.float32
    assert replay_batch.batch["old_log_probs"].shape == (1, 3)
    assert torch.equal(replay_batch.batch["old_log_probs"], torch.zeros((1, 3)))


def test_attach_replay_slot_indices_expands_group_slots():
    replay_batch = SimpleNamespace(
        batch={
            "input_ids": torch.zeros((5, 4), dtype=torch.long),
        }
    )

    attach_replay_slot_indices(
        replay_batch,
        sampled_indices=[10, 20],
        group_sizes=[2, 3],
    )

    assert torch.equal(
        replay_batch.batch["_replay_slot_indices"],
        torch.tensor([10, 10, 20, 20, 20]),
    )


def test_compute_kl_fresh_priorities_keeps_base_when_log_ratio_zero():
    replay_batch = SimpleNamespace(
        batch={
            "old_log_probs": torch.tensor([[-1.0, -2.0, 0.0]]),
            "response_mask": torch.tensor([[0, 1, 1, 0]], dtype=torch.bool),
            "scores": torch.tensor([[0.0, 0.0, 2.0, 0.0]]),
        }
    )
    current_log_probs = replay_batch.batch["old_log_probs"].clone()

    priorities, stats = compute_kl_fresh_priorities(replay_batch, current_log_probs)

    assert torch.allclose(priorities, torch.tensor([2.000001]), atol=1e-6)
    assert stats["kl_fresh/divergence_mean"] == 0.0
    assert stats["kl_fresh/decay_mean"] == 1.0


def test_compute_kl_fresh_priorities_downweights_offpolicy_sample():
    replay_batch = SimpleNamespace(
        batch={
            "old_log_probs": torch.tensor([[-1.0, -1.0]]),
            "response_mask": torch.tensor([[0, 1, 1]], dtype=torch.bool),
            "scores": torch.tensor([[0.0, 2.0, 0.0]]),
        }
    )
    current_log_probs = torch.tensor([[0.0, 0.0]])

    priorities, stats = compute_kl_fresh_priorities(
        replay_batch,
        current_log_probs,
        eta=1.0,
        log_ratio_clip=10.0,
    )

    assert priorities.item() < 2.0
    assert stats["kl_fresh/divergence_mean"] > 0.0
    assert stats["kl_fresh/decay_mean"] < 1.0
