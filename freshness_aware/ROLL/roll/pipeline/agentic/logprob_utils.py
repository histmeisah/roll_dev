from typing import Any, Dict, List, Optional, Tuple

import torch


def preserve_replay_behavior_old_log_probs(replay_batch: Any, logger: Optional[Any] = None) -> Any:
    """Keep replay old_log_probs tied to the behavior policy stored in the buffer."""
    if "old_log_probs" in replay_batch.batch:
        replay_batch.batch["old_log_probs"] = replay_batch.batch["old_log_probs"].float()
    else:
        # Legacy buffer entries may lack stored behavior logprobs. Keep this as a
        # last-resort fallback only; valid FreshPER replay should not reach it.
        replay_batch.batch["old_log_probs"] = torch.zeros_like(
            replay_batch.batch["attention_mask"][:, 1:], dtype=torch.float32
        )
        if logger is not None:
            logger.warning("Replay batch missing behavior old_log_probs; falling back to zeros.")
    return replay_batch


def attach_replay_slot_indices(
    replay_batch: Any,
    sampled_indices: List[int],
    group_sizes: Optional[List[int]] = None,
    key: str = "_replay_slot_indices",
) -> Any:
    """Attach buffer slot ids to each flat replay row so reordering stays traceable."""
    if not sampled_indices:
        return replay_batch

    if group_sizes is not None and len(group_sizes) == len(sampled_indices):
        slot_ids = []
        for slot_idx, group_size in zip(sampled_indices, group_sizes):
            slot_ids.extend([int(slot_idx)] * int(group_size))
    else:
        slot_ids = [int(idx) for idx in sampled_indices]

    batch_size = replay_batch.batch["input_ids"].shape[0]
    if len(slot_ids) != batch_size:
        raise ValueError(
            f"replay slot id count {len(slot_ids)} does not match batch size {batch_size}"
        )

    device = replay_batch.batch["input_ids"].device
    replay_batch.batch[key] = torch.tensor(slot_ids, dtype=torch.long, device=device)
    return replay_batch


def compute_kl_fresh_priorities(
    replay_batch: Any,
    current_log_probs: torch.Tensor,
    eta: float = 1.0,
    log_ratio_clip: float = 10.0,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute reward-base priorities decayed by an observed reverse-KL estimate.

    The divergence term is
        E_mu[r log r - r + 1], r = pi_theta(a|s) / pi_mu(a|s),
    estimated on replayed response tokens. This is a behavior-sample estimator of
    D_KL(pi_theta || pi_mu) with the zero-mean control variate (r - 1).
    """
    if "old_log_probs" not in replay_batch.batch:
        raise KeyError("KL-FreshPER requires replay old_log_probs from the behavior policy")
    if "response_mask" not in replay_batch.batch:
        raise KeyError("KL-FreshPER requires response_mask")
    if "scores" not in replay_batch.batch:
        raise KeyError("KL-FreshPER requires scores for reward base priority")

    behavior_log_probs = replay_batch.batch["old_log_probs"].to(
        device=current_log_probs.device, dtype=torch.float32
    )
    current_log_probs = current_log_probs.float()
    response_mask = replay_batch.batch["response_mask"][:, 1:].to(
        device=current_log_probs.device, dtype=torch.bool
    )

    min_seq_len = min(
        current_log_probs.shape[1],
        behavior_log_probs.shape[1],
        response_mask.shape[1],
    )
    current_log_probs = current_log_probs[:, :min_seq_len]
    behavior_log_probs = behavior_log_probs[:, :min_seq_len]
    response_mask = response_mask[:, :min_seq_len]

    log_ratio = current_log_probs - behavior_log_probs
    if log_ratio_clip is not None and log_ratio_clip > 0:
        log_ratio = log_ratio.clamp(min=-float(log_ratio_clip), max=float(log_ratio_clip))
    ratio = log_ratio.exp()
    reverse_kl_terms = ratio * log_ratio - ratio + 1.0

    mask_f = response_mask.float()
    token_counts = mask_f.sum(dim=1).clamp(min=1.0)
    divergence = (reverse_kl_terms * mask_f).sum(dim=1) / token_counts
    divergence = divergence.clamp(min=0.0)

    rewards = replay_batch.batch["scores"].to(
        device=current_log_probs.device, dtype=torch.float32
    ).sum(dim=1)
    base_priority = rewards.abs().add(float(epsilon))
    decay = torch.exp(-float(eta) * divergence)
    priorities = (base_priority * decay).clamp(min=float(epsilon))

    valid_log_ratio = log_ratio[response_mask]
    stats = {
        "kl_fresh/divergence_mean": divergence.mean().detach().item(),
        "kl_fresh/divergence_max": divergence.max().detach().item(),
        "kl_fresh/decay_mean": decay.mean().detach().item(),
        "kl_fresh/priority_mean": priorities.mean().detach().item(),
    }
    if valid_log_ratio.numel() > 0:
        stats.update({
            "kl_fresh/log_ratio_mean": valid_log_ratio.mean().detach().item(),
            "kl_fresh/log_ratio_abs_mean": valid_log_ratio.abs().mean().detach().item(),
        })

    return priorities.detach().cpu(), stats
