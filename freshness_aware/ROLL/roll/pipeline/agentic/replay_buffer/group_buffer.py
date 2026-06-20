"""
Group-Level Replay Buffer for ROLL Framework

Stores and samples trajectory groups as atomic units, ensuring GRPO compatibility.
Each group contains K trajectories sharing the same traj_group_id (same prompt/state),
so group-level reward normalization works correctly on replay data.

When group_size=1, degrades to standard per-trajectory replay buffer behavior.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import random
import numpy as np
import torch
from transformers import PreTrainedTokenizer
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger
from .base_buffer import BaseReplayBuffer
from .segment_tree import SumSegmentTree, MinSegmentTree, next_power_of_2
from .trajectory_buffer import TrajectoryEntry

logger = get_logger()


@dataclass
class TrajectoryGroup:
    """
    A group of K trajectories sharing the same traj_group_id.
    This is the atomic storage/sampling unit for GRPO-compatible replay.
    """
    traj_group_id: str
    trajectories: List[TrajectoryEntry]
    group_size: int  # K, the number of trajectories in this group

    # Group-level metadata
    tag: str = ""
    mean_episode_score: float = 0.0
    stored_at_step: int = 0

    # Priority-related
    priority: float = 1.0
    sample_count: int = 0
    global_step: int = 0

    def __post_init__(self):
        self.group_size = len(self.trajectories)
        if self.trajectories:
            scores = [
                float(t.episode_scores) if isinstance(t.episode_scores, (int, float, np.floating))
                else float(t.episode_scores[-1]) if hasattr(t.episode_scores, '__len__') and len(t.episode_scores) > 0
                else 0.0
                for t in self.trajectories
            ]
            self.mean_episode_score = float(np.mean(scores))
            self.tag = self.trajectories[0].tag


class GroupReplayBuffer(BaseReplayBuffer):
    """
    Replay buffer that stores and samples trajectory groups as atomic units.

    Designed for GRPO compatibility: each group contains K trajectories
    from the same prompt/state, so group-level reward normalization
    works correctly on replay data.

    When group_size=1, this degrades to standard per-trajectory replay.

    Storage unit: TrajectoryGroup (K trajectories)
    Sampling unit: N groups → N×K trajectories as DataProto
    Priority unit: group-level (mean reward, mean advantage, etc.)
    """

    def __init__(
        self,
        capacity: int = 10000,
        batch_size: int = 32,
        seed: int = 42,
        priority_fn: callable = None,
        priority_exponent: float = 0.6,
        age_decay: float = 1000.0,
        enable_age_decay: bool = False,
        eviction_strategy: str = "fifo",
    ):
        """
        Args:
            capacity: Maximum number of groups (not individual trajectories)
            batch_size: Default number of groups to sample (actual trajectory count = batch_size × K)
            seed: Random seed
            priority_fn: Priority function for group-level priority
            priority_exponent: Alpha in PER (0=uniform, 1=full prioritization)
            age_decay: Age decay constant for freshness weighting
            enable_age_decay: Enable age-based freshness weighting
            eviction_strategy: "fifo" or "smart"
        """
        super().__init__(capacity, batch_size, seed)
        self.groups: List[Optional[TrajectoryGroup]] = [None] * capacity
        self.valid_mask: List[bool] = [False] * capacity
        self.num_valid: int = 0
        self.rng = random.Random(seed)
        self.eviction_strategy = eviction_strategy.lower()

        # Priority
        from .priority_functions import uniform_priority
        self.priority_fn = priority_fn or uniform_priority
        self.priority_exponent = priority_exponent

        # Segment tree for O(log n) prioritized sampling
        self._tree_capacity = next_power_of_2(capacity)
        self._it_sum = SumSegmentTree(self._tree_capacity)
        self._it_min = MinSegmentTree(self._tree_capacity)
        self._max_priority = 1.0

        # Age decay
        self.enable_age_decay = enable_age_decay
        self.age_decay = age_decay
        self.current_global_step = 0

        logger.info(
            f"GroupReplayBuffer: capacity={capacity} groups, "
            f"priority_fn={self.priority_fn.__name__}, "
            f"priority_exponent={priority_exponent}, "
            f"eviction={eviction_strategy}"
        )

    @property
    def buffer_type(self) -> str:
        return "group"

    # ─── Push ────────────────────────────────────────────────────────────

    def push_from_dataproto(self, batch: DataProto, global_step: int) -> None:
        """
        Store trajectory data grouped by traj_group_id.

        Collects trajectories sharing the same traj_group_id into groups,
        then stores each complete group as one atomic unit.
        """
        self.current_global_step = global_step
        batch_size = batch.batch["input_ids"].shape[0]

        # Step 1: group trajectories by traj_group_id
        group_map: Dict[str, List[int]] = defaultdict(list)
        for i in range(batch_size):
            tgid = str(batch.non_tensor_batch["traj_group_id"][i])
            group_map[tgid].append(i)

        # Step 2: build TrajectoryGroup for each group and store
        num_stored = 0
        for tgid, indices in group_map.items():
            entries = []
            for i in indices:
                entry = self._extract_trajectory_entry(batch, i, global_step)
                entries.append(entry)

            group = TrajectoryGroup(
                traj_group_id=tgid,
                trajectories=entries,
                group_size=len(entries),
                stored_at_step=global_step,
                global_step=global_step,
            )

            # Compute group-level priority as the mean over all trajectories in the
            # GRPO group. Using only the first completion can misprioritize prompts
            # where a later sampled completion is the informative/rewarding one.
            try:
                if self.priority_fn.__name__ == "grpo_signal_priority":
                    group.priority = self._compute_group_signal_priority(entries)
                else:
                    priorities = [
                        float(self.priority_fn(entry, global_step, age_decay=self.age_decay))
                        for entry in entries
                    ]
                    group.priority = float(np.mean(priorities)) if priorities else 1.0
            except Exception as e:
                logger.warning(f"Failed to calculate group priority, using 1.0: {e}")
                group.priority = 1.0

            self._store_group(group)
            num_stored += 1

        logger.debug(
            f"Pushed {num_stored} groups ({batch_size} trajectories) at step {global_step}. "
            f"Buffer: {self.num_valid}/{self.capacity} groups"
        )

    def _extract_trajectory_entry(self, batch: DataProto, idx: int, global_step: int) -> TrajectoryEntry:
        """Extract a single TrajectoryEntry from a DataProto batch at index idx."""
        input_ids = batch.batch["input_ids"][idx].cpu().numpy()
        attention_mask = batch.batch["attention_mask"][idx].cpu().numpy()
        position_ids = batch.batch["position_ids"][idx].cpu().numpy()
        response_mask = batch.batch["response_mask"][idx].cpu().numpy()
        prompt_mask = batch.batch["prompt_mask"][idx].cpu().numpy()
        scores = batch.batch["scores"][idx].cpu().numpy()

        # penalty was removed from rollout batch in upstream; default to 0.0
        if "penalty" in batch.batch:
            penalty = float(batch.batch["penalty"][idx].cpu().item())
        else:
            penalty = 0.0

        # upstream renamed behavior_log_probs → infer_logprobs
        target_len = max(int(input_ids.shape[0]) - 1, 0)
        if "behavior_log_probs" in batch.batch:
            behavior_log_probs = batch.batch["behavior_log_probs"][idx].cpu().numpy()
        elif "infer_logprobs" in batch.batch:
            behavior_log_probs = batch.batch["infer_logprobs"][idx].cpu().numpy()
        else:
            behavior_log_probs = np.zeros((target_len,), dtype=np.float32)

        # Upstream removed `messages_list` and `frames` from text TrajEnvManager.
        # VLM env manager still emits `messages_list`. Make them optional here.
        def _nt(key, default):
            arr = batch.non_tensor_batch.get(key, None)
            if arr is None:
                return default
            return arr[idx]

        return TrajectoryEntry(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_mask=response_mask,
            prompt_mask=prompt_mask,
            scores=scores,
            penalty=penalty,
            behavior_log_probs=behavior_log_probs,
            env_id=_nt("env_ids", ""),
            group_id=_nt("group_ids", ""),
            messages_list=_nt("messages_list", []),
            tag=_nt("tags", ""),
            frames=_nt("frames", []),
            step_scores=_nt("step_scores", []),
            episode_scores=_nt("episode_scores", 0.0),
            traj_group_id=_nt("traj_group_id", ""),
            traj_id=_nt("traj_id", ""),
            model_answer=_nt("model_answer", None),
            gold_answer=_nt("gold_answer", None),
            answer_source=_nt("answer_source", None),
            unboxed_answer=_nt("unboxed_answer", None),
            stored_at_step=global_step,
            episode_length=int(attention_mask.sum()),
            global_step=global_step,
            # VLM: preserve per-sample multimodal processor output dict
            # (pixel_values / image_grid_thw / ...). None for text-only rollouts.
            multi_modal_inputs=_nt("multi_modal_inputs", None),
        )

    def _store_group(self, group: TrajectoryGroup) -> None:
        """Store a group into the buffer, evicting if full."""
        if self.num_valid >= self.capacity:
            slot_idx = self._evict_and_get_slot()
        else:
            slot_idx = self._find_empty_slot()
            self.num_valid += 1

        self.groups[slot_idx] = group
        self.valid_mask[slot_idx] = True

        # New groups get max_priority to ensure they're sampled at least once
        priority_alpha = self._max_priority ** self.priority_exponent
        self._it_sum[slot_idx] = priority_alpha
        self._it_min[slot_idx] = priority_alpha
        self.total_stored += 1

    def _find_empty_slot(self) -> int:
        for i in range(self.capacity):
            if not self.valid_mask[i]:
                return i
        raise RuntimeError("No empty slot found but num_valid < capacity")

    def _evict_and_get_slot(self) -> int:
        """FIFO eviction: evict the group with the smallest global_step."""
        oldest_idx = -1
        oldest_step = float('inf')
        for i in range(self.capacity):
            if self.valid_mask[i] and self.groups[i].global_step < oldest_step:
                oldest_step = self.groups[i].global_step
                oldest_idx = i
        if oldest_idx == -1:
            oldest_idx = 0
        # Clear the slot; num_valid stays constant because _store_group
        # will immediately refill this slot (atomic evict-then-store).
        self.valid_mask[oldest_idx] = False
        self.groups[oldest_idx] = None
        self._it_sum[oldest_idx] = 0.0
        self._it_min[oldest_idx] = float('inf')
        return oldest_idx

    # ─── Sample ──────────────────────────────────────────────────────────

    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        required = batch_size or self.batch_size
        return self.num_valid >= required

    def sample_for_training(
        self,
        batch_size: Optional[int] = None,
        device: str = 'cpu',
        tokenizer: Optional[PreTrainedTokenizer] = None,
        sequence_length: int = 4096,
        sampling_mode: str = "trajectory",
        steps_per_episode: int = 1,
        sample_method: str = "uniform",
        candidates_per_group: int = 1,
        group_sampling: str = "uniform",
        compute_importance_weights: bool = False,
        importance_weight_beta: float = 0.4,
    ) -> Optional[Tuple[DataProto, List[int]]]:
        """
        Sample N complete groups and unpack into N×K trajectories.

        Args:
            batch_size: Number of GROUPS to sample (not trajectories).
                        Actual trajectory count = batch_size × group_size.
            sample_method: "uniform", "lifo", "fifo", or priority-based
            Others: see base class

        Returns:
            (DataProto, sampled_slot_indices)
            DataProto contains N×K trajectories with group structure preserved.
        """
        num_groups = batch_size or self.batch_size
        if not self.can_sample(num_groups):
            return None, []

        # Collect valid groups
        valid_groups: List[Tuple[int, TrajectoryGroup]] = []
        for i in range(self.capacity):
            if self.valid_mask[i]:
                valid_groups.append((i, self.groups[i]))

        # Sort by global_step for deterministic strategies
        valid_groups.sort(key=lambda x: x[1].global_step)

        # Select groups
        sampled_slots: List[int] = []
        sampled_groups: List[TrajectoryGroup] = []

        priority_fn_name = self.priority_fn.__name__

        if priority_fn_name == "lifo_priority":
            # Newest N groups
            selected = valid_groups[-num_groups:]
            sampled_slots = [s for s, _ in selected]
            sampled_groups = [g for _, g in selected]
        elif priority_fn_name == "fifo_priority":
            # Oldest N groups
            selected = valid_groups[:num_groups]
            sampled_slots = [s for s, _ in selected]
            sampled_groups = [g for _, g in selected]
        elif priority_fn_name == "uniform_priority":
            selected = self.rng.sample(valid_groups, num_groups)
            sampled_slots = [s for s, _ in selected]
            sampled_groups = [g for _, g in selected]
        else:
            # PER: weighted sampling via segment tree
            slot_indices = self._sample_proportional_slots(num_groups)
            sampled_slots = slot_indices
            sampled_groups = [self.groups[i] for i in slot_indices]
            for idx in slot_indices:
                self.groups[idx].sample_count += 1

        # Unpack groups into flat trajectory list, preserving group order
        all_trajectories: List[TrajectoryEntry] = []
        for group in sampled_groups:
            all_trajectories.extend(group.trajectories)

        total_trajs = len(all_trajectories)
        if total_trajs == 0:
            return None, []

        # Build DataProto from flat trajectory list
        max_seq_len = sequence_length
        target_device = torch.device(device if device else 'cpu')

        batch_input_ids = torch.zeros((total_trajs, max_seq_len), dtype=torch.long, device=target_device)
        batch_attention_mask = torch.zeros((total_trajs, max_seq_len), dtype=torch.bool, device=target_device)

        # Detect VLM multi-dimensional position_ids
        first_pos = all_trajectories[0].position_ids
        if first_pos.ndim > 1:
            pos_dims = first_pos.shape[0]
            batch_position_ids = torch.zeros((total_trajs, pos_dims, max_seq_len), dtype=torch.long, device=target_device)
        else:
            batch_position_ids = torch.zeros((total_trajs, max_seq_len), dtype=torch.long, device=target_device)

        batch_response_mask = torch.zeros((total_trajs, max_seq_len), dtype=torch.bool, device=target_device)
        batch_prompt_mask = torch.zeros((total_trajs, max_seq_len), dtype=torch.bool, device=target_device)
        batch_scores = torch.zeros((total_trajs, max_seq_len), dtype=torch.float32, device=target_device)
        batch_penalties = torch.zeros((total_trajs,), dtype=torch.float32, device=target_device)
        batch_old_log_probs = torch.zeros((total_trajs, max_seq_len - 1), dtype=torch.float32, device=target_device)

        # Non-tensor data
        env_ids, group_ids, messages_lists, tags = [], [], [], []
        frames_lists, step_scores_lists, episode_scores_lists = [], [], []
        traj_group_ids, traj_ids = [], []
        multi_modal_inputs_list = []
        model_answers, gold_answers, answer_sources, unboxed_answers = [], [], [], []

        for i, traj in enumerate(all_trajectories):
            seq_len = min(len(traj.input_ids), max_seq_len)

            batch_input_ids[i] = pad_to_length(torch.from_numpy(traj.input_ids), max_seq_len, 0)
            batch_attention_mask[i] = pad_to_length(torch.from_numpy(traj.attention_mask), max_seq_len, False)

            pos_tensor = torch.from_numpy(traj.position_ids)
            if pos_tensor.ndim > 1:
                for d in range(pos_tensor.shape[0]):
                    batch_position_ids[i, d] = pad_to_length(pos_tensor[d], max_seq_len, 0)
            else:
                batch_position_ids[i] = pad_to_length(pos_tensor, max_seq_len, 0)

            batch_response_mask[i] = pad_to_length(torch.from_numpy(traj.response_mask), max_seq_len, False)
            batch_prompt_mask[i] = pad_to_length(torch.from_numpy(traj.prompt_mask), max_seq_len, False)
            batch_scores[i] = pad_to_length(torch.from_numpy(traj.scores), max_seq_len, 0.0)
            batch_penalties[i] = traj.penalty
            batch_old_log_probs[i] = pad_to_length(
                torch.from_numpy(traj.behavior_log_probs), max_seq_len - 1, 0.0
            )

            env_ids.append(traj.env_id)
            group_ids.append(traj.group_id)
            messages_lists.append(traj.messages_list)
            tags.append(traj.tag)
            frames_lists.append(traj.frames)
            step_scores_lists.append(traj.step_scores)
            episode_scores_lists.append(traj.episode_scores)
            traj_group_ids.append(traj.traj_group_id)
            traj_ids.append(traj.traj_id)
            multi_modal_inputs_list.append(traj.multi_modal_inputs)
            model_answers.append(traj.model_answer)
            gold_answers.append(traj.gold_answer)
            answer_sources.append(traj.answer_source)
            unboxed_answers.append(traj.unboxed_answer)

        dataproto = DataProto()
        dataproto.batch = TensorDict({
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "position_ids": batch_position_ids,
            "response_mask": batch_response_mask,
            "prompt_mask": batch_prompt_mask,
            "scores": batch_scores,
            "penalty": batch_penalties,
            "old_log_probs": batch_old_log_probs,
        }, batch_size=[total_trajs])

        dataproto.non_tensor_batch = {
            "env_ids": np.array(env_ids, dtype=object),
            "group_ids": np.array(group_ids, dtype=object),
            "messages_list": np.array(messages_lists, dtype=object),
            "tags": np.array(tags, dtype=object),
            "frames": np.array(frames_lists, dtype=object),
            "step_scores": np.array(step_scores_lists, dtype=object),
            "episode_scores": np.array(episode_scores_lists, dtype=object),
            "traj_group_id": np.array(traj_group_ids, dtype=object),
            "traj_id": np.array(traj_ids, dtype=object),
        }
        if any(x is not None for x in model_answers):
            dataproto.non_tensor_batch["model_answer"] = np.array(model_answers, dtype=object)
        if any(x is not None for x in gold_answers):
            dataproto.non_tensor_batch["gold_answer"] = np.array(gold_answers, dtype=object)
        if any(x is not None for x in answer_sources):
            dataproto.non_tensor_batch["answer_source"] = np.array(answer_sources, dtype=object)
        if any(x is not None for x in unboxed_answers):
            dataproto.non_tensor_batch["unboxed_answer"] = np.array(unboxed_answers, dtype=object)

        # Only attach multi_modal_inputs when at least one trajectory had it at push time.
        # Downstream strategies gate on `if "multi_modal_inputs" in data.non_tensor_batch`,
        # so omitting the key entirely preserves text-only behavior.
        if any(x is not None for x in multi_modal_inputs_list):
            mm_arr = np.empty(total_trajs, dtype=object)
            mm_arr[:] = multi_modal_inputs_list
            dataproto.non_tensor_batch["multi_modal_inputs"] = mm_arr

        dataproto.meta_info = {
            "from_replay_buffer": True,
            "buffer_type": "group",
            "num_groups": num_groups,
            "group_sizes": [g.group_size for g in sampled_groups],
            "total_trajectories": total_trajs,
            "buffer_utilization": self.num_valid / self.capacity,
            "sampled_indices": sampled_slots,
        }

        # PER importance weights (group-level, broadcast to all trajectories in group)
        if compute_importance_weights and priority_fn_name not in [
            "lifo_priority", "fifo_priority", "uniform_priority"
        ]:
            group_weights = self.compute_importance_weights(sampled_slots, beta=importance_weight_beta)
            # Broadcast group weight to each trajectory in the group
            traj_weights = []
            for gi, group in enumerate(sampled_groups):
                traj_weights.extend([group_weights[gi]] * group.group_size)
            dataproto.batch["importance_weights"] = torch.tensor(
                traj_weights, dtype=torch.float32, device=target_device
            )

        logger.debug(
            f"Sampled {num_groups} groups → {total_trajs} trajectories "
            f"(group_sizes={[g.group_size for g in sampled_groups]})"
        )
        return dataproto, sampled_slots

    # ─── PER helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_group_signal_priority(entries: List[TrajectoryEntry], floor: float = 0.05) -> float:
        """Priority for GRPO groups: reward dispersion implies non-zero gradients."""
        rewards = []
        for entry in entries:
            mask = entry.attention_mask.astype(bool)
            valid_scores = entry.scores[mask]
            rewards.append(float(valid_scores[-1]) if len(valid_scores) > 0 else 0.0)
        if len(rewards) <= 1:
            return float(abs(rewards[0]) + floor) if rewards else 1.0
        return float(np.std(np.asarray(rewards, dtype=np.float32)) + floor)

    def _sample_proportional_slots(self, num_groups: int) -> List[int]:
        """Sample slot indices proportional to priority without replacement."""
        indices = []
        valid_slots = [j for j in range(self.capacity) if self.valid_mask[j]]
        if num_groups >= len(valid_slots):
            return valid_slots

        available = valid_slots
        weights = [max(float(self._it_sum[j]), 0.0) for j in available]
        for _ in range(num_groups):
            total = sum(weights)
            if total <= 0:
                pick_pos = self.rng.randrange(len(available))
            else:
                threshold = self.rng.random() * total
                cumulative = 0.0
                pick_pos = len(available) - 1
                for pos, weight in enumerate(weights):
                    cumulative += weight
                    if cumulative >= threshold:
                        pick_pos = pos
                        break
            indices.append(available.pop(pick_pos))
            weights.pop(pick_pos)
        return indices

    def compute_importance_weights(self, indices: List[int], beta: float = 0.4) -> np.ndarray:
        """Compute PER importance sampling weights for given slot indices."""
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum(0, self._tree_capacity - 1)
        max_weight = (p_min * self.num_valid) ** (-beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum(0, self._tree_capacity - 1)
            weight = (p_sample * self.num_valid) ** (-beta)
            weights.append(weight / max_weight)

        return np.array(weights, dtype=np.float32)

    def update_priorities(
        self,
        indices: List[int],
        priorities: np.ndarray,
        current_global_step: Optional[int] = None,
    ) -> None:
        """
        Update group-level priorities after training (standard PER).

        When enable_age_decay=True, writes `priority * exp(-age/age_decay)` raised to
        priority_exponent into the segment tree (consistent with TrajectoryReplayBuffer),
        so that stale groups get deprioritized over time. `groups[idx].priority` still
        stores the raw (non-decayed) priority so refresh_all_age_decay can recompute
        the freshness factor at the current global_step.
        """
        global_step = (
            current_global_step if current_global_step is not None else self.current_global_step
        )
        for idx, priority in zip(indices, priorities):
            if not (0 <= idx < self.capacity) or not self.valid_mask[idx]:
                continue
            priority = max(float(priority), 1e-6)
            self.groups[idx].priority = priority
            self._max_priority = max(self._max_priority, priority)

            if self.enable_age_decay:
                age = max(0, global_step - self.groups[idx].global_step)
                freshness_weight = np.exp(-age / self.age_decay) if self.age_decay > 0 else 1.0
                effective_priority = max(priority * freshness_weight, 1e-8)
            else:
                effective_priority = priority

            priority_alpha = effective_priority ** self.priority_exponent
            self._it_sum[idx] = priority_alpha
            self._it_min[idx] = priority_alpha

    def refresh_all_age_decay(self, current_global_step: int) -> int:
        """
        Refresh age decay for ALL groups in the buffer, rewriting segment-tree entries.

        Mirrors TrajectoryReplayBuffer.refresh_all_age_decay at the group granularity:
        without this call, only groups that happen to be sampled get their age-decayed
        priority updated (via update_priorities), leaving unsampled old groups with
        stale (overly high) priorities. Pipeline should call this periodically; it is
        designed to be safe to run asynchronously off the GPU critical path.

        Time complexity: O(capacity).

        Returns:
            Number of groups whose segment-tree entry was refreshed. Returns 0 when
            age decay is disabled (no-op).
        """
        if not self.enable_age_decay:
            return 0

        self.current_global_step = current_global_step
        refreshed = 0
        max_age = 0
        total_age = 0

        for idx in range(self.capacity):
            if not self.valid_mask[idx]:
                continue
            group = self.groups[idx]
            if group is None:
                continue

            age = max(0, current_global_step - group.global_step)
            freshness_weight = np.exp(-age / self.age_decay) if self.age_decay > 0 else 1.0
            effective_priority = max(group.priority * freshness_weight, 1e-8)
            priority_alpha = effective_priority ** self.priority_exponent
            self._it_sum[idx] = priority_alpha
            self._it_min[idx] = priority_alpha

            refreshed += 1
            max_age = max(max_age, age)
            total_age += age

        if refreshed > 0:
            avg_age = total_age / refreshed
            logger.debug(
                f"[AGE_DECAY][group] Refreshed {refreshed} groups at step {current_global_step}. "
                f"avg_age={avg_age:.1f}, max_age={max_age}, age_decay={self.age_decay}"
            )
        return refreshed

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Group-level stats. Parallels TrajectoryReplayBuffer.get_stats so the pipeline-side
        `replay/priority/*`, `replay/age/*`, `replay/freshness/*` monitors work identically.
        All priority/age/freshness/sample_count aggregates are per-group.
        """
        stats = {
            "buffer_type": "group",
            "capacity": self.capacity,
            "num_groups": self.num_valid,
            "current_size": self.num_valid,  # alias to match Trajectory/Step buffer naming
            "total_stored_groups": self.total_stored,
            "total_stored": self.total_stored,  # alias
            "utilization": self.num_valid / self.capacity if self.capacity > 0 else 0.0,
            "total_evicted": max(0, self.total_stored - self.num_valid),
            "eviction_rate": max(0, self.total_stored - self.num_valid) / max(1, self.total_stored),
        }
        if self.num_valid == 0:
            return stats

        valid_groups = [self.groups[i] for i in range(self.capacity) if self.valid_mask[i]]

        # Group-size and episode-score summaries (pre-existing)
        group_sizes = np.array([g.group_size for g in valid_groups])
        scores = np.array([g.mean_episode_score for g in valid_groups])
        stats["total_trajectories"] = int(group_sizes.sum())
        stats["avg_group_size"] = float(group_sizes.mean())
        stats["avg_episode_score"] = float(scores.mean())

        # Priority statistics from segment tree (only valid slots)
        priorities = np.array([
            self._it_sum[i] for i in range(self.capacity) if self.valid_mask[i]
        ])
        stats.update({
            "priority/mean": float(priorities.mean()),
            "priority/std": float(priorities.std()),
            "priority/max": float(priorities.max()),
            "priority/min": float(priorities.min()),
            "priority_fn": self.priority_fn.__name__,
            "priority_exponent": self.priority_exponent,
            "max_priority": self._max_priority,
            "enable_age_decay": self.enable_age_decay,
            "age_decay": self.age_decay,
        })

        # Age distribution (group-level)
        ages = np.array(
            [self.current_global_step - g.global_step for g in valid_groups], dtype=np.float64
        )
        stats.update({
            "age/mean": float(ages.mean()),
            "age/std": float(ages.std()),
            "age/max": float(ages.max()),
            "age/min": float(ages.min()),
            "age/median": float(np.median(ages)),
            "age/p95": float(np.percentile(ages, 95)),
        })

        # Estimated gradient-step age under current replay ratio
        estimated_replay_ratio = getattr(self, 'train_steps_per_env_step', 2.0)
        est_gradient_ages = ages * estimated_replay_ratio
        stats.update({
            "gradient_age/mean_est": float(est_gradient_ages.mean()),
            "gradient_age/max_est": float(est_gradient_ages.max()),
        })

        # Freshness (exp(-age / age_decay)); safe against age_decay<=0
        if self.age_decay and self.age_decay > 0:
            freshness_weights = np.exp(-ages / self.age_decay)
        else:
            freshness_weights = np.ones_like(ages)
        stats.update({
            "freshness/mean": float(freshness_weights.mean()),
            "freshness/std": float(freshness_weights.std()),
            "freshness/min": float(freshness_weights.min()),
            "freshness_ratio": float(np.sum(freshness_weights > 0.5) / self.num_valid),
        })

        # Per-group sample counts (how many times each group has been drawn)
        sample_counts = np.array([g.sample_count for g in valid_groups])
        if sample_counts.sum() > 0:
            stats.update({
                "sample_count/mean": float(sample_counts.mean()),
                "sample_count/max": float(sample_counts.max()),
                "sample_count/never_sampled": float(np.sum(sample_counts == 0) / self.num_valid),
            })

        return stats
