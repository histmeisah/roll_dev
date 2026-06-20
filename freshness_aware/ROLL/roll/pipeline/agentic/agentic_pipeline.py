import json
import os
import os.path
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.datasets.global_dataset import GlobalDatasetManager
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.router import RouterManager
from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.configs.base_config import RouterArguments
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.utils import (
    agentic_compute_advantage,
    compute_discounted_returns,
    compute_response_level_rewards,
    dump_rollout_trajectories,
    get_agentic_response_level_mask,
)
from roll.pipeline.agentic.logprob_utils import (
    attach_replay_slot_indices,
    compute_kl_fresh_priorities,
    preserve_replay_behavior_old_log_probs,
)
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import (
    RunningMoments,
    agg_loss,
    apply_kl_penalty,
    compute_advantage,
    compute_clip_fraction,
    compute_token_reward,
    masked_mean,
    reduce_metrics,
    batch_balance
)
from roll.utils.train_infer_corrections import apply_train_infer_correction_to_batch
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

# Optional imports for replay buffer features
try:
    from roll.pipeline.agentic.replay_buffer import (
        create_replay_buffer,
        detect_manager_type_from_config,
        BaseReplayBuffer
    )
    from roll.pipeline.agentic.replay_buffer.priority_functions import get_update_metric
except ImportError:
    create_replay_buffer = None
    detect_manager_type_from_config = None
    BaseReplayBuffer = None
    get_update_metric = None

try:
    from roll.pipeline.agentic.offpolicy_monitor import (
        compute_offpolicy_metrics,
        validate_replay_batch_fields,
        log_offpolicy_diagnostics
    )
except ImportError:
    compute_offpolicy_metrics = None
    validate_replay_batch_fields = None
    log_offpolicy_diagnostics = None

# Off-policy filter (aicoder_ib parity): filter high-ratio samples before training.
try:
    from roll.pipeline.agentic.filter_utils import filter_replay_batch_with_mini_batches
except ImportError:
    filter_replay_batch_with_mini_batches = None

# Hierarchical RL (aicoder_ib parity): dual-level GAE for step-level + token-level.
try:
    from roll.pipeline.agentic.hierarchical_computer import HierarchicalAdvantageComputer
    from roll.pipeline.agentic.hierarchical_config import validate_hierarchical_config
except ImportError:
    HierarchicalAdvantageComputer = None
    validate_hierarchical_config = None


logger = get_logger()


def is_lora_training(pipeline_config: AgenticConfig) -> bool:
    return pipeline_config.actor_train.model_args.lora_target is not None


class AgenticPipeline(BasePipeline):
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        self.use_ref_model = self.pipeline_config.enable_reference and (not is_lora_training(self.pipeline_config))

        # Derived configuration for partial GPU mode (auto-detected from device_mapping)
        self.partial_gpu_mode: bool = False

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        # INIT PHASE: Create Clusters
        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )

        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        download_clusters = [self.actor_train, self.actor_infer]

        if self.use_ref_model:
            self.reference: Any = Cluster(
                name=self.pipeline_config.reference.name,
                worker_cls=self.pipeline_config.reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reference,
            )
            download_clusters.append(self.reference)


        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
            download_clusters.append(self.critic)

        # INIT PHASE: Create Reward Cluster (if device_mapping is configured)
        self.reward = None
        self.reward_scheduler = None
        if (
            self.pipeline_config.reward is not None
            and len(self.pipeline_config.reward.device_mapping) > 0
        ):
            self.reward: Any = Cluster(
                name=self.pipeline_config.reward.name,
                worker_cls=self.pipeline_config.reward.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reward,
            )
            download_clusters.append(self.reward)

        # INIT PHASE: Download Models
        self.download_models(*download_clusters)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        if self.reward:
            # Create reward scheduler as Ray named actor for environment managers to access
            self.reward_scheduler = ray.remote(RouterManager).options(
                name=f"RewardScheduler-{self.pipeline_config.reward.name}",
                get_if_exists=True,
                namespace=RAY_NAMESPACE,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                actor_cluster=self.reward,
                router_args=RouterArguments(router_name="EnvAffinityRouter"),
                num_gpus_per_node=self.pipeline_config.num_gpus_per_node
            )
            ray.get(self.reward_scheduler.initialize.remote())
            logger.info(f"Created reward scheduler as Ray named actor: RewardScheduler-{self.pipeline_config.reward.name}")

        # INIT PHASE: Create RolloutSchedulers
        self.train_rollout_scheduler = ray.remote(RolloutScheduler).options(
            name="RolloutScheduler-train",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False)).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.train_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="train",
        )

        self.val_rollout_scheduler = ray.remote(RolloutScheduler).options(
            name="RolloutScheduler-val",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False)).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.val_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="val",
        )
        self.val_dataset_manager = GlobalDatasetManager.options(name=f"val_dataset_manager",
                                                                get_if_exists=True,
                                                                namespace=RAY_NAMESPACE).remote()
        # INIT PHASE: Initialize Clusters
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs = []
        if self.reward:
            # INIT PHASE: Initialize Reward Cluster
            refs.extend(self.reward.initialize(pipeline_config=self.pipeline_config, blocking=False))
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        if self.use_ref_model:
            refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))

        ray.get([self.train_rollout_scheduler.initialize.remote(), self.val_rollout_scheduler.initialize.remote()])

        # INIT PHASE: Setup Operations
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = RunningMoments()

        # Initialize replay buffer if enabled and available
        self.replay_buffer: Optional[object] = None
        self._priority_function = 'uniform'
        if (create_replay_buffer is not None
                and hasattr(self.pipeline_config, 'replay')
                and getattr(self.pipeline_config.replay, 'enabled', False)):
            rb_cfg = self.pipeline_config.replay
            manager_type = detect_manager_type_from_config(self.pipeline_config)
            batch_size = self.pipeline_config.rollout_batch_size if rb_cfg.use_rollout_batch_size else rb_cfg.minibatch_size
            priority_function = getattr(rb_cfg, 'priority_function', 'uniform')
            priority_exponent = getattr(rb_cfg, 'priority_exponent', 0.6)
            self._priority_function = priority_function

            group_level = getattr(rb_cfg, 'group_level', True)
            logger.info(
                f"Creating replay buffer: manager_type={manager_type}, capacity={rb_cfg.capacity}, "
                f"priority_fn={priority_function}, priority_exponent={priority_exponent}, "
                f"group_level={group_level}"
            )

            self.replay_buffer = create_replay_buffer(
                manager_type=manager_type,
                capacity=rb_cfg.capacity,
                batch_size=batch_size,
                seed=self.pipeline_config.seed,
                priority_function=priority_function,
                priority_exponent=priority_exponent,
                enable_nstep=getattr(rb_cfg, 'enable_nstep', False),
                n_step=getattr(rb_cfg, 'n_step', 5),
                gamma=getattr(rb_cfg, 'nstep_gamma', 0.99),
                enable_age_decay=getattr(rb_cfg, 'enable_age_decay', False),
                age_decay=getattr(rb_cfg, 'age_decay', 1000.0),
                eviction_strategy=getattr(rb_cfg, 'eviction_strategy', 'fifo'),
                group_level=group_level,
            )
            logger.info(f"Successfully initialized replay buffer: {type(self.replay_buffer).__name__}")

        # Hierarchical RL integration (aicoder_ib parity). Only active when config.hierarchical.enabled.
        self.hierarchical_computer = None
        hier_cfg = getattr(self.pipeline_config, 'hierarchical', None)
        if (HierarchicalAdvantageComputer is not None
                and hier_cfg is not None
                and getattr(hier_cfg, 'enabled', False)):
            if validate_hierarchical_config is not None:
                validate_hierarchical_config(hier_cfg)
            self.hierarchical_computer = HierarchicalAdvantageComputer(hier_cfg)
            logger.info(
                f"Hierarchical RL enabled: "
                f"step_estimator={getattr(hier_cfg, 'step_level_estimator', '?')}, "
                f"token_estimator={getattr(hier_cfg, 'token_level_estimator', '?')}"
            )

        # Async age-decay refresh infrastructure. Only spun up when the buffer exposes
        # refresh_all_age_decay (TrajectoryReplayBuffer / StepReplayBuffer) AND age decay is on.
        # GroupReplayBuffer currently lacks refresh_all_age_decay, so this stays disabled there.
        self._age_decay_executor = None
        self._age_decay_future = None
        if (self.replay_buffer is not None
                and getattr(self.pipeline_config.replay, 'enable_age_decay', False)
                and hasattr(self.replay_buffer, 'refresh_all_age_decay')):
            self._age_decay_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="age_decay_refresh",
            )
            logger.info(
                f"Async age-decay refresh enabled: "
                f"refresh_interval={getattr(self.pipeline_config.replay, 'refresh_interval', 1)}, "
                f"age_decay={getattr(self.pipeline_config.replay, 'age_decay', 1000.0)}"
            )

        # Validate partial GPU mode configuration and set self.partial_gpu_mode
        if self.pipeline_config.partial_gpu_mode:
            self.partial_gpu_mode = self._validate_partial_gpu_config()
        else:
            self.partial_gpu_mode = False

    @torch.no_grad()
    def run(self):
        # Calculate tokens-per-second system throughput
        tps_timer = _Timer(window_size=5)

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}

            # Add overall step timing
            with Timer(name="pipeline_step_total", logger=None) as step_timer:
                with tps_timer:
                    # PHASE 1: Offload States
                    if self.pipeline_config.adv_estimator == "gae":
                        self.critic.offload_states(blocking=True)
                    self.actor_train.offload_states(blocking=True)

                    # PHASE 2: Suspend & Stop Server
                    # Suspend rollout scheduler to pause request processing
                    ray.get(self.train_rollout_scheduler.suspend.remote())

                    # Stop generation server if using async mode (will restart after model update)
                    if self.pipeline_config.async_pipeline:
                        self.actor_infer.offload_states(include=OffloadStateType.other_params)

                    # PHASE 3: Model Update
                    with Timer(name="model_update", logger=None) as model_update_timer:
                        model_update_metrics: Dict = self.model_update(global_step)
                    metrics["time/step_model_update"] =model_update_timer.last
                    metrics.update(model_update_metrics)

                    # PHASE 4: init kv cache
                    self.actor_infer.load_states()
                    if self.reward:
                        self.reward.load_states()

                    # PHASE 5: Expand Sampler (partial GPU mode, step > 0)
                    # Restore routing state: model_update loaded states to ALL GPUs, now update active_dp_ranks
                    # Step 0: active_dp_ranks initialized with all ranks {0,1,2,3}, no expand needed
                    # Step 1+: After shrink in previous iteration, active_dp_ranks was {2,3}.
                    #          model_update just loaded states to [0,1,2,3], so update routing state to match.
                    #          Use skip_load=True to avoid re-loading already-loaded model states.
                    if self.partial_gpu_mode and global_step > 0:
                        target_gpus = []
                        if hasattr(self.actor_train.worker_config, 'device_mapping') and self.actor_train.worker_config.device_mapping:
                            target_gpus.extend(self.actor_train.worker_config.device_mapping)
                        if self.pipeline_config.adv_estimator == "gae":
                            if hasattr(self.critic.worker_config, 'device_mapping') and self.critic.worker_config.device_mapping:
                                target_gpus.extend(self.critic.worker_config.device_mapping)

                        if target_gpus:
                            expand_metrics = ray.get(
                                self.train_rollout_scheduler.expand_sampler.remote(target_gpus, skip_load=True)
                            )
                            logger.info(f"Expand routing state (skip_load): {expand_metrics}")
                            metrics.update({"expand/" + k: v for k, v in expand_metrics.items()})

                    batch: DataProto = DataProto()
                    batch.meta_info = {"global_step": global_step}

                    # PHASE 6: Validation (every eval_steps) - Async
                    val_future = None
                    val_metrics = {}
                    with Timer(name="val", logger=None) as val_timer:
                        if self.pipeline_config.eval_steps > 0 and global_step % self.pipeline_config.eval_steps == 0:
                            # Submit val task to thread pool asynchronously
                            val_future = self.executor.submit(self.val, global_step)

                        # PHASE 7: Rollout Get Batch
                        with Timer(name="rollout", logger=None) as rollout_timer:
                            batch = ray.get(self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size))
                            sample_uuids = [f"{traj_id}_{i}" for i, traj_id in enumerate(batch.non_tensor_batch['traj_id'])]
                            batch.non_tensor_batch['sample_uuid'] = np.array(sample_uuids, dtype=object)
                            if "get_batch_return_start_time" in batch.meta_info:
                                metrics["time/get_batch_cost_train"] = time.time() - batch.meta_info.pop("get_batch_return_start_time")
                            actor_infer_metrics = self.actor_infer.get_metrics()
                            metrics.update(reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {})))
                            metrics.update(compute_rollout_traj_metrics(batch))

                            dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

                        metrics["time/step_rollout"] = rollout_timer.last
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                        batch.meta_info["global_step"] = global_step
                        batch.meta_info["_broadcast_non_tensor_batch"] = True
                        batch.meta_info["loss_mask_keys"] = ["response_mask"]

                        # PHASE 8: Stop Server Sync (sync mode only) - Wait for async val to complete
                        if val_future is not None:
                            val_metrics = val_future.result()

                    if len(val_metrics) > 0:
                        metrics.update(val_metrics)
                        metrics["time/step_val"] = val_timer.last

                    if not self.pipeline_config.async_pipeline:
                        # Suspend scheduler before offload actor infer, because there may be
                        # some inflight redundant trajectories.
                        ray.get(self.train_rollout_scheduler.suspend.remote())
                        self.actor_infer.offload_states()
                        if self.reward:
                            self.reward.offload_states()

                    # PHASE 9: Shrink Sampler (partial GPU mode)
                    # Partial GPU overlap: Shrink sampler to free training GPUs before training phase
                    # This offloads actor_infer models from training GPUs (e.g., [0,1]) so they can be
                    # used by actor_train and critic for the training phase. After shrink, actor_infer
                    # only has models loaded on inference-dedicated GPUs (e.g., [2,3]).
                    #
                    # Example with actor_infer on [0,1,2,3], actor_train on [0,1]:
                    #   Before shrink: actor_infer has models on all GPUs [0,1,2,3]
                    #   After shrink: actor_infer offloads from [0,1], keeps models on [2,3]
                    #   During training: actor_train uses freed GPUs [0,1]
                    #   Next iteration: model_update reloads actor_infer to all GPUs [0,1,2,3]
                    elif self.partial_gpu_mode:
                        with Timer(name="cal_ref_log_probs", logger=None) as shrink_timer:
                            target_gpus = []
                            # Collect actor_train GPUs
                            if hasattr(self.actor_train.worker_config, 'device_mapping') and self.actor_train.worker_config.device_mapping:
                                target_gpus.extend(self.actor_train.worker_config.device_mapping)
                            # Collect critic GPUs if using GAE
                            if self.pipeline_config.adv_estimator == "gae":
                                if hasattr(self.critic.worker_config, 'device_mapping') and self.critic.worker_config.device_mapping:
                                    target_gpus.extend(self.critic.worker_config.device_mapping)

                            assert target_gpus, "cannot be empty"
                            shrink_metrics = ray.get(self.train_rollout_scheduler.shrink_sampler.remote(target_gpus))
                            logger.info(f"Shrink sampler: {shrink_metrics}")
                            metrics.update({"shrink/" + k: v for k, v in shrink_metrics.items()})
                        metrics["time/step_shrink"] = shrink_timer.last

                    # PHASE 10: Replay buffer push is moved to AFTER PHASE 14 training.
                    # The storage block attaches behavior-policy log_probs before pushing.
                    # See the fresh-data replay block after actor train_step.

                    batch = compute_discounted_returns(batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma)

                    batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    # PHASE 11: Reference Log Probs
                    with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                        # TODO better the code structure, move the dynamic batching and sequence packing to worker/strategy
                        if self.pipeline_config.enable_reference:
                            worker_config = self.pipeline_config.reference if self.use_ref_model else self.pipeline_config.actor_train
                            worker = self.reference if self.use_ref_model else self.pipeline_config.actor_train
                            if worker_config.use_dynamic_batching_in_infer:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    worker.dp_size,
                                    worker_config.max_tokens_per_microbatch_in_infer,
                                    worker_config.sequence_length_round_in_infer,
                                    worker_config.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                    worker_config.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                    "reference/compute_log_probs",
                                )
                                metrics.update(dynamic_batching_metrics)
                            if not self.use_ref_model:
                                batch.meta_info["disable_adapter"] = True
                                batch.meta_info["is_offload_states"] = False
                                batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                                ref_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                            else:
                                batch_balance(batch, dp_size=self.reference.dp_size, minibatch_size=len(batch))
                                ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)

                            ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                            ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                            batch = batch.union(ref_log_probs)
                            avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                            metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                            metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                    metrics["time/step_ref_log_probs_values_reward"] = cal_timer.last

                    # PHASE 12: Old Log Probs & Values
                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        if self.pipeline_config.enable_reference and not self.use_ref_model:
                            batch.meta_info["disable_adapter"] = False
                        batch.meta_info["is_offload_states"] = False
                        if self.pipeline_config.enable_old_logprobs_recompute:
                            batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                                    self.pipeline_config.actor_train.sequence_length_round_in_infer,
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                    "actor_train/compute_log_probs",
                                )
                                metrics.update(dynamic_batching_metrics)
                            old_log_probs: DataProto = self.actor_train.compute_log_probs(batch, blocking=True)
                            batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                            avg_old_log_prob = masked_mean(batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:])
                            metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})
                            metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                            agg_entropy = agg_loss(
                                loss_mat=old_log_probs.batch["entropy"],
                                loss_mask=batch.batch["response_mask"][:, 1:],
                                loss_agg_mode="token-mean",
                            )
                            metrics.update({"critic/entropy/mean": agg_entropy.item()})
                        else:
                            # NOTE: attention_mask can be torch.bool (replay buffer stores it as bool;
                            # also true on rehydration). Force float so downstream log_probs subtraction
                            # in PPO ratio / KL works under PyTorch >= 2.x (no bool subtraction allowed).
                            batch.batch["old_log_probs"] = torch.zeros_like(
                                batch.batch["attention_mask"][:, 1:], dtype=torch.float32
                            )

                        if self.pipeline_config.adv_estimator == "gae":
                            values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                        if self.pipeline_config.adv_estimator == "gae":
                            values = DataProto.materialize_concat(data_refs=values_refs)
                            batch = batch.union(values)
                            metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))

                        # Mock ref_log_probs using old_log_probs if reference cluster is disabled
                        if not self.pipeline_config.enable_reference:
                            batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                            avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                            metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})

                    metrics["time/step_old_log_probs_values"] = cal_old_logpb_timer.last

                    # TODO 当前这个还没用处
                    with Timer(name="cal_response_level_mask", logger=None) as timer:
                        # TODO 补充完善的过滤要求，不同环境需要维持统一过滤标识
                        batch, mask_metrics = get_agentic_response_level_mask(batch, self.pipeline_config)
                        metrics.update(mask_metrics)
                    metrics["time/step_cal_response_level_mask"] = timer.last

                    # PHASE 13: Advantage Computation
                    with Timer(name="cal_response_norm_rewards", logger=None) as timer:
                        # Rewards need to be processed after grouping
                        # We can group by tag(env_type)/traj_group_id(group)/batch(rollout_batch)... to compute rewards / advantages
                        # The compute_response_level_rewards function injects a response_level_rewards key into batch.batch.
                        batch, reward_metrics = compute_response_level_rewards(batch=batch, pipeline_config=self.pipeline_config)
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                        metrics.update(reward_metrics)
                    metrics["time/step_cal_norm_rewards"] = timer.last

                    with Timer(name="cal_token_reward", logger=None) as timer:
                        # Expand compute_response_level_rewards and add kl_penalty.
                        # batch, kl_metrics = apply_kl_penalty(data=batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.pipeline_config.kl_penalty)
                        batch, token_level_metrics = compute_token_reward(batch, self.pipeline_config, self.kl_ctrl)
                        metrics.update(token_level_metrics)
                    metrics["time/step_cal_token_reward"] = timer.last

                    with Timer(name="compute_advantage", logger=None) as timer:
                        # aicoder_ib parity: route through hierarchical dual-level GAE if enabled;
                        # else fall back to standard per-batch agentic advantage.
                        if self.hierarchical_computer is not None:
                            batch = self._apply_hierarchical_advantage(batch)
                        else:
                            batch = agentic_compute_advantage(
                                data=batch,
                                gamma=self.pipeline_config.gamma,
                                lambd=self.pipeline_config.lambd,
                                adv_estimator=self.pipeline_config.adv_estimator,
                                advantage_clip=self.pipeline_config.advantage_clip,
                                whiten_advantages=self.pipeline_config.whiten_advantages,
                                whiten_rewards=self.pipeline_config.whiten_rewards,
                                pipeline_config=self.pipeline_config,
                            )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    metrics["time/step_adv"] = timer.last

                    if self.pipeline_config.enable_old_logprobs_recompute:
                        batch, corr_metrics = apply_train_infer_correction_to_batch(self.pipeline_config, batch,
                                                                                    update_mask_keys=batch.meta_info['loss_mask_keys'])
                        metrics.update(corr_metrics)

                    # PHASE 14: Training (critic + actor)
                    actor_train_metrics = None
                    with Timer(name="train_timer", logger=None) as train_timer:
                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                        # implement critic warmup
                        if self.pipeline_config.critic_warmup <= global_step:
                            batch_balance_metrics = batch_balance(batch, dp_size=self.actor_train.dp_size,
                                minibatch_size=self.actor_train.dp_size * self.pipeline_config.actor_train.training_args.per_device_train_batch_size *
                                self.pipeline_config.actor_train.training_args.gradient_accumulation_steps,
                                logging_prefix="global_seqlen/actor_train")
                            metrics.update(batch_balance_metrics)
                            # update actor
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                    self.pipeline_config.actor_train.sequence_length_round_in_train,
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                    "actor_train/train_step",
                                )
                                metrics.update(dynamic_batching_metrics)
                            # aicoder_ib parity: ask worker/strategy to collect training-time log_probs
                            # so we can reuse them below as behavior_log_probs (avoids a 2nd forward).
                            if self.replay_buffer is not None:
                                batch.meta_info["need_collect_log_probs"] = True
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
                            metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                            metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                        tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                    metrics["time/step_train"] = train_timer.last

                    # Off-policy monitor on fresh batch (aicoder_ib parity): compute ratio distribution
                    # between training-time log_probs and rollout-time old_log_probs. Reuses
                    # actor_train_metrics.batch["log_probs"] when available (no extra forward).
                    fresh_monitor_enabled = (
                        compute_offpolicy_metrics is not None
                        and getattr(self.pipeline_config, 'offpolicy_monitor', None) is not None
                        and getattr(self.pipeline_config.offpolicy_monitor, 'enabled', False)
                        and getattr(self.pipeline_config.offpolicy_monitor, 'monitor_fresh_batch', False)
                    )
                    if fresh_monitor_enabled and actor_train_metrics is not None:
                        try:
                            fresh_offpolicy_metrics = compute_offpolicy_metrics(
                                current_batch=batch,
                                actor_train_cluster=None,
                                pg_clip=self.pipeline_config.pg_clip,
                                training_metrics=actor_train_metrics,
                            )
                            # Strip raw histogram tensors before adding to metrics JSON (not serializable).
                            fresh_offpolicy_metrics.pop("_raw_importance_weights", None)
                            fresh_offpolicy_metrics.pop("_raw_log_importance_weights", None)
                            fresh_offpolicy_metrics.pop("_raw_sample_importance_weights", None)
                            metrics.update(fresh_offpolicy_metrics)
                            if (global_step % max(self.pipeline_config.logging_steps, 1) == 0
                                    and fresh_offpolicy_metrics
                                    and log_offpolicy_diagnostics is not None):
                                log_offpolicy_diagnostics(
                                    metrics=fresh_offpolicy_metrics,
                                    batch=batch,
                                    global_step=global_step,
                                    logger_func=logger.debug,
                                )
                        except Exception as e:
                            logger.warning(f"fresh off-policy monitor failed: {e}", exc_info=True)

                    # PHASE 14.5: Push fresh batch to replay buffer AFTER training.
                    # Store behavior-policy log_probs with the batch. The most faithful source is
                    # the rollout engine's infer_logprobs; next best is Phase 12 old_log_probs
                    # computed before actor updates. Training-time log_probs are only a fallback.
                    if self.replay_buffer is not None and self.pipeline_config.critic_warmup <= global_step:
                        if "infer_logprobs" in batch.batch:
                            batch.batch["behavior_log_probs"] = batch.batch["infer_logprobs"].float()
                            logger.debug("Attached infer_logprobs as behavior_log_probs")
                        elif (self.pipeline_config.enable_old_logprobs_recompute
                                and "old_log_probs" in batch.batch):
                            batch.batch["behavior_log_probs"] = batch.batch["old_log_probs"].float()
                            logger.debug("Attached pre-train old_log_probs as behavior_log_probs")
                        elif (actor_train_metrics is not None
                                and actor_train_metrics.batch is not None
                                and "log_probs" in actor_train_metrics.batch):
                            batch.batch["behavior_log_probs"] = actor_train_metrics.batch["log_probs"]
                            logger.debug("Attached training log_probs as behavior_log_probs (fallback reuse path)")
                        else:
                            with Timer(name="behavior_log_probs_fallback", logger=None) as behavior_timer:
                                batch = self._compute_and_attach_behavior_log_probs(batch)
                            metrics["time/behavior_log_probs_fallback"] = behavior_timer.last
                            logger.warning(
                                "actor_train did not return log_probs; fell back to explicit compute_log_probs"
                            )
                        with Timer(name="replay_push", logger=None) as replay_push_timer:
                            self.replay_buffer.push_from_dataproto(batch, global_step)
                            rb_stats = self.replay_buffer.get_stats()
                            metrics["replay/buffer_utilization"] = rb_stats.get("utilization", 0.0)
                            metrics["replay/num_groups"] = rb_stats.get("num_groups", rb_stats.get("current_size", 0))
                        metrics["time/step_replay_push"] = replay_push_timer.last
                        # Kick off async whole-buffer age-decay refresh; joined before next sample.
                        self._async_refresh_age_decay(global_step)

                    # PHASE 15: Replay buffer training (extra off-policy training steps)
                    if self.replay_buffer is not None and self.pipeline_config.critic_warmup <= global_step:
                        rb_cfg = self.pipeline_config.replay
                        train_steps = getattr(rb_cfg, 'train_steps_per_env_step', 1)
                        # Determine number of groups to sample
                        # For GroupReplayBuffer: batch_size = number of groups
                        # Actual trajectories = num_groups × group_size
                        group_size = self.pipeline_config.train_env_manager.group_size
                        num_groups_to_sample = self.pipeline_config.rollout_batch_size // max(group_size, 1)
                        replay_min_size = max(
                            num_groups_to_sample,
                            int(getattr(rb_cfg, 'min_size', 0) or 0),
                        )

                        # Join async age-decay refresh before sampling so priorities are up-to-date.
                        self._wait_age_decay_refresh()

                        # Resolve which metric this priority_function expects to be updated with
                        # (reward / advantage / td_error / None). None means no post-train update
                        # is needed (uniform / lifo / fifo / recency / length).
                        priority_metric = (
                            get_update_metric(self._priority_function)
                            if get_update_metric is not None else None
                        )

                        # aicoder_ib parity: off-policy filter opt-in (when enable_offpolicy_filter + ratio_clip_max)
                        enable_filter = (
                            filter_replay_batch_with_mini_batches is not None
                            and getattr(rb_cfg, 'enable_offpolicy_filter', False)
                            and getattr(rb_cfg, 'ratio_clip_max', None) is not None
                        )

                        if not self.replay_buffer.can_sample(replay_min_size):
                            rb_stats = self.replay_buffer.get_stats()
                            current_size = rb_stats.get("num_groups", rb_stats.get("current_size", 0))
                            logger.info(
                                f"Replay buffer warmup not reached "
                                f"(need min_size={replay_min_size}, have={current_size}), skipping replay"
                            )
                            metrics["replay/train_steps"] = 0
                            metrics["replay/warmup_min_size"] = replay_min_size
                            metrics["replay/current_size"] = current_size
                            metrics["time/step_replay_train"] = 0.0
                        else:
                            replay_train_steps = 0
                            with Timer(name="replay_train", logger=None) as replay_train_timer:
                                for replay_step in range(train_steps):
                                    if not self.replay_buffer.can_sample(num_groups_to_sample):
                                        logger.info(
                                            f"Replay buffer insufficient for sampling "
                                            f"(need {num_groups_to_sample} groups), skipping replay step {replay_step}"
                                        )
                                        break

                                    if enable_filter:
                                        # filter_utils internally calls sample_for_training in mini-batches,
                                        # computes current policy log_probs, and drops samples with ratio above
                                        # ratio_clip_max. This protects GRPO from extreme off-policy outliers.
                                        filter_outcome = filter_replay_batch_with_mini_batches(
                                            replay_buffer=self.replay_buffer,
                                            actor_train=self.actor_train,
                                            tokenizer=self.tokenizer,
                                            pipeline_config=self.pipeline_config,
                                            target_batch_size=num_groups_to_sample,
                                            mini_batch_size=getattr(rb_cfg, 'filter_mini_batch_size', 32),
                                            ratio_clip_max=rb_cfg.ratio_clip_max,
                                            max_attempts=getattr(rb_cfg, 'filter_max_attempts', 20),
                                            global_step=global_step,
                                            adaptive_mini_batch=getattr(rb_cfg, 'filter_adaptive_mini_batch', False),
                                        )
                                        filter_batch_out, filter_stats = filter_outcome
                                        metrics.update({f"replay/{k}": v for k, v in filter_stats.items()})

                                        if isinstance(filter_batch_out, tuple):
                                            replay_batch, sampled_indices = filter_batch_out
                                        else:
                                            replay_batch, sampled_indices = filter_batch_out, []

                                        if replay_batch is None:
                                            logger.info(
                                                f"Off-policy filter returned None at replay step {replay_step}, "
                                                "skipping this replay iteration"
                                            )
                                            break
                                    else:
                                        # Normal path: single sample from replay buffer
                                        replay_result = self.replay_buffer.sample_for_training(
                                            batch_size=num_groups_to_sample,
                                            device='cpu',
                                            sequence_length=batch.batch["input_ids"].shape[1],
                                            sample_method=getattr(rb_cfg, 'sample_method', 'uniform'),
                                            candidates_per_group=getattr(rb_cfg, 'candidates_per_group', 1),
                                            group_sampling=getattr(rb_cfg, 'group_sampling', 'uniform'),
                                            compute_importance_weights=(
                                                getattr(rb_cfg, 'importance_sampling_correction', False)
                                            ),
                                            importance_weight_beta=getattr(rb_cfg, 'importance_beta', 0.4),
                                        )
                                        if replay_result is None or replay_result[0] is None:
                                            logger.debug(f"Replay sampling returned None at step {replay_step}")
                                            break

                                        replay_batch, sampled_indices = replay_result
                                    replay_batch.meta_info["global_step"] = global_step
                                    replay_batch.meta_info["_broadcast_non_tensor_batch"] = True
                                    replay_batch.meta_info["loss_mask_keys"] = ["response_mask"]
                                    replay_batch.meta_info["is_offload_states"] = False
                                    self._log_trajectory_samples(
                                        replay_batch,
                                        global_step=global_step,
                                        source="replay",
                                        extra_meta={
                                            "replay_step": replay_step,
                                            "sampled_indices": sampled_indices,
                                        },
                                    )

                                    # Snapshot pre-adjust size and group_sizes so we can tell whether
                                    # adjust_batch reshaped the batch (in which case PER priority update
                                    # cannot be safely aligned to sampled_indices and we skip it).
                                    pre_adjust_size = replay_batch.batch["input_ids"].shape[0]
                                    group_sizes_for_update = replay_batch.meta_info.get("group_sizes", None)

                                    # Replay batch goes through the same pipeline: reward norm → advantage → train
                                    replay_batch = compute_discounted_returns(
                                        replay_batch, self.pipeline_config.adv_estimator,
                                        self.pipeline_config.step_reward_gamma
                                    )
                                    replay_batch = self.adjust_batch(
                                        replay_batch, mode=self.pipeline_config.batch_adjust_mode
                                    )
                                    post_adjust_size = replay_batch.batch["input_ids"].shape[0]
                                    adjust_changed_size = (pre_adjust_size != post_adjust_size)

                                    if priority_metric == "kl_fresh" and not adjust_changed_size:
                                        try:
                                            replay_batch = attach_replay_slot_indices(
                                                replay_batch,
                                                sampled_indices=sampled_indices,
                                                group_sizes=group_sizes_for_update,
                                            )
                                        except Exception as e:
                                            logger.warning(
                                                f"[KL-FreshPER] Failed to attach replay slot ids: {e}"
                                            )

                                    # Replay old_log_probs must remain the behavior-policy logprobs
                                    # recorded at collection time (pi_mu in the paper). Recomputing
                                    # them with the current actor would collapse the off-policy ratio
                                    # exp(log pi_theta - log pi_mu) toward 1 and break FreshPER's
                                    # priority-staleness semantics.
                                    replay_batch = preserve_replay_behavior_old_log_probs(
                                        replay_batch, logger=logger
                                    )

                                    # Reference log probs for KL
                                    if not self.pipeline_config.enable_reference:
                                        replay_batch.batch["ref_log_probs"] = replay_batch.batch["old_log_probs"].clone()
                                    elif self.use_ref_model:
                                        batch_balance(replay_batch, dp_size=self.reference.dp_size,
                                                      minibatch_size=len(replay_batch))
                                        if self.pipeline_config.reference.use_dynamic_batching_in_infer:
                                            replay_batch, _ = dynamic_batching_shard(
                                                replay_batch,
                                                self.reference.dp_size,
                                                self.pipeline_config.reference.max_tokens_per_microbatch_in_infer,
                                                self.pipeline_config.reference.sequence_length_round_in_infer,
                                                self.pipeline_config.reference.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                                self.pipeline_config.reference.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                                "replay_reference/compute_log_probs",
                                            )
                                        ref_lp_refs = self.reference.compute_log_probs(replay_batch, blocking=False)
                                        ref_lp = DataProto.materialize_concat(data_refs=ref_lp_refs)
                                        replay_batch.batch["ref_log_probs"] = ref_lp.batch["log_probs"]
                                    else:
                                        replay_batch.meta_info["disable_adapter"] = True
                                        batch_balance(replay_batch, dp_size=self.actor_train.dp_size,
                                                      minibatch_size=len(replay_batch))
                                        if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                                            replay_batch, _ = dynamic_batching_shard(
                                                replay_batch,
                                                self.actor_train.dp_size,
                                                self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                                                self.pipeline_config.actor_train.sequence_length_round_in_infer,
                                                self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                                self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                                "replay_ref_via_actor/compute_log_probs",
                                            )
                                        ref_lp_refs = self.actor_train.compute_log_probs(replay_batch, blocking=False)
                                        ref_lp = DataProto.materialize_concat(data_refs=ref_lp_refs)
                                        replay_batch.batch["ref_log_probs"] = ref_lp.batch["log_probs"]
                                        replay_batch.meta_info["disable_adapter"] = False

                                    # GAE replay needs fresh critic values before advantage computation.
                                    # On-policy data gets values in Phase 12; replay batches must do the
                                    # same recomputation here because buffer entries only carry behavior
                                    # log-probs and rewards.
                                    if self.pipeline_config.adv_estimator == "gae":
                                        batch_balance(replay_batch, dp_size=self.critic.dp_size,
                                                      minibatch_size=len(replay_batch))
                                        if self.pipeline_config.critic.use_dynamic_batching_in_infer:
                                            replay_batch, critic_dynamic_metrics = dynamic_batching_shard(
                                                replay_batch,
                                                self.critic.dp_size,
                                                self.pipeline_config.critic.max_tokens_per_microbatch_in_infer,
                                                self.pipeline_config.critic.sequence_length_round_in_infer,
                                                self.pipeline_config.critic.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                                self.pipeline_config.critic.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                                "replay_critic/compute_values",
                                            )
                                            metrics.update({
                                                f"replay/{k}": v for k, v in critic_dynamic_metrics.items()
                                            })
                                        values_refs = self.critic.compute_values(replay_batch, blocking=False)
                                        values = DataProto.materialize_concat(data_refs=values_refs)
                                        replay_batch = replay_batch.union(values)
                                        values_metrics = reduce_metrics(values.meta_info.pop("metrics", {}))
                                        metrics.update({
                                            f"replay/{k}": v for k, v in values_metrics.items()
                                        })

                                    # Reward normalization (group structure preserved by GroupReplayBuffer)
                                    replay_batch, _ = compute_response_level_rewards(
                                        batch=replay_batch, pipeline_config=self.pipeline_config
                                    )
                                    replay_batch, _ = compute_token_reward(
                                        replay_batch, self.pipeline_config, self.kl_ctrl
                                    )

                                    # Advantage computation (aicoder_ib parity: hierarchical branch)
                                    if self.hierarchical_computer is not None:
                                        replay_batch = self._apply_hierarchical_advantage(replay_batch)
                                    else:
                                        replay_batch = agentic_compute_advantage(
                                            data=replay_batch,
                                            gamma=self.pipeline_config.gamma,
                                            lambd=self.pipeline_config.lambd,
                                            adv_estimator=self.pipeline_config.adv_estimator,
                                            advantage_clip=self.pipeline_config.advantage_clip,
                                            whiten_advantages=self.pipeline_config.whiten_advantages,
                                            whiten_rewards=self.pipeline_config.whiten_rewards,
                                            pipeline_config=self.pipeline_config,
                                        )

                                    # Extract PER priority signal BEFORE batch_balance/dynamic_batching_shard
                                    # would reorder the batch. Only safe when adjust_batch left the batch
                                    # size unchanged (otherwise we can't align back to sampled_indices).
                                    per_sample_priorities = None
                                    if priority_metric is not None and not adjust_changed_size:
                                        per_sample_priorities = self._extract_priority_signal(
                                            replay_batch, priority_metric
                                        )

                                    if self.pipeline_config.enable_old_logprobs_recompute:
                                        replay_batch, _ = apply_train_infer_correction_to_batch(
                                            self.pipeline_config, replay_batch,
                                            update_mask_keys=replay_batch.meta_info['loss_mask_keys']
                                        )

                                    # Actor training on replay data
                                    batch_balance(replay_batch, dp_size=self.actor_train.dp_size,
                                        minibatch_size=self.actor_train.dp_size *
                                        self.pipeline_config.actor_train.training_args.per_device_train_batch_size *
                                        self.pipeline_config.actor_train.training_args.gradient_accumulation_steps,
                                        logging_prefix="global_seqlen/replay_train")
                                    # train_step requires global_micro_batch_indices when dynamic batching is on
                                    if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                                        replay_batch, _ = dynamic_batching_shard(
                                            replay_batch,
                                            self.actor_train.dp_size,
                                            self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                            self.pipeline_config.actor_train.sequence_length_round_in_train,
                                            self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                            self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                            "replay_actor_train/train_step",
                                        )
                                    replay_monitor_enabled = (
                                        compute_offpolicy_metrics is not None
                                        and getattr(self.pipeline_config, 'offpolicy_monitor', None) is not None
                                        and getattr(self.pipeline_config.offpolicy_monitor, 'enabled', False)
                                        and getattr(self.pipeline_config.offpolicy_monitor, 'monitor_replay_batch', False)
                                    )
                                    kl_fresh_priority_enabled = (
                                        priority_metric == "kl_fresh"
                                        and not adjust_changed_size
                                        and "_replay_slot_indices" in replay_batch.batch
                                    )
                                    if replay_monitor_enabled or kl_fresh_priority_enabled:
                                        replay_batch.meta_info["need_collect_log_probs"] = True

                                    replay_critic_train_refs = None
                                    if self.pipeline_config.adv_estimator == "gae":
                                        replay_critic_train_refs = self.critic.train_step(
                                            replay_batch, blocking=False
                                        )

                                    replay_train_refs = self.actor_train.train_step(replay_batch, blocking=False)
                                    replay_train_result = DataProto.materialize_concat(data_refs=replay_train_refs)
                                    replay_metrics = reduce_metrics(
                                        replay_train_result.meta_info.pop("metrics", {})
                                    )
                                    metrics.update({
                                        f"replay/{k}": v for k, v in replay_metrics.items()
                                    })

                                    if replay_critic_train_refs is not None:
                                        replay_critic_result = DataProto.materialize_concat(
                                            data_refs=replay_critic_train_refs
                                        )
                                        replay_critic_metrics = reduce_metrics(
                                            replay_critic_result.meta_info.pop("metrics", {})
                                        )
                                        metrics.update({
                                            f"replay/{k}": v for k, v in replay_critic_metrics.items()
                                        })

                                    # Off-policy monitor on replay batch (aicoder_ib parity)
                                    if replay_monitor_enabled and replay_train_result is not None:
                                        try:
                                            replay_offpolicy_metrics = compute_offpolicy_metrics(
                                                current_batch=replay_batch,
                                                actor_train_cluster=None,
                                                pg_clip=self.pipeline_config.pg_clip,
                                                training_metrics=replay_train_result,
                                            )
                                            replay_offpolicy_metrics.pop("_raw_importance_weights", None)
                                            replay_offpolicy_metrics.pop("_raw_log_importance_weights", None)
                                            replay_offpolicy_metrics.pop("_raw_sample_importance_weights", None)
                                            metrics.update({
                                                f"replay/{k}": v for k, v in replay_offpolicy_metrics.items()
                                            })
                                            if (global_step % max(self.pipeline_config.logging_steps, 1) == 0
                                                    and replay_offpolicy_metrics
                                                    and log_offpolicy_diagnostics is not None):
                                                log_offpolicy_diagnostics(
                                                    metrics=replay_offpolicy_metrics,
                                                    batch=replay_batch,
                                                    global_step=global_step,
                                                    logger_func=logger.debug,
                                                )
                                        except Exception as e:
                                            logger.warning(f"replay off-policy monitor failed: {e}", exc_info=True)

                                    # PER priority write-back: only when the priority_function actually
                                    # needs one (reward / advantage / td_error / reward_fresh) and we
                                    # managed to extract an aligned signal above.
                                    if kl_fresh_priority_enabled:
                                        if (
                                            replay_train_result is not None
                                            and replay_train_result.batch is not None
                                            and "log_probs" in replay_train_result.batch
                                        ):
                                            try:
                                                kl_priorities, kl_priority_metrics = compute_kl_fresh_priorities(
                                                    replay_batch=replay_batch,
                                                    current_log_probs=replay_train_result.batch["log_probs"],
                                                    eta=getattr(rb_cfg, "kl_fresh_eta", 1.0),
                                                    log_ratio_clip=getattr(rb_cfg, "kl_fresh_log_ratio_clip", 10.0),
                                                )
                                                metrics.update({
                                                    f"replay/{k}": v for k, v in kl_priority_metrics.items()
                                                })
                                                self._update_replay_priorities_by_slot(
                                                    slot_indices=replay_batch.batch["_replay_slot_indices"],
                                                    per_sample_priorities=kl_priorities,
                                                    global_step=global_step,
                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    f"[KL-FreshPER] Failed to compute/update priorities: {e}",
                                                    exc_info=True,
                                                )
                                        else:
                                            logger.warning(
                                                "[KL-FreshPER] actor.train_step did not return log_probs; "
                                                "skip KL priority update"
                                            )
                                    elif per_sample_priorities is not None:
                                        self._update_replay_priorities(
                                            sampled_indices=sampled_indices,
                                            per_sample_priorities=per_sample_priorities,
                                            group_sizes=group_sizes_for_update,
                                            global_step=global_step,
                                        )

                                    logger.info(
                                        f"Replay train step {replay_step}/{train_steps}: "
                                        f"{replay_batch.batch['input_ids'].shape[0]} trajectories"
                                    )
                                    replay_train_steps += 1

                            metrics["time/step_replay_train"] = replay_train_timer.last
                            metrics["replay/train_steps"] = replay_train_steps

                        # Surface priority-related buffer stats for debugging PER behavior.
                        rb_stats = self.replay_buffer.get_stats()
                        for k in ("priority/mean", "priority/max", "priority/min",
                                  "max_priority", "priority_fn",
                                  "age/mean", "age/max", "freshness/mean"):
                            if k in rb_stats:
                                metrics[f"replay/{k}"] = rb_stats[k]

                with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                    data_metrics = compute_train_data_metrics(batch=batch)

                metrics["time/step_compute_data_metrics"] = data_metrics_timer.last
                metrics.update(data_metrics)
                metrics["system/tps"] = tps_timer.mean_throughput
                metrics["system/samples"] = (global_step + 1) * self.pipeline_config.rollout_batch_size

                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)

                self.do_checkpoint(global_step=global_step)

                with Timer(name="log", logger=None) as log_timer:
                    if self.pipeline_config.logging_steps > 0 and global_step % self.pipeline_config.logging_steps == 0:
                        if int(os.environ.get("RAY_PROFILING", "0")):
                            timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                            os.makedirs(timeline_dir, exist_ok=True)
                            ray.timeline(
                                filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                            )

                        saved_samples = self._log_trajectory_samples(
                            batch,
                            global_step=global_step,
                            source="on_policy",
                        )
                        if saved_samples > 0:
                            logger.info(
                                f"Saved {saved_samples} on-policy trajectory samples "
                                f"to {self._trajectory_log_path()}"
                            )
                        logger.info(json.dumps(metrics, ensure_ascii=False))

                metrics["time/step_log"] = log_timer.last

            metrics["time/step_total"] = step_timer.last
            self.tracker.log(values=metrics, step=global_step)

            logger.info(f"pipeline step {global_step} finished")
            global_step += 1
            logger.info(f"epoch {global_step} finished")

        ray.get([
            self.train_rollout_scheduler.shutdown.remote(),
            self.val_rollout_scheduler.shutdown.remote(),
        ])


        logger.info("pipeline complete!")

    def _trajectory_log_path(self) -> str:
        cfg = getattr(self.pipeline_config, "trajectory_log", None)
        filename = getattr(cfg, "filename", "trajectory_samples.jsonl")
        return os.path.join(self.pipeline_config.logging_dir, filename)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): AgenticPipeline._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [AgenticPipeline._json_safe(v) for v in value]
        return value

    def _log_trajectory_samples(
        self,
        batch: DataProto,
        global_step: int,
        source: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        cfg = getattr(self.pipeline_config, "trajectory_log", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return 0
        if self.pipeline_config.logging_steps <= 0 or global_step % self.pipeline_config.logging_steps != 0:
            return 0
        if batch is None or batch.non_tensor_batch is None or "traj_id" not in batch.non_tensor_batch:
            return 0

        save_ratio = max(0.0, min(1.0, float(getattr(cfg, "save_ratio", 0.0))))
        max_samples = max(0, int(getattr(cfg, "max_samples_per_step", 0)))
        if save_ratio <= 0.0 or max_samples <= 0:
            return 0

        batch_grouped = list(batch.group_by(keys="traj_id").items())
        if len(batch_grouped) == 0:
            return 0
        sample_count = min(max_samples, int(np.ceil(len(batch_grouped) * save_ratio)))
        if sample_count <= 0:
            return 0
        if sample_count < len(batch_grouped):
            batch_grouped = random.sample(batch_grouped, sample_count)

        log_path = self._trajectory_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        written = 0
        with open(log_path, "a", encoding="utf-8") as f:
            for sample_rank, (traj_id, group_batch) in enumerate(batch_grouped):
                if "step" in group_batch.non_tensor_batch.keys():
                    indices = torch.argsort(torch.from_numpy(group_batch.non_tensor_batch["step"].astype(np.int64)))
                    group_batch.reorder(indices)

                prompt_mask = group_batch.batch["prompt_mask"]
                non_prompt_mask = torch.logical_not(group_batch.batch["prompt_mask"]) * group_batch.batch["attention_mask"]
                input_ids = group_batch.batch["input_ids"]
                prompt_ids_list = [
                    input_ids[i][mask.bool()].detach().cpu()
                    for i, mask in enumerate(prompt_mask)
                ]
                response_ids_list = [
                    input_ids[i][mask.bool()].detach().cpu()
                    for i, mask in enumerate(non_prompt_mask)
                ]
                prompts = self.tokenizer.batch_decode(prompt_ids_list, skip_special_tokens=False)
                responses = self.tokenizer.batch_decode(response_ids_list, skip_special_tokens=False)

                group_len = len(group_batch)
                for row_idx, (prompt, response) in enumerate(zip(prompts, responses)):
                    non_tensor_item = {
                        key: self._json_safe(values[row_idx])
                        for key, values in group_batch.non_tensor_batch.items()
                        if len(values) > row_idx
                    }
                    record = {
                        "global_step": global_step,
                        "source": source,
                        "sample_rank": sample_rank,
                        "row_in_trajectory": row_idx,
                        "trajectory_rows": group_len,
                        "traj_id": self._json_safe(traj_id),
                        "traj_group_id": non_tensor_item.get("traj_group_id"),
                        "tag": non_tensor_item.get("tags"),
                        "step": non_tensor_item.get("step"),
                        "episode_score": non_tensor_item.get("episode_scores"),
                        "step_score": non_tensor_item.get("step_scores"),
                        "model_answer": non_tensor_item.get("model_answer"),
                        "gold_answer": non_tensor_item.get("gold_answer"),
                        "answer_source": non_tensor_item.get("answer_source"),
                        "unboxed_answer": non_tensor_item.get("unboxed_answer"),
                        "prompt": prompt,
                        "response": response,
                        "non_tensor": non_tensor_item,
                    }
                    if "importance_weights" in group_batch.batch:
                        record["importance_weight"] = self._json_safe(
                            group_batch.batch["importance_weights"][row_idx]
                        )
                    if extra_meta is not None:
                        record["extra_meta"] = self._json_safe(extra_meta)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
        return written


    @staticmethod
    def _extract_episode_boundaries_from_batch(batch: DataProto):
        """
        aicoder_ib parity helper. Group batch rows by `traj_id` in `non_tensor_batch`
        to recover episode boundaries for hierarchical RL. Returns (boundaries, steps_per_episode),
        or (None, None) when traj_id is missing / batch is empty.
        """
        if batch.non_tensor_batch is None or "traj_id" not in batch.non_tensor_batch:
            return None, None
        traj_ids = batch.non_tensor_batch["traj_id"]
        if len(traj_ids) == 0:
            return None, None
        episode_boundaries = []
        episode_sizes = []
        current_traj_id = None
        current_episode_start = 0
        for i, traj_id in enumerate(traj_ids):
            if traj_id != current_traj_id:
                if current_traj_id is not None:
                    episode_sizes.append(i - current_episode_start)
                episode_boundaries.append(i)
                current_traj_id = traj_id
                current_episode_start = i
        if current_traj_id is not None:
            episode_sizes.append(len(traj_ids) - current_episode_start)
        if len(episode_boundaries) == 0:
            return None, None
        if len(set(episode_sizes)) == 1:
            steps_per_episode = episode_sizes[0]
        else:
            steps_per_episode = max(episode_sizes)
            logger.warning(
                f"extract_episode_boundaries: variable episode sizes {episode_sizes}, "
                f"using max {steps_per_episode}"
            )
        return episode_boundaries, steps_per_episode

    def _apply_hierarchical_advantage(self, batch: DataProto) -> DataProto:
        """
        aicoder_ib parity: replace batch['advantages'] / batch['returns'] with hierarchical
        dual-level GAE results when self.hierarchical_computer is active.

        Requires batch.batch to contain `response_level_rewards` and `values` (GAE critic path).
        Extracts done flags from non_tensor_batch["done"] if present, else falls back to
        meta_info["dones"] / episode_boundaries via traj_id grouping.
        """
        if self.hierarchical_computer is None:
            return batch

        env_rewards = batch.batch.get("response_level_rewards", None)
        if env_rewards is None:
            raise ValueError("Hierarchical RL requires response_level_rewards (env rewards)")
        token_values = batch.batch.get("values", None)
        if token_values is None:
            raise ValueError("Hierarchical RL requires values from critic")

        response_mask = batch.batch["response_mask"][:, 1:]
        original_token_rewards = batch.batch.get("token_level_rewards", None)

        dones = None
        episode_boundaries = batch.meta_info.get("episode_boundaries", None)
        steps_per_episode = batch.meta_info.get("steps_per_episode", None)

        if batch.non_tensor_batch is not None and "done" in batch.non_tensor_batch:
            done_array = batch.non_tensor_batch["done"]
            dones = torch.tensor([bool(d) for d in done_array], dtype=torch.float32,
                                 device=env_rewards.device)
        elif "dones" in batch.meta_info:
            dones_raw = batch.meta_info["dones"]
            dones = dones_raw if isinstance(dones_raw, torch.Tensor) else torch.tensor(
                dones_raw, dtype=torch.float32, device=env_rewards.device
            )
        elif episode_boundaries is None:
            episode_boundaries, steps_per_episode = self._extract_episode_boundaries_from_batch(batch)

        hier_results = self.hierarchical_computer.compute(
            env_rewards=env_rewards,
            token_values=token_values,
            response_masks=response_mask,
            original_token_rewards=original_token_rewards,
            episode_boundaries=episode_boundaries,
            steps_per_episode=steps_per_episode,
            dones=dones,
        )
        batch.batch["advantages"] = hier_results["token_advantages"]
        batch.batch["returns"] = hier_results["token_returns"]
        if "step_values" in hier_results:
            batch.batch["step_values"] = hier_results["step_values"]
        if "step_returns" in hier_results:
            batch.batch["step_returns"] = hier_results["step_returns"]
        return batch

    def _compute_and_attach_behavior_log_probs(self, batch: DataProto) -> DataProto:
        """
        aicoder_ib parity helper. Compute current-policy log_probs on `batch` via
        actor_train and attach them as `batch["behavior_log_probs"]`.

        Why: downstream pieces (replay buffer storage, filter_utils ratio checks,
        offpolicy_monitor metrics) all need the rollout-time policy log_probs. Without
        this attach, buffer stores zeros and any later off-policy ratio computation
        is meaningless.

        Cost: one extra forward pass per rollout batch (no gradients). Acceptable since
        replay training path does several forwards per step anyway.
        """
        try:
            behavior_refs = self.actor_train.compute_log_probs(batch, blocking=False)
            behavior = DataProto.materialize_concat(data_refs=behavior_refs)
            if behavior.batch is not None and "log_probs" in behavior.batch:
                batch.batch["behavior_log_probs"] = behavior.batch["log_probs"]
                logger.debug("Computed behavior_log_probs using actor_train")
            else:
                logger.warning("Failed to compute behavior_log_probs: no log_probs in result")
        except Exception as e:
            logger.warning(f"Failed to compute behavior_log_probs: {e}", exc_info=True)
        return batch

    def _async_refresh_age_decay(self, global_step: int) -> None:
        """
        Submit a full-buffer age-decay refresh to the background thread.
        The refresh runs in parallel with the ongoing GPU work; a later
        `_wait_age_decay_refresh` will join it before the next sample.
        """
        if self._age_decay_executor is None or self.replay_buffer is None:
            return
        refresh_interval = getattr(self.pipeline_config.replay, 'refresh_interval', 1)
        if global_step % refresh_interval != 0:
            return
        # Wait for any previous refresh to finish before scheduling a new one.
        self._wait_age_decay_refresh()
        self._age_decay_future = self._age_decay_executor.submit(
            self.replay_buffer.refresh_all_age_decay, global_step
        )
        logger.debug(f"[AGE_DECAY] Submitted async refresh at step {global_step}")

    def _wait_age_decay_refresh(self) -> None:
        """Join the in-flight age-decay refresh. Call before sampling."""
        if self._age_decay_future is None:
            return
        try:
            refreshed = self._age_decay_future.result(timeout=30.0)
            logger.debug(f"[AGE_DECAY] Refresh completed, refreshed={refreshed}")
        except Exception as e:
            logger.warning(f"[AGE_DECAY] Refresh failed or timed out: {e}")
        finally:
            self._age_decay_future = None

    def _update_replay_priorities(
        self,
        sampled_indices: List[int],
        per_sample_priorities: np.ndarray,
        group_sizes: Optional[List[int]],
        global_step: int,
    ) -> None:
        """
        Write priorities back into the replay buffer after a training step.

        `per_sample_priorities` is aligned to the flat batch (N for trajectory/step,
        N*K for group). When `group_sizes` is provided (GroupReplayBuffer), we
        aggregate per-trajectory priorities to per-group priorities via mean before
        writing, so indices/priorities line up with sampled_indices.
        """
        if self.replay_buffer is None or not sampled_indices:
            return
        try:
            priorities = np.asarray(per_sample_priorities, dtype=np.float32)
            if group_sizes is not None and len(group_sizes) == len(sampled_indices):
                offsets = np.cumsum([0] + list(group_sizes))
                if offsets[-1] != priorities.shape[0]:
                    logger.warning(
                        f"[PER] group_sizes sum {offsets[-1]} != priorities len {priorities.shape[0]}, "
                        f"skip priority update"
                    )
                    return
                priorities = np.array(
                    [priorities[offsets[i]:offsets[i + 1]].mean() for i in range(len(group_sizes))],
                    dtype=np.float32,
                )
            if priorities.shape[0] != len(sampled_indices):
                logger.warning(
                    f"[PER] priorities len {priorities.shape[0]} != indices len {len(sampled_indices)}, "
                    f"skip priority update"
                )
                return

            # Signature compatibility: GroupReplayBuffer.update_priorities takes (indices, priorities)
            # whereas Trajectory/StepReplayBuffer also accept current_global_step.
            import inspect
            sig = inspect.signature(self.replay_buffer.update_priorities)
            kwargs = {"indices": list(sampled_indices), "priorities": priorities}
            if "current_global_step" in sig.parameters:
                kwargs["current_global_step"] = global_step
            self.replay_buffer.update_priorities(**kwargs)
            logger.debug(
                f"[PER] updated {len(priorities)} priorities: "
                f"mean={priorities.mean():.4f}, max={priorities.max():.4f}"
            )
        except Exception as e:
            logger.warning(f"[PER] Failed to update replay priorities: {e}", exc_info=True)

    def _update_replay_priorities_by_slot(
        self,
        slot_indices: torch.Tensor,
        per_sample_priorities: torch.Tensor,
        global_step: int,
    ) -> None:
        """
        Write replay priorities back using per-row buffer slot ids.

        This path is used by KL-FreshPER because priorities are computed after
        actor.train_step returns current log_probs. By then batch_balance or
        dynamic batching may have reordered rows, so sampled_indices alone is no
        longer reliable. The `_replay_slot_indices` tensor is reordered together
        with the batch and lets us aggregate rows back to buffer slots.
        """
        if self.replay_buffer is None:
            return
        try:
            slots = slot_indices.detach().cpu().numpy().astype(np.int64)
            priorities_np = per_sample_priorities.detach().cpu().numpy().astype(np.float32)
            if slots.shape[0] != priorities_np.shape[0]:
                logger.warning(
                    f"[PER] slot len {slots.shape[0]} != priorities len {priorities_np.shape[0]}, "
                    "skip priority update"
                )
                return

            by_slot = {}
            for slot, priority in zip(slots, priorities_np):
                by_slot.setdefault(int(slot), []).append(float(priority))

            update_indices = list(by_slot.keys())
            update_priorities = np.array(
                [np.mean(by_slot[idx]) for idx in update_indices],
                dtype=np.float32,
            )

            import inspect
            sig = inspect.signature(self.replay_buffer.update_priorities)
            kwargs = {"indices": update_indices, "priorities": update_priorities}
            if "current_global_step" in sig.parameters:
                kwargs["current_global_step"] = global_step
            self.replay_buffer.update_priorities(**kwargs)
            logger.debug(
                f"[PER] updated {len(update_indices)} KL-FreshPER priorities: "
                f"mean={update_priorities.mean():.4f}, max={update_priorities.max():.4f}"
            )
        except Exception as e:
            logger.warning(f"[PER] Failed to update KL-FreshPER priorities: {e}", exc_info=True)

    @staticmethod
    def _extract_priority_signal(replay_batch: DataProto, priority_metric: str) -> Optional[np.ndarray]:
        """
        Extract per-sample priority signal from a replay batch. Returns None when
        the metric is not applicable or the required field is missing.
        """
        if priority_metric is None:
            return None
        if priority_metric == "advantage" and "advantages" in replay_batch.batch:
            adv = torch.abs(replay_batch.batch["advantages"])
            mask = replay_batch.batch["response_mask"].float() if "response_mask" in replay_batch.batch else None
            if mask is not None and mask.shape == adv.shape:
                num_tokens = mask.sum(dim=1).clamp(min=1.0)
                return (adv * mask).sum(dim=1).div(num_tokens).detach().cpu().numpy()
            return adv.mean(dim=1).detach().cpu().numpy()
        if priority_metric == "reward" and "scores" in replay_batch.batch:
            return torch.abs(replay_batch.batch["scores"].sum(dim=1)).detach().cpu().numpy()
        if priority_metric == "positive_reward" and "scores" in replay_batch.batch:
            rewards = replay_batch.batch["scores"].sum(dim=1)
            return torch.clamp(rewards, min=0.0).add(0.05).detach().cpu().numpy()
        if priority_metric == "grpo_signal":
            if "advantages" in replay_batch.batch:
                adv = torch.abs(replay_batch.batch["advantages"])
                mask = replay_batch.batch["response_mask"].float() if "response_mask" in replay_batch.batch else None
                if mask is not None and mask.shape == adv.shape:
                    num_tokens = mask.sum(dim=1).clamp(min=1.0)
                    return (adv * mask).sum(dim=1).div(num_tokens).add(0.05).detach().cpu().numpy()
                return adv.mean(dim=1).add(0.05).detach().cpu().numpy()
            if "response_level_rewards" in replay_batch.batch:
                return torch.abs(replay_batch.batch["response_level_rewards"]).add(0.05).detach().cpu().numpy()
            if "scores" in replay_batch.batch:
                rewards = replay_batch.batch["scores"].sum(dim=1)
                return torch.clamp(rewards, min=0.0).add(0.05).detach().cpu().numpy()
        if priority_metric == "td_error":
            if "td_error" in replay_batch.batch:
                td = replay_batch.batch["td_error"]
                td_abs = torch.abs(td)
                return (td_abs.mean(dim=1) if td_abs.dim() > 1 else td_abs).detach().cpu().numpy()
            if "returns" in replay_batch.batch and "values" in replay_batch.batch:
                td = torch.abs(replay_batch.batch["returns"] - replay_batch.batch["values"])
                return (td.mean(dim=1) if td.dim() > 1 else td).detach().cpu().numpy()
        return None

    def val(self, global_step):
        batch = DataProto()
        metrics = {}
        batch.meta_info["is_offload_states"] = False
        batch.meta_info["global_step"] = global_step
        ray.get(self.val_dataset_manager.reset.remote())
        eval_batch = ray.get(self.val_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.val_batch_size))

        if "get_batch_return_start_time" in eval_batch.meta_info:
            metrics["time/get_batch_cost_val"] = time.time() - eval_batch.meta_info.pop("get_batch_return_start_time")

        dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, eval_batch)
        eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
        eval_score = get_episode_scores(eval_batch)
        eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
        eval_metrics["score/max"] = torch.max(eval_score).detach().item()
        eval_metrics["score/min"] = torch.min(eval_score).detach().item()

        batch_grouped = eval_batch.group_by(keys="tags")
        for group_name, group_batch in batch_grouped.items():
            traj_group_scores = []
            batch_traj_grouped = group_batch.group_by(keys="traj_group_id")
            for batch_traj_group_name, batch_traj_group in batch_traj_grouped.items():
                traj_group_score = get_episode_scores(batch_traj_group)
                traj_group_scores.append(traj_group_score.mean().item())
            eval_score = torch.tensor(traj_group_scores, dtype=torch.float)
            eval_metrics[f"{group_name}/score/mean"] = torch.mean(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/max"] = torch.max(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/min"] = torch.min(eval_score).detach().item()

        metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
        logger.info(f"val_batch_size: {len(eval_batch)}")
        logger.info(f"val metrics: {metrics}")

        return metrics

    def adjust_batch(self, data: DataProto, mode="copy") -> DataProto:
        """
        ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/agent_system/multi_turn_rollout/utils.py#L86
        """
        actor_train_train_bsz = self.pipeline_config.actor_train.training_args.per_device_train_batch_size * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps * self.actor_train.dp_size
        actor_train_infer_bsz = self.pipeline_config.actor_train.infer_batch_size * self.actor_train.dp_size

        ref_infer_bsz = 1
        if hasattr(self, "reference"):
            ref_infer_bsz = self.pipeline_config.reference.infer_batch_size * self.reference.dp_size
        critic_train_bsz = 1
        critic_infer_bsz = 1
        if self.pipeline_config.adv_estimator == "gae":
            critic_train_bsz = self.pipeline_config.critic.training_args.per_device_train_batch_size * self.pipeline_config.critic.training_args.gradient_accumulation_steps * self.critic.dp_size
            critic_infer_bsz = self.pipeline_config.critic.infer_batch_size * self.critic.dp_size

        size_divide = np.lcm.reduce(np.array([actor_train_train_bsz, actor_train_infer_bsz, ref_infer_bsz, critic_infer_bsz, critic_train_bsz])).item()
        batch_size = data.batch.batch_size[0]
        threshold = batch_size % size_divide

        if threshold == 0:
            return data

        if mode == "auto":
            if threshold >= 0.5 * batch_size or  batch_size // size_divide == 0:
                mode = "copy"
            else:
                mode = "delete"
        elif mode == "random_sample":
            if batch_size < size_divide:
                mode = "copy"

        metrics = data.meta_info.get("metrics", {})
        metrics["system/batch_add_count"] = 0
        metrics["system/batch_remove_count"] = 0

        # 防止删除所有样本导致空批次
        if mode == "delete" and threshold >= batch_size:
            mode = "copy"

        if mode == "delete":
            remove_indices = np.random.choice(batch_size, threshold, replace=False)
            remove_indices = np.sort(remove_indices)
            keep_mask = np.ones(batch_size, dtype=bool)
            keep_mask[remove_indices] = False
            keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
            tensor_data = data.batch[keep_mask_tensor]
            non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
            adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
            metrics["system/batch_remove_count"] = len(remove_indices)
        elif mode == "copy":
            to_add = size_divide - threshold
            dup_indices = np.random.choice(batch_size, to_add, replace=True) if to_add > batch_size else np.random.choice(batch_size, to_add, replace=False)
            dup_proto = data.select_idxs(dup_indices)
            # TODO: set dup_proto response_mask to 0
            adjusted_batch = DataProto.concat([data, dup_proto])
            metrics["system/batch_add_count"] = to_add
        elif mode == "random_sample":
            select_indices = np.random.choice(batch_size, size_divide, replace=False)
            select_indices = np.sort(select_indices)
            adjusted_batch = data.select_idxs(select_indices)
            metrics["system/batch_remove_count"] = batch_size - size_divide
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        adjusted_batch.meta_info["metrics"] = metrics

        return adjusted_batch

    def _validate_partial_gpu_config(self) -> bool:
        """Derive partial_gpu_mode from device_mapping and validate all requirements.

        Universal validations (both Model A and B):
        - Reference colocation with actor_train

        Partial mode validations (Model B only - when train ⊂ infer):
        1. Minimum DP size (≥2)
        2. Async generation requirement (>0)
        3. Critic disjoint from actor_train
        4. Freed GPU capacity check
        5. TP/PP/EP compatibility
        6. At least 1 rank remains active

        Returns:
            partial_gpu_mode: True if train ⊂ infer (Configuration Model B),
                              False if train ∩ infer = ∅ (Configuration Model A)

        Raises:
            ValueError: Invalid configuration (device_mapping overlap, capacity issues,
                        DP size too small, missing async_generation_ratio, reference not colocated)
        """
        # rvst: yangpeng
        # Extract device mappings
        train_devices = set(self.actor_train.worker_config.device_mapping)
        infer_devices = set(self.actor_infer.worker_config.device_mapping)
        critic_devices = set(self.critic.worker_config.device_mapping) if hasattr(self, 'critic') and self.critic else set()
        ref_devices = set(self.reference.worker_config.device_mapping) if self.pipeline_config.enable_reference else set()
        reward_devices = set(self.reward.worker_config.device_mapping) if self.reward else set()

        # VAL: VAL_NON_EMPTY - ensure device_mapping not empty
        if not train_devices or not infer_devices:
            raise ValueError(
                f"device_mapping cannot be empty: "
                f"train={list(train_devices)}, infer={list(infer_devices)}"
            )

        # Universal validation: Reference must always colocate with actor_train (both Model A and B)
        # VAL: VAL_SUBSET (exact match) - reference colocation
        if self.pipeline_config.enable_reference:
            assert ref_devices == train_devices, (
                f"Reference device_mapping must match actor_train exactly: "
                f"ref={list(ref_devices)}, train={list(train_devices)}"
            )

        # Determine configuration mode
        if train_devices.isdisjoint(infer_devices):
            # Configuration Model A: Disjoint GPUs
            partial_gpu_mode = False
            logger.info("Detected Configuration Model A: Disjoint device_mapping, partial_gpu_mode=False")
            return partial_gpu_mode

        elif train_devices.issubset(infer_devices) and len(train_devices) < len(infer_devices):
            # Configuration Model B: Partial overlap
            partial_gpu_mode = True
            logger.info("Detected Configuration Model B: Subset device_mapping, partial_gpu_mode=True")

            # CRITICAL VALIDATIONS (6 checks for partial mode)

            # Validation 1: Minimum DP size
            # VAL: VAL_INT_RANGE(min=2, max=inf) - infer_dp_size
            infer_dp_size = self.actor_infer.worker_config.world_size
            assert infer_dp_size >= 2, (
                f"partial_gpu_mode requires actor_infer.dp_size >= 2, "
                f"got {infer_dp_size}"
            )

            # Validation 2: Async generation required
            # VAL: VAL_INT_RANGE(min=0.0, exclusive) - async_generation_ratio
            async_ratio = self.pipeline_config.async_generation_ratio
            assert async_ratio > 0, (
                f"partial_gpu_mode requires async_generation_ratio > 0, got {async_ratio}"
            )

            # Validation 3: Critic disjoint validation
            # VAL: VAL_SUBSET(critic_devices, infer_devices) + disjoint check
            if hasattr(self, 'critic') and self.critic is not None:
                assert critic_devices.issubset(infer_devices), (
                    f"Critic device_mapping must be subset of actor_infer: "
                    f"critic={list(critic_devices)}, infer={list(infer_devices)}"
                )
                assert critic_devices.isdisjoint(train_devices), (
                    f"Critic device_mapping must be disjoint from actor_train: "
                    f"critic={list(critic_devices)}, train={list(train_devices)}"
                )

            # Validation 4: Freed GPU capacity
            # VAL: VAL_INT_RANGE - freed GPU count check (no overlap)


            # Validation 5: TP/PP/EP compatibility
            # VAL: VAL_INT_RANGE(min=1) + device_mapping divisibility check
            # Extract TP and PP sizes from strategy config since workers aren't initialized yet
            infer_strategy_config = self.actor_infer.worker_config.strategy_args.strategy_config
            tp_size = infer_strategy_config.get("tensor_parallel_size", 1)
            pp_size = infer_strategy_config.get("pipeline_parallel_size", 1)

            assert tp_size >= 1 and pp_size >= 1, (
                f"tp_size and pp_size must be >= 1: tp={tp_size}, pp={pp_size}"
            )

            expected_gpu_count = tp_size * pp_size * infer_dp_size
            actual_gpu_count = len(infer_devices)
            assert expected_gpu_count == actual_gpu_count, (
                f"Parallelism configuration mismatch: "
                f"tp_size * pp_size * dp_size = {tp_size} * {pp_size} * {infer_dp_size} = {expected_gpu_count}, "
                f"but device_mapping has {actual_gpu_count} GPUs"
            )

            # Validation 6: At least 1 rank remains active
            # VAL: VAL_SUBSET, AST: AST_POSTCONDITION(remaining_ranks >= 1)
            gpus_per_dp_rank = tp_size * pp_size
            freed_gpus = train_devices | critic_devices
            freed_gpu_list = list(freed_gpus)
            self._validate_minimum_active_ranks(
                infer_dp_size, infer_devices, freed_gpu_list, gpus_per_dp_rank
            )

            logger.info(
                f"Partial GPU mode validated: infer_dp_size={infer_dp_size}, "
                f"freed_gpus={sorted(freed_gpus)}"
            )

            return partial_gpu_mode

        else:
            partial_gpu_mode = False
            assert len(train_devices) == len(infer_devices) + len(reward_devices),  "colocating mode"
            assert self.pipeline_config.async_generation_ratio == 0, "colocating mode only support sync/on-policy training"

            return partial_gpu_mode


    def _validate_minimum_active_ranks(
        self,
        infer_dp_size: int,
        infer_devices: set,
        freed_gpu_list: list,
        gpus_per_dp_rank: int
    ) -> None:
        """Validate at least 1 DP rank remains active after shrink.

        Args:
            infer_dp_size: Total DP size
            infer_devices: Infer device_mapping (as set for validation)
            freed_gpu_list: List of GPUs to free (train_devices | critic_devices)
            gpus_per_dp_rank: GPUs per DP rank (tp * pp)

        Raises:
            ValueError: If all ranks would be offloaded
        """
        # First validate that freed GPUs are subset of infer GPUs
        freed_gpu_set = set(freed_gpu_list)
        if not freed_gpu_set.issubset(infer_devices):
            raise ValueError(
                f"Freed GPUs (train + critic) must be subset of infer device_mapping: "
                f"freed={sorted(freed_gpu_list)}, infer={sorted(infer_devices)}"
            )

        # Convert infer_devices to ordered list to match DP rank assignment
        infer_devices_list = sorted(list(infer_devices))

        # Iterate through all DP ranks to find at least one that remains active
        # Each DP rank uses gpus_per_dp_rank consecutive GPUs from device_mapping
        at_least_one_active = False
        for dp_rank in range(infer_dp_size):
            # Get GPU range for this DP rank
            start_idx = dp_rank * gpus_per_dp_rank
            end_idx = start_idx + gpus_per_dp_rank
            dp_rank_gpus = set(infer_devices_list[start_idx:end_idx])

            # Check if this DP rank's GPUs are NOT in the freed set
            if dp_rank_gpus.isdisjoint(freed_gpu_set):
                at_least_one_active = True
                break

        if not at_least_one_active:
            raise ValueError(
                f"At least 1 DP rank must remain active after shrink. "
                f"All {infer_dp_size} DP ranks have at least one GPU in freed set. "
                f"infer_devices={sorted(infer_devices_list)}, freed_gpus={sorted(freed_gpu_list)}, "
                f"gpus_per_rank={gpus_per_dp_rank}"
            )

def get_episode_scores(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["episode_scores"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)

def get_traj_rollout_time(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["traj_rollout_time"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)

def get_traj_env_time(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["traj_env_time"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)


def compute_rollout_traj_metrics(batch) -> Dict:
    """
    Compute metrics for the rollout trajectory, before sample for train
    """
    episode_scores = get_episode_scores(batch)
    # fix: https://github.com/volcengine/verl/pull/60
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    prompt_mask = batch.batch["prompt_mask"].bool() # 首轮 prompt length
    prompt_lengths = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    non_prompt_mask = (torch.logical_not(batch.batch["prompt_mask"]) * batch.batch["attention_mask"]).float().sum(-1)

    metrics = {
        # score, sequence_score from env
        "rollout/score/mean": torch.mean(episode_scores).detach().item(),
        "rollout/score/max": torch.max(episode_scores).detach().item(),
        "rollout/score/min": torch.min(episode_scores).detach().item(),
        # response length
        "rollout/response_length/mean": torch.mean(response_length).detach().item(),
        "rollout/response_length/max": torch.max(response_length).detach().item(),
        "rollout/response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "rollout/prompt_length/mean": torch.mean(prompt_lengths).detach().item(),
        "rollout/prompt_length/max": torch.max(prompt_lengths).detach().item(),
        "rollout/prompt_length/min": torch.min(prompt_lengths).detach().item(),
        # non-prompt length
        "rollout/non_prompt_length/mean": torch.mean(non_prompt_mask).detach().item(),
        "rollout/non_prompt_length/max": torch.max(non_prompt_mask).detach().item(),
        "rollout/non_prompt_length/min": torch.min(non_prompt_mask).detach().item(),
    }
    return metrics

def compute_train_data_metrics(batch):
    """
    Compute metrics on the training data.
    This is different from `rollout_traj`: `rollout_traj` contains trajectory data for the entire batch,
    while under `step_wise`, `train_batch` is sampled from `rollout_batch`, so the data distributions will differ.
    """
    # token_level_scores are per-token scores assigned by the reward model, possibly after normalization/clipping
    # score denotes the raw environment reward
    episode_scores = get_episode_scores(batch)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    advantages = batch.batch["advantages"]
    # fix: https://github.com/volcengine/verl/pull/60
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    prompt_mask = batch.batch["prompt_mask"].bool() # 首轮 prompt length
    prompt_lengths = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    returns = batch.batch["returns"]
    non_prompt_mask = (torch.logical_not(batch.batch["prompt_mask"]) * batch.batch["attention_mask"]).float().sum(-1)

    # 从 batch 中提取 traj_rollout_time 相关指标
    # traj_rollout_times = []
    metrics = {
        # score, sequence_score from env
        "critic/score/mean": torch.mean(episode_scores).detach().item(),
        "critic/score/max": torch.max(episode_scores).detach().item(),
        "critic/score/min": torch.min(episode_scores).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": masked_mean(advantages, response_mask).detach().item(),
        "critic/advantages/max": torch.max(advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        "critic/advantages/min": torch.min(advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        # returns
        "critic/returns/mean": masked_mean(returns, response_mask).detach().item(),
        "critic/returns/max": torch.max(returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        "critic/returns/min": torch.min(returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        # response length
        "tokens/response_length/mean": torch.mean(response_length).detach().item(),
        "tokens/response_length/max": torch.max(response_length).detach().item(),
        "tokens/response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "tokens/prompt_length/mean": torch.mean(prompt_lengths).detach().item(),
        "tokens/prompt_length/max": torch.max(prompt_lengths).detach().item(),
        "tokens/prompt_length/min": torch.min(prompt_lengths).detach().item(),
        # prompt length(sys_obs)
        # "tokens/prompt_length_sys_obs/mean": torch.mean(prompt_lengths_sys_obs).detach().item(),
        # "tokens/prompt_length_sys_obs/max": torch.max(prompt_lengths_sys_obs).detach().item(),
        # "tokens/prompt_length_sys_obs/min": torch.min(prompt_lengths_sys_obs).detach().item(),
        # non-prompt length
        "tokens/non_prompt_length/mean": torch.mean(non_prompt_mask).detach().item(),
        "tokens/non_prompt_length/max": torch.max(non_prompt_mask).detach().item(),
        "tokens/non_prompt_length/min": torch.min(non_prompt_mask).detach().item(),
    }

    if "values" in batch.batch.keys():
        values = batch.batch["values"]
        # values
        metrics.update(
            {
                "critic/values/mean": masked_mean(values, response_mask).detach().item(),
                "critic/values/max": torch.max(values[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
                "critic/values/min": torch.min(values[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
            }
        )
    if "episode_rewards_norm" in batch.batch.keys():
        episode_rewards_norm = batch.batch["episode_rewards_norm"]
        step_rewards_norm = batch.batch["step_rewards_norm"]
        metrics.update({
            "critic/episode_rewards_norm/mean": episode_rewards_norm.mean().detach().item(),
            "critic/episode_rewards_norm/max": episode_rewards_norm.max().detach().item(),
            "critic/episode_rewards_norm/min": episode_rewards_norm.min().detach().item(),
            "critic/step_rewards_norm/mean": step_rewards_norm.mean().detach().item(),
            "critic/step_rewards_norm/max": step_rewards_norm.max().detach().item(),
            "critic/step_rewards_norm/min": step_rewards_norm.min().detach().item(),
        })
    return metrics

class GroupFilter:
    """
    User defined group filter.
    """
    def __init__(self, config: AgenticConfig, env_manager_config: EnvManagerConfig, mode: str):
        pass

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        """
        return True to filter out this group
        """
        return False
