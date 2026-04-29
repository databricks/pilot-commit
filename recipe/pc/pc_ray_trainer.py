# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
from .utils import select_prompts, extract_original_prompts
from .replay_buffer import ReplayBuffer

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.rollout_tracker import RolloutMetricsTracker
from verl.utils.profiler import marked_timer


@dataclass
class TrainingStepMetrics:
    """Tracks metrics aggregated across sampling rounds within a training step."""
    # Batch size
    num_prompt_in_batch: int = 0
    
    # Sampling rounds
    num_gen_batches: int = 0

    # Rollout counts
    num_rollouts_train: int = 0
    num_rollouts_sample_commit: int = 0
    num_rollouts_sample_explore: int = 0
    num_rollouts_sample: int = 0

    # Buffer operation counts
    local_buffer_add_count: int = 0
    local_buffer_pop_count: int = 0
    local_buffer_evict_count: int = 0
    
    # Off-policy tracking (when samples were added to buffer)
    local_off_policy_tracker: list = field(default_factory=list)

    # filtering metrics
    num_candidates: int = 0
    pass_count: int = 0
    too_correct_count: int = 0
    too_incorrect_count: int = 0
    
    @property
    def pass_rate(self):
        return self.pass_count / self.num_candidates if self.num_candidates > 0 else 0.0
    
    @property
    def too_correct_rate(self):
        return self.too_correct_count / self.num_candidates if self.num_candidates > 0 else 0.0
    
    @property
    def too_incorrect_rate(self):
        return self.too_incorrect_count / self.num_candidates if self.num_candidates > 0 else 0.0
    
    def reset(self):
        """Reset all metrics for the next training step."""
        self.num_prompt_in_batch = 0
        self.num_gen_batches = 0
        self.num_rollouts_train = 0
        self.num_rollouts_sample_commit = 0
        self.num_rollouts_sample_explore = 0
        self.num_rollouts_sample = 0
        self.local_buffer_add_count = 0
        self.local_buffer_pop_count = 0
        self.local_buffer_evict_count = 0
        self.local_off_policy_tracker.clear()
        self.num_candidates = 0
        self.pass_count = 0
        self.too_correct_count = 0
        self.too_incorrect_count = 0
    
    def to_metrics(self, global_steps: int, diversity_threshold_upper=None, diversity_threshold_lower=None):
        """Convert step metrics to metrics dictionary."""
        off_policy_metrics = {}
        # === how many steps are between the explore and commit? ===
        if self.local_off_policy_tracker:
            off_policy_steps = global_steps - np.array(self.local_off_policy_tracker)
            off_policy_metrics["off_policy_stats/local/count"] = len(off_policy_steps)
            off_policy_metrics["off_policy_stats/local/mean"] = off_policy_steps.mean()
            off_policy_metrics["off_policy_stats/local/std"] = off_policy_steps.std()
            off_policy_metrics["off_policy_stats/local/min"] = off_policy_steps.min()
            off_policy_metrics["off_policy_stats/local/max"] = off_policy_steps.max()
        else:
            off_policy_metrics["off_policy_stats/local/count"] = 0.0
            off_policy_metrics["off_policy_stats/local/mean"] = 0.0
            off_policy_metrics["off_policy_stats/local/std"] = 0.0
            off_policy_metrics["off_policy_stats/local/min"] = 0.0
            off_policy_metrics["off_policy_stats/local/max"] = 0.0
        
        # some sanity check
        assert self.num_rollouts_sample == self.num_rollouts_sample_commit + self.num_rollouts_sample_explore, \
            f"{self.num_rollouts_sample=} != {self.num_rollouts_sample_commit=} + {self.num_rollouts_sample_explore=}"
        training_metrics = {
            "training/num_prompt_in_batch": self.num_prompt_in_batch,
            "training/num_gen_batches": self.num_gen_batches,
        }

        rollout_metrics = {
            "rollout_stats/num_rollouts_train": self.num_rollouts_train,
            "rollout_stats/num_rollouts_sample_commit": self.num_rollouts_sample_commit,
            "rollout_stats/num_rollouts_sample_explore": self.num_rollouts_sample_explore,
            "rollout_stats/num_rollouts_sample": self.num_rollouts_sample,
        }

        buffer_metrics = {
            "buffer_stats/local/evict_count": self.local_buffer_evict_count,
            "buffer_stats/local/add_count": self.local_buffer_add_count,
            "buffer_stats/local/pop_count": self.local_buffer_pop_count,
        }

        filtering_metrics = {
            "filtering_stats/num_candidates": self.num_candidates,
            "filtering_stats/pass_count": self.pass_count,
            "filtering_stats/pass_rate": self.pass_rate,
            "filtering_stats/too_correct_threshold_upper": diversity_threshold_upper if diversity_threshold_upper is not None else 0.0,
            "filtering_stats/too_correct_threshold_lower": diversity_threshold_lower if diversity_threshold_lower is not None else 0.0,
            "filtering_stats/too_correct_count": self.too_correct_count,
            "filtering_stats/too_incorrect_count": self.too_incorrect_count,
            "filtering_stats/too_correct_rate": self.too_correct_rate,
            "filtering_stats/too_incorrect_rate": self.too_incorrect_rate,
        }

        return {**off_policy_metrics, **training_metrics, **buffer_metrics, **filtering_metrics, **rollout_metrics}

class RayPCTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track indices to exclude in next epoch
        self.indices_to_exclude_next_epoch = set()
        self.excluded_indices_log = []  # Log of excluded indices per epoch

        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            max_size=self.config.algorithm.buffer_size,
            max_off_steps=self.config.algorithm.buffer_max_off_steps,
            metric_name=self.config.algorithm.agg_metric
        )
    
    def mark_prompt_for_exclusion(self, batch_indices, reason="prompt too easy"):
        """
        Mark prompts to be excluded in the next epoch.
        
        Args:
            batch_indices: List of original dataset indices to exclude
            reason: Optional reason for exclusion (for logging)
        """
        if isinstance(batch_indices, (int, np.integer)):
            batch_indices = [batch_indices]
        
        # Normalize all indices to int to avoid int/str key conflicts
        batch_indices = [int(idx) for idx in batch_indices]
        self.indices_to_exclude_next_epoch.update(batch_indices)
        print(f"Marked {len(batch_indices)} prompts for exclusion in next epoch. Reason: {reason}")
        print(f"Total prompts to exclude next epoch: {len(self.indices_to_exclude_next_epoch)}")
    
    def apply_exclusions_to_dataset(self):
        """Apply accumulated exclusions to the dataset and prepare for next epoch."""
        if self.indices_to_exclude_next_epoch:
            # Add to dataset's excluded source indices
            self.train_dataset.add_excluded_source_indices(list(self.indices_to_exclude_next_epoch))
            
            # Log the exclusions
            self.excluded_indices_log.append({
                'epoch': getattr(self, 'current_epoch', 0),
                'global_step': getattr(self, 'global_steps', 0),
                'excluded_source_indices': list(self.indices_to_exclude_next_epoch),
                'count': len(self.indices_to_exclude_next_epoch)
            })
            
            print(f"Applied {len(self.indices_to_exclude_next_epoch)} exclusions to dataset")
            print(f"Dataset now has {len(self.train_dataset.get_excluded_source_indices())} total excluded source indices")
            
            # Clear the temporary set
            self.indices_to_exclude_next_epoch.clear()
    
    def get_exclusion_stats(self):
        """Get statistics about excluded indices."""
        if hasattr(self.train_dataset, 'get_excluded_source_indices') and hasattr(self.train_dataset, 'get_original_dataset_size'):
            total_excluded = len(self.train_dataset.get_excluded_source_indices())
            original_size = self.train_dataset.get_original_dataset_size()
            current_size = len(self.train_dataset)
            exclusion_ratio = (total_excluded / original_size) if original_size > 0 else 0
            
            return {
                'total_excluded': total_excluded,
                'original_dataset_size': original_size,
                'current_dataset_size': current_size,
                'exclusion_ratio': exclusion_ratio,
                'to_exclude_next_epoch': len(self.indices_to_exclude_next_epoch),
                'exclusion_log': self.excluded_indices_log
            }
        else:
            return {
                'total_excluded': 0,
                'original_dataset_size': 0,
                'current_dataset_size': len(self.train_dataset) if hasattr(self, 'train_dataset') else 0,
                'exclusion_ratio': 0,
                'to_exclude_next_epoch': len(self.indices_to_exclude_next_epoch),
                'exclusion_log': self.excluded_indices_log
            }
    
    def _adjust_commit_batch_size_for_gpu_divisibility(self, commit_batch_size: int) -> int:
        """
        Adjusts commit_batch_size to make total rollouts divisible by number of GPUs.
        Reduces commit_batch_size to the largest value that makes total rollouts divisible by n_gpus.
        
        Args:
            commit_batch_size: The number of prompts we need to (can) fetch from buffer
            
        Returns:
            Adjusted commit batch size (always <= original commit_batch_size)
        """
        n_gpus = self.resource_pool_manager.get_n_gpus()
        explore_rollouts = self.n_explore * self.config.data.gen_batch_size
        commit_rollouts = commit_batch_size * self.config.actor_rollout_ref.rollout.n
        total_rollouts = explore_rollouts + commit_rollouts
        print(f"explore_rollouts: {explore_rollouts}, commit_rollouts: {commit_rollouts}, total_rollouts: {total_rollouts}")
        
        if total_rollouts % n_gpus != 0:
            rollout_n = self.config.actor_rollout_ref.rollout.n
            
            if explore_rollouts % n_gpus == 0:
                # Simple case: explore_rollouts is divisible by n_gpus
                # We just need commit_rollouts to be divisible by LCM(n_gpus, rollout_n)
                from math import gcd
                lcm = (n_gpus * rollout_n) // gcd(n_gpus, rollout_n)
                target_commit_rollouts = (commit_rollouts // lcm) * lcm
            else:
                # General case: find largest commit_rollouts that is:
                # 1. A multiple of rollout_n (so we get integer prompts)
                # 2. Makes (explore_rollouts + commit_rollouts) divisible by n_gpus
                target_commit_rollouts = (commit_rollouts // rollout_n) * rollout_n
                while target_commit_rollouts > 0 and (explore_rollouts + target_commit_rollouts) % n_gpus != 0:
                    target_commit_rollouts -= rollout_n
            
            # Ensure we don't have negative commit rollouts
            if target_commit_rollouts < 0:
                target_commit_rollouts = 0
            
            adjusted_commit_batch_size = target_commit_rollouts // rollout_n
            print(f"Reducing commit_batch_size from {commit_batch_size} to {adjusted_commit_batch_size} to make total rollouts ({target_commit_rollouts + explore_rollouts}) divisible by {n_gpus} GPUs")
            return adjusted_commit_batch_size
        else:
            print(f"Total rollouts ({total_rollouts}) already divisible by {n_gpus} GPUs")
            return commit_batch_size

    def filter_batch(self, batch_indices, batch_metric_values, diversity_threshold_upper, diversity_threshold_lower, expected_num_prompts):
        """
        Filter commit batch based on the metric values.
        Args:
            batch_indices: List of indices of the batch
            batch_metric_values: List of metric values of the batch
            diversity_threshold_upper: Threshold for excluding prompts if too correct
            diversity_threshold_lower: Threshold for excluding prompts if too incorrect
            expected_num_prompts: Expected number of unique prompts in the batch (defaults to self.commit_bsz)
        Returns:
            pass_batch_indices: List of indices of the batch that pass the filters
            filter_stats: Dictionary containing filtering statistics
        """
        # aggregate metric values by prompt indices
        aggregated_metric_values = defaultdict(list)
        for idx, metric_value in zip(batch_indices, batch_metric_values):
            aggregated_metric_values[idx].append(metric_value)
        
        assert len(aggregated_metric_values) == expected_num_prompts, \
            f"Expected {expected_num_prompts} unique prompts in batch, but got {len(aggregated_metric_values)}"

        pass_batch_indices = []
        too_correct_indices = []
        too_incorrect_indices = []
        
        for idx, metric_values in aggregated_metric_values.items():
            mean_metric = np.mean(metric_values)
            if mean_metric >= diversity_threshold_lower and mean_metric <= (1 - diversity_threshold_upper):
                pass_batch_indices.append(idx)
            elif mean_metric > (1 - diversity_threshold_upper):
                too_correct_indices.append(idx)
            else:  # mean_metric < diversity_threshold_lower
                too_incorrect_indices.append(idx)
        
        filter_stats = {
            'num_candidates': len(aggregated_metric_values),
            'pass_count': len(pass_batch_indices),
            'too_correct_count': len(too_correct_indices),
            'too_incorrect_count': len(too_incorrect_indices),
        }
        
        return pass_batch_indices, filter_stats
    
    def exclude_batch(self, batch_indices, batch_metric_values, exclude_threshold_upper):
        """
        Exclude prompts from the batch based on the metric values.
        Args:
            batch_indices: List of indices of the batch
            batch_metric_values: List of metric values of the batch
            exclude_threshold_upper: Threshold for excluding prompts if too easy
        """
        if exclude_threshold_upper is None:
            return []
        
        # aggregate metric values by prompt indices
        aggregated_metric_values = defaultdict(list)
        for idx, metric_value in zip(batch_indices, batch_metric_values):
            aggregated_metric_values[idx].append(metric_value)

        exclude_batch_indices = []
        for idx, metric_values in aggregated_metric_values.items():
            mean_metric = np.mean(metric_values)
            if mean_metric >= exclude_threshold_upper:
                exclude_batch_indices.append(idx)
        
        return exclude_batch_indices
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.total_consumed_prompts = 0
        self.total_sampling_rounds = 0

        # configure n_commit and n_explore
        self.n_commit = self.config.actor_rollout_ref.rollout.n
        self.n_explore = self.config.algorithm.exploration.n
        if self.config.algorithm.exploration.n_effective is not None:
            self.n_explore_effective = self.config.algorithm.exploration.n_effective
        else:
            self.n_explore_effective = self.n_explore
        print(f"n_commit: {self.n_commit}, n_explore: {self.n_explore}, n_explore_effective: {self.n_explore_effective}")
        
        # Initialize rollout tracker
        metric_name = self.config.algorithm.agg_metric
        self.rollout_tracker = RolloutMetricsTracker(
            logger=logger,
            metric_name=metric_name
        )
        self.train_bsz = self.config.data.train_batch_size

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate(timing_raw={})
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                logger.finish()
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        step_metrics = TrainingStepMetrics()

        for epoch in range(1, self.config.trainer.total_epochs + 1):
            self.current_epoch = epoch
            print(f"=== Starting Epoch {epoch} ===")
            print(f"Dataloader size: {len(self.train_dataloader)}")
            print(f"Total number of prompts: {len(self.train_dataloader) * self.config.data.gen_batch_size}")
            print(f"exclude_threshold_upper: {self.config.algorithm.exclude_threshold_upper}")
            
            for batch_dict in self.train_dataloader:
                metrics = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                explore_batch: DataProto = DataProto.from_single_dict(batch_dict)
                assert len(explore_batch) == self.config.data.gen_batch_size
                step_metrics.num_gen_batches += 1
                self.total_consumed_prompts += len(explore_batch)
                self.total_sampling_rounds += 1

                # add prompt uid and from buffer flag
                explore_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(explore_batch.batch))], dtype=object)
                explore_batch.non_tensor_batch["from_buffer"] = np.array([False] * len(explore_batch.batch), dtype=bool)

                # repeat batch for exploration
                print(f"================== step: {self.global_steps}, round: {step_metrics.num_gen_batches} ===================")
                print(f"=== fetching explore_batch for exploration ===")
                print(f"len(explore_batch): {len(explore_batch)}")
                explore_batch = explore_batch.repeat(repeat_times=self.n_explore, interleave=True)
                print(f"len(explore_batch) after repeat: {len(explore_batch)}")
                step_metrics.num_rollouts_sample_explore += len(explore_batch)
                
                # pop keys to make gen_batch
                if "multi_modal_data" in explore_batch.non_tensor_batch.keys():
                    explore_gen_batch = explore_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    explore_gen_batch = explore_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # commit - get prompts from buffer for training
                    if len(self.buffer) > 0:
                        commit_batch_exists = True

                        with marked_timer("prepare_commit_batch", timing_raw):
                            print(f"=== preparing commit batch ===")
                            # fetch from buffer
                            # NOTE this will never exceed the size of train batch
                            # TODO modify this to have uniform commit batch size for each step
                            assert self.train_bsz - step_metrics.num_prompt_in_batch >= 0

                            # Adjust commit_batch_size to make total rollouts divisible by number of GPUs
                            # necessary_commit_batch_size: the number of prompts we need to fill up the train batch
                            # commit_batch_size: the number of prompts we can fetch from buffer
                            necessary_commit_batch_size = self.train_bsz if batch is None else self.train_bsz - step_metrics.num_prompt_in_batch
                            commit_batch_size = min(necessary_commit_batch_size, len(self.buffer))
                            commit_batch_size = self._adjust_commit_batch_size_for_gpu_divisibility(commit_batch_size)
                            # sample n_commit prompts from buffer, for each prompt, we sample n_explore_effective responses from it
                            commit_batch_previous_samples, when_added = self.buffer.sample(
                                size=commit_batch_size,
                                n_responses=self.n_explore_effective,
                                prompt_sampling_strategy=self.config.algorithm.buffer_sampling_method,
                                response_sampling_strategy=self.config.algorithm.buffer_response_sampling_method
                            )
                            step_metrics.local_buffer_pop_count += len(when_added)
                            step_metrics.local_off_policy_tracker.extend(when_added)
                            assert len(commit_batch_previous_samples) == commit_batch_size * self.n_explore_effective
                            print(f"per prompt, sampled {self.n_explore_effective} responses from {self.n_explore} explorations")
                            print(f"fetched {len(commit_batch_previous_samples) // self.n_explore_effective} prompts / {len(commit_batch_previous_samples)} responses from buffer")
                            print(f"prompts left in buffer: {len(self.buffer)}")
                            
                            # commit_batch_previous_samples contains responses
                            # extract original prompts and match the format of explore_batch
                            commit_batch = extract_original_prompts(
                                commit_batch_previous_samples,
                                non_tensor_batch_keys = list(explore_batch.non_tensor_batch.keys()) + ["raw_prompt_ids"]
                            )
                            # print(f"explore_batch keys: batch = {explore_batch.batch.keys()}, non_tensor_batch = {explore_batch.non_tensor_batch.keys()}")
                            # print(f"commit_batch keys: batch = {commit_batch.batch.keys()}, non_tensor_batch = {commit_batch.non_tensor_batch.keys()}")
                            
                            # repeat for rollout
                            commit_batch = commit_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            print(f"commit_batch after repeat: {len(commit_batch)}")
                            step_metrics.num_rollouts_sample_commit += len(commit_batch)

                            # pop keys to make gen_batch
                            commit_gen_batch = commit_batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids"],
                            )
                            
                            # merge exploration + commit
                            gen_batch = DataProto.concat([explore_gen_batch, commit_gen_batch])
                            print(f"len(gen_batch) after union: {len(gen_batch)}")
                    else:
                        commit_batch_exists = False
                        gen_batch = explore_gen_batch

                    
                    # rollout (explore + commit)
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    print(f"=== generating sequences ===")
                    print(f"len(gen_batch_output): {len(gen_batch_output)}")
                    step_metrics.num_rollouts_sample += len(gen_batch_output)

                    # add rollout outputs to the current batch
                    # explore batch always exists, but commit batch may not exist
                    explore_batch = explore_batch.union(gen_batch_output[:len(explore_batch)])
                    if commit_batch_exists:
                        commit_batch = commit_batch.union(gen_batch_output[len(explore_batch):])
                        new_batch = DataProto.concat([explore_batch, commit_batch])
                    else:
                        new_batch = explore_batch

                    with marked_timer("reward", timing_raw, "yellow"):
                        print("=== computing rewards ===")
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # Split commit_batch and explore_batch
                    if commit_batch_exists:
                        print("=== splitting commit and explore batch ===")
                        explore_batch = new_batch[:len(explore_batch)]
                        commit_batch = new_batch[len(explore_batch):]
                    else:
                        print("=== no commit batch ===")
                        explore_batch = new_batch
                        commit_batch = None
                    
                    # === Track rollout metrics of explore batch before filtering ===
                    # NOTE this can be new_batch (explore + commit batch) as well
                    self.rollout_tracker.track_generation_step(
                        epoch=epoch,
                        global_step=self.global_steps,
                        gen_step=step_metrics.num_gen_batches,
                        batch=explore_batch
                    )
                    
                    with marked_timer("filter_explore_batch", timing_raw, "yellow"):
                        print(f"=== filtering explore batch based on threshold {self.config.algorithm.diversity_threshold_lower} <= acc mean <= {(1 - self.config.algorithm.diversity_threshold_upper)} ===")
                        assert self.config.algorithm.agg_metric in explore_batch.non_tensor_batch.keys()
                        assert len(explore_batch) == self.config.data.gen_batch_size * self.n_explore
                        pass_indices, filter_stats = self.filter_batch(
                            explore_batch.non_tensor_batch["index"],
                            explore_batch.non_tensor_batch[self.config.algorithm.agg_metric],
                            self.config.algorithm.diversity_threshold_upper,
                            self.config.algorithm.diversity_threshold_lower,
                            self.config.data.gen_batch_size,
                        )
                        print(f"num_pass_explore_prompts: {len(pass_indices)} ({(len(pass_indices) / self.config.data.gen_batch_size) * 100:.2f}%)")
                        
                        # Track filtering effectiveness
                        step_metrics.num_candidates += filter_stats['num_candidates']
                        step_metrics.pass_count += filter_stats['pass_count']
                        step_metrics.too_correct_count += filter_stats['too_correct_count']
                        step_metrics.too_incorrect_count += filter_stats['too_incorrect_count']

                        # filter explore batch
                        explore_batch_passed = explore_batch.filter(lambda item: item.non_tensor_batch["index"] in pass_indices)
                        assert len(explore_batch_passed) == len(pass_indices) * self.n_explore
                        
                        # add explore_batch_passed to buffer
                        print("=== adding passed explore batch to buffer ===")
                        explore_batch_splitted = explore_batch_passed.split(self.n_explore)
                        self.buffer.add(explore_batch_splitted, step=self.global_steps)
                        step_metrics.local_buffer_add_count += len(explore_batch_splitted)
                        print(f"buffer length: {len(self.buffer)}")
                        
                        # exclude too easy prompts in the next epoch
                        print("=== excluding too easy prompts in the next epoch ===")
                        exclude_indices = self.exclude_batch(
                            explore_batch.non_tensor_batch["index"],
                            explore_batch.non_tensor_batch[self.config.algorithm.agg_metric],
                            exclude_threshold_upper=self.config.algorithm.exclude_threshold_upper
                        )
                        print(f"num_exclude_prompts: {len(exclude_indices)}")
                        self.mark_prompt_for_exclusion(exclude_indices)

                    if commit_batch_exists:
                        # drop unnecessary keys from commit_batch before concat
                        commit_batch.pop(non_tensor_batch_keys=['input_ids_original', 'attention_mask_original', 'position_ids_original'])

                        # concat previous explorations and commits
                        print("=== concat previous explorations and commits ===")
                        print(f"commit_batch_length: {len(commit_batch)}")
                        pilot_commit_batch = DataProto.concat([commit_batch_previous_samples, commit_batch])
                        print(f"pilot + commit batch length: {len(pilot_commit_batch)}")
                        assert len(pilot_commit_batch) == commit_batch_size * (self.n_explore_effective + self.n_commit)

                        # concat commit batch with current batch
                        batch = pilot_commit_batch if batch is None else DataProto.concat([batch, pilot_commit_batch])
                        step_metrics.num_prompt_in_batch += commit_batch_size
                        print(f"current train batch size: {step_metrics.num_prompt_in_batch}")
                    
                    # if batch is not full, keep generating
                    if step_metrics.num_prompt_in_batch < self.train_bsz:
                        print(f"{step_metrics.num_prompt_in_batch=} < {self.train_bsz=}")
                        max_num_gen_batches = self.config.algorithm.max_num_gen_batches
                        if max_num_gen_batches <= 0 or step_metrics.num_gen_batches < max_num_gen_batches:
                            print(f"sampling round: {step_metrics.num_gen_batches}, current batch size: {step_metrics.num_prompt_in_batch}, buffer size: {len(self.buffer)}. Keep generating...")
                            # progress_bar.update(1)
                            # self.gen_steps += 1
                            continue
                        else:
                            # already generated max_num_gen_batches batches, but batch is not full
                            raise ValueError(
                                f"{step_metrics.num_gen_batches=} >= {max_num_gen_batches=}."
                                + " Generated too many. Please check if your data are too difficult."
                                + " You could also try set max_num_gen_batches=0 to enable endless trials."
                            )
                    else:
                        print(f"Finished sampling {step_metrics.num_prompt_in_batch=} prompts in {step_metrics.num_gen_batches=} rounds.")
                        # # truncate the batch to the same size as the train batch size
                        # traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        # batch = batch[:traj_bsz]
                    
                    # check final train batch size
                    assert len(batch) == self.train_bsz * (self.n_explore_effective + self.n_commit)
                    assert len(set(batch.non_tensor_batch["index"])) == self.train_bsz, f"train batch indices: {len(set(batch.non_tensor_batch['index']))} != {self.train_bsz}"
                    print("=== final batch state ===")
                    print(f"len(batch): {len(batch)}")
                    print(f"train_bsz: {self.train_bsz} * (n_explore_effective: {self.n_explore_effective} + n_commit: {self.n_commit})")
                    step_metrics.num_rollouts_train += len(batch)
                    
                    # flush buffer
                    print("=== flushing buffer ===")
                    step_metrics.local_buffer_evict_count = self.buffer.flush(current_step=self.global_steps)
                    print(f"evict count: {step_metrics.local_buffer_evict_count}")

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        # entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        # entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        # old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        # metrics.update(old_log_prob_metrics)
                        # old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate(timing_raw=timing_raw)
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                # === collect and log metrics ===
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                # Step metrics
                metrics.update(step_metrics.to_metrics(
                    self.global_steps,
                    diversity_threshold_upper=self.config.algorithm.diversity_threshold_upper,
                    diversity_threshold_lower=self.config.algorithm.diversity_threshold_lower
                ))
                step_metrics.reset()  # clear step metrics
                metrics["buffer_stats/local/buffer_size"] = len(self.buffer)
                
                # Running global metrics
                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                metrics["training/total_consumed_prompts"] = self.total_consumed_prompts
                metrics["training/total_sampling_rounds"] = self.total_sampling_rounds
                
                # Finalize rollout tracking for this training step
                rollout_summary_metrics = self.rollout_tracker.finalize_training_step(self.global_steps)
                metrics.update(rollout_summary_metrics)
                
                # clear batch
                batch = None

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                
                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()
                
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    # Explicitly finish logger to ensure all data is synced
                    print("Finishing logger and syncing data...")
                    logger.finish()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
            
            print(f"=== End of Epoch {epoch} ===")
            # End of epoch - flush any partial batch and reset step metrics
            if batch is not None:
                print(f"=== flushing partial batch (size: {step_metrics.num_prompt_in_batch}/{self.train_bsz}) ===")
                batch = None
                step_metrics.reset()
            # End of epoch - flush buffer
            print("=== flushing local buffer ===")
            eoe_local_evict_count = self.buffer.flush(current_step=self.global_steps, enforce=True)
            assert len(self.buffer) == 0
            print(f"evict count: {eoe_local_evict_count}")
            
            # End of epoch - apply any marked exclusions to the dataset
            if self.indices_to_exclude_next_epoch:
                print(f"Applying {len(self.indices_to_exclude_next_epoch)} exclusions to dataset for next epoch")
                self.apply_exclusions_to_dataset()
            
            # Log exclusion statistics
            exclusion_stats = self.get_exclusion_stats()
            print(f"Exclusion stats after epoch {epoch}:")
            print(f"  Total excluded: {exclusion_stats['total_excluded']}/{exclusion_stats['original_dataset_size']}")
            print(f"  Current dataset size: {exclusion_stats['current_dataset_size']}")
            
            # Log exclusion stats to the logger
            logger.log(data={
                'exclusion_stats/total_excluded': exclusion_stats['total_excluded'],
                'exclusion_stats/original_dataset_size': exclusion_stats['original_dataset_size'],
                'exclusion_stats/current_dataset_size': exclusion_stats['current_dataset_size'],
                'exclusion_stats/exclusion_ratio': exclusion_stats['exclusion_ratio'],
                'exclusion_stats/epoch': epoch
            }, step=self.global_steps)
