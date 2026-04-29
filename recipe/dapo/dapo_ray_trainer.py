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
from dataclasses import dataclass
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
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
    num_prompt_in_batch: int = 0
    num_gen_batches: int = 0
    num_rollouts_train: int = 0
    num_rollouts_sample: int = 0

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
        self.num_rollouts_sample = 0
        self.num_candidates = 0
        self.pass_count = 0
        self.too_correct_count = 0
        self.too_incorrect_count = 0

    def to_metrics(self, n_rollouts=None):
        """Convert step metrics to metrics dictionary.
        
        Args:
            n_rollouts: Number of rollouts per prompt, used to compute effective diversity threshold
        """
        training_metrics = {
            "training/num_prompt_in_batch": self.num_prompt_in_batch,
            "training/num_gen_batches": self.num_gen_batches,
        }
        rollout_metrics = {
            "rollout_stats/num_rollouts_train": self.num_rollouts_train,
            "rollout_stats/num_rollouts_sample": self.num_rollouts_sample,
        }
        if self.num_candidates > 0:
            # Compute effective diversity threshold: need at least 1 diverse response out of n
            # So the minimum diversity ratio to pass filtering is 1/n
            effective_threshold = 1.0 / n_rollouts if n_rollouts is not None and n_rollouts > 0 else 0.0
            
            filtering_metrics = {
                "filtering_stats/num_candidates": self.num_candidates,
                "filtering_stats/pass_count": self.pass_count,
                "filtering_stats/pass_rate": self.pass_rate,
                "filtering_stats/too_correct_threshold_upper": effective_threshold,
                "filtering_stats/too_correct_threshold_lower": effective_threshold,
                "filtering_stats/too_correct_count": self.too_correct_count,
                "filtering_stats/too_incorrect_count": self.too_incorrect_count,
                "filtering_stats/too_correct_rate": self.too_correct_rate,
                "filtering_stats/too_incorrect_rate": self.too_incorrect_rate,
            }
        else:
            filtering_metrics = {}
        return {**training_metrics, **rollout_metrics, **filtering_metrics} 

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

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

        # Initialize rollout tracker
        metric_name = self.config.algorithm.filter_groups.metric
        self.rollout_tracker = RolloutMetricsTracker(
            logger=logger,
            metric_name=metric_name
        )

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

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                step_metrics.num_gen_batches += 1
                self.total_consumed_prompts += len(new_batch)
                self.total_sampling_rounds += 1
                
                # Print progress info
                print(f"================== step: {self.global_steps}, round: {step_metrics.num_gen_batches} ===================")
                print(f"len(new_batch): {len(new_batch)}")

                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    print(f"=== generating sequences ===")
                    print(f"len(gen_batch_output): {len(gen_batch_output)}")
                    step_metrics.num_rollouts_sample += len(gen_batch_output)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

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

                    # === log reward stats ===
                    if metric_name == "seq_final_reward":
                        # Turn to numpy for easier filtering
                        new_batch.non_tensor_batch["seq_final_reward"] = (
                            new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        )
                    elif metric_name == "seq_reward":
                        new_batch.non_tensor_batch["seq_reward"] = (
                            new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                        )

                    # === Track rollout metrics before filtering ===
                    assert len(new_batch) == self.config.data.gen_batch_size * self.config.actor_rollout_ref.rollout.n, f"Current batch size is {len(new_batch)}, but expected {self.config.data.gen_batch_size * self.config.actor_rollout_ref.rollout.n}"
                    self.rollout_tracker.track_generation_step(
                        epoch=epoch,
                        global_step=self.global_steps,
                        gen_step=step_metrics.num_gen_batches,
                        batch=new_batch
                    )
                    

                    # === filter if enabled ===
                    if not self.config.algorithm.filter_groups.enable:
                        # no filtering, just use the new batch
                        step_metrics.num_prompt_in_batch += len(set(new_batch.non_tensor_batch["index"]))
                        assert len(new_batch) == self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n, f"Current batch size is {len(new_batch)}, but expected {self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n}"
                        batch = new_batch
                        print(f"Regular sampling: Finished sampling {step_metrics.num_prompt_in_batch} prompts in 1 round.")
                    else:
                        # Collect metric values per prompt and categorize
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        # Categorize prompts based on response variance
                        mixed_prompt_uids = []
                        too_correct_prompt_uids = []
                        too_incorrect_prompt_uids = []
                        
                        for uid, metric_vals in prompt_uid2metric_vals.items():
                            has_variance = np.std(metric_vals) > 0
                            all_zero = not np.any(metric_vals)
                            all_one = bool(np.all(metric_vals))
                            
                            if has_variance or len(metric_vals) == 1:
                                # Keep prompts with variance or single response
                                mixed_prompt_uids.append(uid)
                            elif all_one:
                                # All responses correct
                                too_correct_prompt_uids.append(uid)
                            elif all_zero:
                                # All responses incorrect
                                too_incorrect_prompt_uids.append(uid)

                        # Update filtering metrics
                        step_metrics.num_candidates += self.config.data.gen_batch_size
                        step_metrics.pass_count += len(mixed_prompt_uids)
                        step_metrics.too_correct_count += len(too_correct_prompt_uids)
                        step_metrics.too_incorrect_count += len(too_incorrect_prompt_uids)
                        
                        print(f"Filtering: {len(mixed_prompt_uids)} mixed / {len(too_correct_prompt_uids)} too_correct / {len(too_incorrect_prompt_uids)} too_incorrect out of {self.config.data.gen_batch_size} prompts")

                        # Filter trajectories to keep only mixed prompts
                        new_batch = new_batch.filter(lambda item: item.non_tensor_batch["uid"] in mixed_prompt_uids)
                        
                        # Calculate how many more prompts we need for the training batch
                        necessary_prompt_bsz = self.config.data.train_batch_size - step_metrics.num_prompt_in_batch
                        assert necessary_prompt_bsz > 0, f"necessary_prompt_bsz={necessary_prompt_bsz} <= 0"
                        
                        # Take up to necessary_prompt_bsz prompts from filtered batch
                        prompts_to_add = min(len(mixed_prompt_uids), necessary_prompt_bsz)
                        traj_to_add = prompts_to_add * self.config.actor_rollout_ref.rollout.n
                        new_batch = new_batch[:traj_to_add]
                        print(f"Adding {prompts_to_add} prompts ({traj_to_add} trajectories) to training batch")

                        # Concat to batch and update prompt count with ACTUAL prompts added
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
                        step_metrics.num_prompt_in_batch += prompts_to_add

                        # NOTE: When prompts after filtering is less than train batch size,
                        # we go to the next generation round
                        prompt_bsz = self.config.data.train_batch_size
                        if step_metrics.num_prompt_in_batch < prompt_bsz:
                            print(f"num_prompt_in_batch={step_metrics.num_prompt_in_batch} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or step_metrics.num_gen_batches < max_num_gen_batches:
                                print(f"current batch size: {step_metrics.num_prompt_in_batch}, sampling round: {step_metrics.num_gen_batches}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                # already generated max_num_gen_batches batches, but batch is not full
                                raise ValueError(
                                    f"num_gen_batches={step_metrics.num_gen_batches} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            print(f"Finished sampling {step_metrics.num_prompt_in_batch} prompts in {step_metrics.num_gen_batches} rounds.")
                            # Sanity check: due to min() logic, we should have exactly train_batch_size prompts
                            assert step_metrics.num_prompt_in_batch == prompt_bsz, \
                                f"Expected exactly {prompt_bsz} prompts, but got {step_metrics.num_prompt_in_batch}"

                    # === Updating ===
                    print("=== final batch state ===")
                    print(f"len(batch): {len(batch)}")
                    print(f"train_bsz: {self.config.data.train_batch_size} * n_rollouts: {self.config.actor_rollout_ref.rollout.n}")
                    assert len(batch) == self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n, \
                        f"Current batch size is {len(batch)}, but expected {self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n}"
                    step_metrics.num_rollouts_train += len(batch)

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

                        # old_log_prob_metrics = {
                        #     "actor/entropy": entropy_agg.detach().item(),
                        # }
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

                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                # === collect and log metrics ===
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                # Step metrics
                metrics.update(step_metrics.to_metrics(n_rollouts=self.config.actor_rollout_ref.rollout.n))
                step_metrics.reset()  # clear step metrics
                
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
