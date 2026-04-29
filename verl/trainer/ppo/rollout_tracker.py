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
Rollout metrics tracker for logging metrics before filtering.
This tracks all rollout prompts and their metrics across generation steps.
"""

from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Union

import numpy as np
from verl import DataProto

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class RolloutMetricsTracker:
    """
    Tracker for rollout metrics before filtering.
    
    This class accumulates metrics at each generation step within a training step,
    then logs all accumulated data to wandb tables when the training step completes.
    
    The tracker maintains data for:
    - current epoch
    - current training step  
    - current generation step (within training step)
    - prompt IDs
    - metric values of interest
    """
    
    def __init__(self, logger=None, metric_name: str = "acc", log_frequency: int = 1, table_name: str = "rollout_metrics"):
        """
        Initialize the rollout tracker.
        
        Args:
            logger: Tracking logger instance (from verl.utils.tracking)
            log_frequency: How often to log to wandb (every N training steps)
            table_name: Name for the wandb table
        """
        self.logger = logger
        self.log_frequency = log_frequency
        self.table_name = table_name
        self.metric_name = metric_name
        
        # Current training step data (accumulated across generation steps)
        self.current_step_data = []
        
    def track_generation_step(
        self, 
        epoch: int,
        global_step: int, 
        gen_step: int,
        batch: DataProto,
    ):
        """
        Track metrics for a single generation step.
        
        Args:
            epoch: Current training epoch
            global_step: Current training step 
            gen_step: Current generation step within training step
            batch: DataProto batch containing the rollout data
        """
        
        # Extract required data from batch
        if 'index' not in batch.non_tensor_batch:
            raise ValueError("batch.non_tensor_batch must contain 'index' field")
        
        if self.metric_name not in batch.non_tensor_batch:
            raise ValueError(f"batch.non_tensor_batch must contain '{self.metric_name}' field")
            
        prompt_ids = batch.non_tensor_batch['index']
        metric_values = batch.non_tensor_batch[self.metric_name]
        
        # Convert to lists for JSON serialization
        if isinstance(prompt_ids, np.ndarray):
            prompt_ids = prompt_ids.tolist()
        if isinstance(metric_values, np.ndarray):
            metric_values = metric_values.tolist()
        assert len(prompt_ids) == len(metric_values), f"Length of prompt_ids and metric_values must be the same, but got {len(prompt_ids)} and {len(metric_values)}"
            
        # Create generation step record
        gen_step_record = {
            'epoch': epoch,
            'global_step': global_step,
            'gen_step': gen_step,
            'prompt_ids': prompt_ids,
            'metric_values': metric_values,
            'num_prompts': len(set(prompt_ids))
        }
        
        # Add to current step accumulation
        self.current_step_data.append(gen_step_record)
        
    def finalize_training_step(self, global_step: int) -> Dict[str, Any]:
        """
        Finalize the current training step and log to wandb table.
        
        Args:
            global_step: Current training step number
            
        Returns:
            Dictionary containing summary metrics for logging
        """
        
        if not self.current_step_data:
            return {}
        
        # Create training step summary
        training_step_summary = {
            'global_step': global_step,
            'num_gen_steps': len(self.current_step_data),
            'total_prompts_rolled_out': sum(record['num_prompts'] for record in self.current_step_data),
            'generation_steps': self.current_step_data.copy()
        }
        
        # Log to wandb table if enabled
        if self.logger and WANDB_AVAILABLE and global_step % self.log_frequency == 0:
            self._log_to_wandb_table(training_step_summary, global_step)
            print(f"Logged to wandb table {self.table_name}_step_{global_step}")
        
        # Create summary metrics for logging
        summary_metrics = {
            'rollout_tracking/num_gen_steps': training_step_summary['num_gen_steps'],
            'rollout_tracking/total_prompts_rolled_out': training_step_summary['total_prompts_rolled_out'],
        }
        # rollout-level metrics
        all_metric_values = []
        for gen_step_data in self.current_step_data:
            all_metric_values.extend(gen_step_data['metric_values'])
        summary_metrics[f'rollout_tracking/rollouts_{self.metric_name}_mean'] = np.mean(all_metric_values)
        summary_metrics[f'rollout_tracking/rollouts_{self.metric_name}_std'] = np.std(all_metric_values)

        # prompt-level metrics
        # aggregate metrics by prompt across all generation steps
        prompt_metrics = defaultdict(list)
        for gen_step_data in self.current_step_data:
            prompt_ids = gen_step_data['prompt_ids']
            metric_values = gen_step_data['metric_values']
            for prompt_id, metric_value in zip(prompt_ids, metric_values):
                prompt_metrics[prompt_id].append(metric_value)
        # how many rollouts per prompt
        rollout_counts = [len(values) for values in prompt_metrics.values()]
        # how many true rollouts per prompt
        true_counts = [sum(values) for values in prompt_metrics.values()]
        # std of rollout accuracy per prompt
        true_stds = [np.std(values) for values in prompt_metrics.values()]
        # mean of rollout accuracy per prompt
        true_ratios = [np.mean(values) for values in prompt_metrics.values()]
        # count the number of prompts with each true ratio
        true_ratios_counter = Counter(true_ratios)
        
        # Add prompt-level summary metrics
        summary_metrics.update({
            f'rollout_tracking/num_prompts': len(prompt_metrics),
            f'rollout_tracking/num_rollouts_per_prompt_mean': np.mean(rollout_counts),

            f'rollout_tracking/mean_of_means': np.mean(true_ratios),
            f'rollout_tracking/std_of_means': np.std(true_ratios),
            f'rollout_tracking/max_of_means': np.max(true_ratios),
            f'rollout_tracking/min_of_means': np.min(true_ratios),

            f'rollout_tracking/mean_of_stds': np.mean(true_stds),
            f'rollout_tracking/max_of_stds': np.max(true_stds),
            f'rollout_tracking/min_of_stds': np.min(true_stds),
            
            f'rollout_tracking/frac_prompts_all_zero': true_ratios_counter.get(0, 0) / len(prompt_metrics),
            f'rollout_tracking/frac_prompts_all_one': true_ratios_counter.get(1, 0) / len(prompt_metrics),
        })
        
        # Add dynamic threshold-based metrics
        self._add_threshold_metrics(summary_metrics, rollout_counts, true_counts, len(prompt_metrics))
        
        # Clear current step data for next training step
        self.current_step_data = []
        
        return summary_metrics
    
    def _add_threshold_metrics(self, summary_metrics: Dict[str, Any], rollout_counts: List[int], true_counts: List[int], total_prompts: int):
        """
        Add dynamic threshold-based metrics to summary_metrics.
        
        Args:
            summary_metrics: Dictionary to add metrics to
            rollout_counts: List of rollout counts per prompt
            true_counts: List of true counts per prompt  
            total_prompts: Total number of prompts
        """
        # Group prompts by their rollout count
        rollout_groups = defaultdict(list)
        for i, (rollout_count, true_count) in enumerate(zip(rollout_counts, true_counts)):
            rollout_groups[rollout_count].append(true_count)
        
        # Get standard fraction thresholds to track
        standard_fractions = self._get_standard_fractions()
        
        # For each standard fraction, aggregate across all rollout counts
        fraction_counts = defaultdict(int)  # fraction -> count of prompts meeting threshold
        
        for rollout_count, true_count_list in rollout_groups.items():
            for fraction in standard_fractions:
                threshold = int(fraction * rollout_count)
                
                # For thresholds > 0.5, count prompts >= threshold (good performance)
                # For thresholds <= 0.5, count prompts <= threshold (poor performance)
                if fraction > 0.5:
                    count_meeting_threshold = sum(1 for true_count in true_count_list if true_count >= threshold)
                else:
                    count_meeting_threshold = sum(1 for true_count in true_count_list if true_count <= threshold)
                fraction_counts[fraction] += count_meeting_threshold
        
        # Add metrics for each standard fraction
        for fraction, count in fraction_counts.items():
            fraction_of_prompts = count / total_prompts
            
            # Create metric name based on comparison direction
            if fraction > 0.5:
                metric_name = f'rollout_tracking/frac_prompts_ge_{fraction}'
            else:
                metric_name = f'rollout_tracking/frac_prompts_le_{fraction}'
            
            summary_metrics[metric_name] = fraction_of_prompts
    
    def _get_standard_fractions(self) -> List[float]:
        """
        Get standardized fractions to track across all rollout counts.
        
        Returns:
            List of fraction thresholds (0.0 to 1.0) to track
        """
        return [
            0.0625,   # 6.25% - 1/16, 2/32
            0.125,    # 12.5% - 1/8, 2/16, 4/32
            0.25,     # 25% - 2/8, 4/16, 8/32
            0.75,     # 75% - 3/4, 6/8, 12/16, 24/32
            0.875,    # 87.5% - 7/8, 14/16, 28/32
            0.9375,   # 93.75% - 15/16, 30/32
        ]
        
    def _log_to_wandb_table(self, training_step_summary: Dict[str, Any], global_step: int):
        """Log current training step data to wandb table."""
        if not WANDB_AVAILABLE:
            return
            
        # Create table data - each row represents one prompt-metric pair
        table_data = []
        
        for gen_step_data in training_step_summary['generation_steps']:
            epoch = gen_step_data['epoch']
            gen_step = gen_step_data['gen_step']
            prompt_ids = gen_step_data['prompt_ids']
            metric_values = gen_step_data['metric_values']
            
            # Each prompt gets a row in the table
            for prompt_id, metric_value in zip(prompt_ids, metric_values):
                table_data.append([
                    global_step,
                    epoch, 
                    gen_step,
                    prompt_id,
                    metric_value,
                ])
        
        # Create wandb table
        columns = ["global_step", "epoch", "gen_step", "prompt_id", "metric_value"]
        table = wandb.Table(data=table_data, columns=columns)
        
        # Log the table
        table_key = f"{self.table_name}_step_{global_step}"
        self.logger.log({table_key: table}, step=global_step)
