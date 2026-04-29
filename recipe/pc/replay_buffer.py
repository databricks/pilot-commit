import random
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple
from verl.protocol import DataProto


class ReplayBuffer:
    """
    Replay buffer for explore-commit algorithm.
    
    This buffer stores exploration results (prompts with their generated responses and metrics)
    and provides sampling strategies for selecting promising prompts for training.
    """
    
    def __init__(self, max_size: int, max_off_steps: int, metric_name: str = "acc"):
        """
        Initialize replay buffer.
        
        Args:
            max_size (int): Maximum number of prompts in the buffer
            max_off_steps (int): Maximum tolerance for off policy samples.
            metric_name (str): Name of the metric to track (e.g., "acc")
        """
        self.max_size = max_size
        self.max_off_steps = max_off_steps
        self.metric_name = metric_name
        
        # Buffer: stores exploration data grouped by prompt UID
        # Structure: {prompt_uid: {"data": DataProto, "step": int, "metric_stats": dict}}
        self.buffer = {}
        

    def add(self, batch: List[DataProto], step: int, metric_name: str = "acc"):
        """
        Add exploration data to the buffer.
        
        Args:
            batch (DataProto): Batch of exploration data with responses and metrics
            step (int): Global step of data added to buffer
            metric_name (str): Name of the metric to track (e.g., "acc")
        """
        if len(batch) == 0:
            return
            
        for dp_item in batch:
            index_array = dp_item.non_tensor_batch["index"]
            prompt_id_cand = np.unique(index_array)

            assert len(prompt_id_cand) == 1, f"each item in buffer should come from a unique prompt"
            prompt_id = int(prompt_id_cand[0])

            # add metric stats
            assert metric_name in dp_item.non_tensor_batch
            metric_values = dp_item.non_tensor_batch[metric_name]
            metric_stats = {
                "mean": float(np.mean(metric_values)),
                "std": float(np.std(metric_values)),
                "min": float(np.min(metric_values)),
                "max": float(np.max(metric_values)),
                "count": len(metric_values),
            }

            self.buffer[prompt_id] = {
                "data": dp_item,
                "step": step,
                "metric_stats": metric_stats,
            }
    
    def flush(self, current_step: int, enforce=False):
        """
        Evict items in buffer if they are too old.
        
        Args:
            current_step (int): Current global step
        """
        if enforce:
            evict_count = len(self.buffer)
            self.buffer.clear()
            return evict_count
            
        evict_count = 0
        for uid, item in list(self.buffer.items()):
            if current_step - item["step"] > self.max_off_steps:
                del self.buffer[uid]
                evict_count += 1
        return evict_count

    def pop(self, prompt_indices: List[int]) -> Tuple[DataProto, List[int]]:
        """
        Pop prompts from buffer.
        
        Args:
            prompt_indices: List of prompt indices to pop
        """
        data = []
        when_added = []
        for uid in prompt_indices:
            assert uid in self.buffer, f"prompt {uid} not in buffer"
            data.append(self.buffer[uid]["data"])
            when_added.append(self.buffer[uid]["step"])
            del self.buffer[uid]
        return DataProto.concat(data), when_added
    
    
    def sample(
        self,
        size: int,
        n_responses: int,
        prompt_sampling_strategy: str = "random",
        response_sampling_strategy: str = "max_variance"
    ) -> Optional[DataProto]:
        """
        Sample prompts from buffer for training.
        
        Args:
            size (int): Number of prompts to sample
            n_responses (int): Number of responses within each prompt to sample
            prompt_sampling_strategy (str): Prompt sampling strategy - "diversity", "random", "best", "worst"
            response_sampling_strategy (str): Response sampling strategy - "max_variance", "first_n"
            
        Returns:
            Optional[DataProto]: Sampled data or None if buffer is empty
        """
        if len(self.buffer) == 0:
            return [], []
            
        available_uids = list(self.buffer.keys())

        if len(available_uids) <= size:
            # print(f"selected uids: {len(available_uids)}")
            data = self.get_all()
            when_added = [self.buffer[uid]["step"] for uid in self.buffer.keys()]
            self.clear()
            return data, when_added
        
        if prompt_sampling_strategy == "random":
            selected_uids = random.sample(available_uids, min(size, len(available_uids)))
        # TODO check other methods but "random"
        elif prompt_sampling_strategy == "max_variance":
            # Sample to maximize variance of metric values
            selected_uids = self._sample_by_variance(available_uids, size)
        elif prompt_sampling_strategy == "most_recent":
            # Sample prompts from the most recent ones
            selected_uids = self._sample_by_recent(available_uids, size)
        elif prompt_sampling_strategy == "least_recent":
            # Sample prompts from the least recent ones
            selected_uids = self._sample_by_recent(available_uids, size, reverse=True)
        else:
            raise ValueError(f"Unknown sampling strategy: {prompt_sampling_strategy}")
        assert len(selected_uids) == size
        # print(f"selected uids: {len(selected_uids)}")    
        
        # Extract prompt data
        sampled_data = []
        when_added = []
        for uid in selected_uids:
            prompt_data = self.buffer[uid]["data"]
            when_added.append(self.buffer[uid]["step"])
            # Mark as from buffer
            prompt_data.non_tensor_batch["from_buffer"] = np.array([True] * len(prompt_data), dtype=bool)

            # Sample responses if needed
            if n_responses != len(prompt_data):
                if response_sampling_strategy == "first_n":
                    prompt_data = prompt_data.select_idxs(np.arange(n_responses))
                elif response_sampling_strategy == "max_variance":
                    sampled_idx = self._maxvar_downsample_binary(prompt_data.non_tensor_batch[self.metric_name], n_responses)
                    prompt_data = prompt_data.select_idxs(sampled_idx)
                else:
                    raise ValueError(f"Unknown response sampling strategy: {response_sampling_strategy}")
            sampled_data.append(prompt_data)
            
        # Remove sampled prompts from buffer (they're now being used for training)
        for uid in selected_uids:
            del self.buffer[uid]
        
        # Concatenate all sampled data
        data = DataProto.concat(sampled_data)
        assert len(data) == size * n_responses

        assert len(when_added) == size
        return data, when_added

    def get_all(self) -> DataProto:
        """
        Get all data from buffer.
        """
        if len(self.buffer) == 0:
            return DataProto.from_dict(tensors={}, non_tensors={})
        
        all_data = []
        for uid, item in self.buffer.items():
            prompt_data = item["data"]
            # Mark as from buffer
            prompt_data.non_tensor_batch["from_buffer"] = np.array([True] * len(prompt_data), dtype=bool)
            all_data.append(prompt_data)
            
        return DataProto.concat(all_data)

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()

    def _maxvar_downsample_binary(self, rewards: np.ndarray, m: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Selects a subset of size m that maximizes variance in binary rewards.

        Args:
            rewards (np.ndarray): Array of 0/1 rewards of length n.
            m (int): Size of subset to select.
            seed (int, optional): Random seed for reproducibility if multiple choices are possible.

        Returns:
            indices (list): Indices of selected rollouts.
        """
        rng = np.random.default_rng(seed)
        
        ones_idx = np.where(rewards == 1)[0]
        zeros_idx = np.where(rewards == 0)[0]

        # Ideally half-half
        need_ones = m // 2
        need_zeros = m - need_ones

        # Adjust if not enough ones or zeros
        if len(ones_idx) < need_ones:
            need_ones = len(ones_idx)
            need_zeros = m - need_ones
        elif len(zeros_idx) < need_zeros:
            need_zeros = len(zeros_idx)
            need_ones = m - need_zeros

        chosen_ones = rng.choice(ones_idx, size=need_ones, replace=False) if need_ones > 0 else []
        chosen_zeros = rng.choice(zeros_idx, size=need_zeros, replace=False) if need_zeros > 0 else []

        return np.concatenate([chosen_ones, chosen_zeros])

    
    def _sample_by_variance(self, available_uids: List[int], batch_size: int) -> List[int]:
        """
        Sample prompts to maximize std (variance) of metric values.
        """
        if len(available_uids) <= batch_size:
            return available_uids
        
        # Sort by metric std (higher std = more diverse)
        uid_std = []
        for uid in available_uids:
            std = self.buffer[uid]["metric_stats"]["std"]
            step = self.buffer[uid]["step"]
            uid_std.append((uid, std, step))
            
        # sort by std first, then by step (higher the better)
        uid_std.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Take top diverse prompts
        return [uid for uid, _, _ in uid_std[:batch_size]]

    def _sample_by_recent(self, available_uids: List[int], batch_size: int, reverse: bool = True) -> List[int]:
        """
        Sample prompts by metric value.
        """
        if len(available_uids) <= batch_size:
            return available_uids
            
        # Sort by recent value
        uid_recent = []
        for uid in available_uids:
            recent_value = self.buffer[uid]["step"]
            uid_recent.append((uid, recent_value))
            
        uid_recent.sort(key=lambda x: x[1], reverse=reverse)
        
        # Take top/bottom prompts
        return [uid for uid, _ in uid_recent[:batch_size]]

    def __len__(self) -> int:
        """
        Return number of prompts in buffer.
        """
        return len(self.buffer)

    def is_empty(self) -> bool:
        """
        Check if buffer is empty.
        """
        return len(self.buffer) == 0