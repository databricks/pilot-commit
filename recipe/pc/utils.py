import numpy as np
from typing import Dict, List, Tuple
from verl.protocol import DataProto
from tensordict import TensorDict
import torch


def select_prompts(
    prompt_indices: np.ndarray,
    metric_vals: np.ndarray,
    diversity_threshold_upper: float = 0.1,
    diversity_threshold_lower: float = 0.1,
    exclude_threshold_upper: float = None,
) -> Dict[str, List[str]]:
    """
    Select promising prompts based on metric values and diversity.
    
    Args:
        prompt_indices: Array of prompt indices
        metric_vals: Array of metric values (e.g., accuracy)
        diversity_threshold: Threshold for considering prompts too easy/hard
        
    Returns:
        Dict with keys: 'keep', 'too_correct', 'too_incorrect', 'exclude_too_easy'
    """
    # Group by prompt indices
    unique_indices = np.unique(prompt_indices)
    prompt_metrics = {}
    
    for i in unique_indices:
        mask = prompt_indices == i
        metrics = metric_vals[mask]
        prompt_metrics[i] = {
            'mean': np.mean(metrics),
            'std': np.std(metrics),
            'count': len(metrics)
        }
    
    # Categorize prompts
    keep_prompts = []
    too_correct = []
    too_incorrect = []
    exclude_too_easy = []

    for i, stats in prompt_metrics.items():
        mean_metric = stats['mean']

        if mean_metric > (1.0 - diversity_threshold_upper):
            too_correct.append(i)
        elif mean_metric < diversity_threshold_lower:
            too_incorrect.append(i)
        else:
            # diversity_threshold_lower <= correct_ratio <= 1 - diversity_threshold_upper
            keep_prompts.append(i)
        
        if exclude_threshold_upper is not None and mean_metric >= exclude_threshold_upper:
            exclude_too_easy.append(i)
    
    assert len(keep_prompts) + len(too_correct) + len(too_incorrect) == len(unique_indices)
    
    return {
        'keep': keep_prompts,
        'too_correct': too_correct,
        'too_incorrect': too_incorrect,
        'exclude_too_easy': exclude_too_easy
    }


def extract_original_prompts(batch: DataProto, non_tensor_batch_keys: list) -> DataProto:
    assert "input_ids_original" in batch.non_tensor_batch.keys()
    assert "attention_mask_original" in batch.non_tensor_batch.keys()
    assert "position_ids_original" in batch.non_tensor_batch.keys()

    for key in non_tensor_batch_keys:
        assert key in batch.non_tensor_batch.keys(), f"{key} not in batch.non_tensor_batch.keys()"

    input_ids = torch.from_numpy(batch.non_tensor_batch.pop("input_ids_original"))
    attention_mask = torch.from_numpy(batch.non_tensor_batch.pop("attention_mask_original"))
    position_ids = torch.from_numpy(batch.non_tensor_batch.pop("position_ids_original"))
    assert input_ids.shape == attention_mask.shape == position_ids.shape

    new_batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=input_ids.size(0),
    )

    new_non_tensor_batch = {key: batch.non_tensor_batch[key] for key in non_tensor_batch_keys}
    new_batch = DataProto(batch=new_batch, non_tensor_batch=new_non_tensor_batch)

    # remove duplicates
    # new_batch_reduced = new_batch.reduce(["input_ids"])
    new_batch_reduced = new_batch.reduce(["uid"])

    assert len(set(new_batch_reduced.non_tensor_batch['uid'])) == len(new_batch_reduced), "uid after reduce should be unique"
    assert len(set(new_batch.non_tensor_batch['uid'])) == len(set(new_batch_reduced.non_tensor_batch['uid'])), "uid after before and after reduce should be identical"

    return new_batch_reduced
