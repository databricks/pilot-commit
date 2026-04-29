# python convert_data_format.py --reformat_prompt --instruction_location user --new_file_suffix nosystem-boxed


import pandas as pd
import numpy as np
import os
import argparse
from datasets import Dataset
from prompts import OLD_PROMPT_PREFIX, OLD_PROMPT_SUFFIX, NEW_PROMPT_SUFFIX, DEFAULT_SYSTEM_PROMPT


def load_data(file_path):
    df = pd.read_parquet(file_path)
    dataset = Dataset.from_pandas(df)
    return dataset

def convert_data_format(example, idx, reformat_prompt=False, instruction_location="user", data_source="math_dapo"):
    if reformat_prompt:
        if 'raw_problem' in example['extra_info']:
            raw_problem = example['extra_info']['raw_problem'].strip()
        else:
            raw_problem = example['prompt'][0]['content'].replace(OLD_PROMPT_PREFIX, '')
            raw_problem = raw_problem.replace(OLD_PROMPT_SUFFIX, '').strip()
        
        if instruction_location == "user":
            new_prompt = raw_problem + NEW_PROMPT_SUFFIX
    else:
        new_prompt = example['prompt'][0]['content']
    user_prompt = {
        'content': new_prompt,
        'role': 'user',
    }

    if instruction_location == "user":
        system_prompt = {
            'content': '',
            'role': 'system',
        }
    else:
        system_prompt = {
            'content': DEFAULT_SYSTEM_PROMPT,
            'role': 'system',
        }

    data = {
        'data_source': data_source,
        'prompt': [system_prompt, user_prompt],
        'ability': 'MATH',
        'reward_model': {
            'ground_truth': example['reward_model']['ground_truth'],
            'style': 'rule',
        },
        'extra_info': {
            'index': idx,
            'original_id': str(example['extra_info']['index']),
            'raw_problem': raw_problem,
            'split': None,
            'difficulty': "",  # not available
        }
    }
    return data

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="/share/data/files/dapo_formatted")
    args.add_argument("--reformat_prompt", action="store_true")
    args.add_argument("--instruction_location", type=str, default="user")

    args.add_argument("--num_proc", type=int, default=16)
    args.add_argument("--new_file_suffix", type=str, default="nosystem")
    args = args.parse_args()
    
    data_dir = args.data_dir
    dapo_dataset = load_data(os.path.join(data_dir, "dapo-math-17k.parquet"))

    # remove duplicates by extra_info['index'] efficiently
    indices = [ex['index'] for ex in dapo_dataset['extra_info']]
    _, unique_row_indices = np.unique(indices, return_index=True)
    unique_row_indices = sorted(unique_row_indices)  # preserve original order
    dapo_dataset = dapo_dataset.select(unique_row_indices)
    print(f"Deduped dapo dataset size: {len(dapo_dataset)}")

    # map dapo dataset to the new data format
    dapo_dataset = dapo_dataset.map(
        convert_data_format,
        num_proc=args.num_proc,
        with_indices=True,
        fn_kwargs={
            "reformat_prompt": args.reformat_prompt,
            "instruction_location": args.instruction_location,
            "data_source": "math_dapo",
        }
    )
    print(dapo_dataset[0])
    print()

    # save to parquet with new file suffix
    file_name = f'dapo-math-17k-{args.new_file_suffix}-dedup.parquet'
    print(f"Saving to {os.path.join(data_dir, file_name)}")
    dapo_dataset.to_parquet(os.path.join(data_dir, file_name))