"""
Preprocess the MATH-500 dataset to parquet format
"""

import argparse
import os
import datasets
from prompts import OLD_PROMPT_PREFIX, OLD_PROMPT_SUFFIX, NEW_PROMPT_SUFFIX, DEFAULT_SYSTEM_PROMPT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/share/data/files/math_test")
    parser.add_argument("--output_name", type=str, default="math-500-nosystem-boxed.parquet")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--n_trials", type=int, default=1)

    args = parser.parse_args()

    # Configure data source and split based on year
    data_source = "zwhe99/MATH"
    split_name = "math500"
    subset = "default"

    print(f"Loading MATH-500 dataset from {data_source}...", flush=True)
    
    if subset:
        dataset = datasets.load_dataset(data_source, subset, trust_remote_code=True)
    else:
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset[split_name]

    def convert_data_format(example, idx):
        raw_problem = example['problem']
        answer = example['expected_answer']
        index = idx

        data = {
            'data_source': f'math_500',
            'prompt': [
                {'content': '', 'role': 'system'},
                {'content': raw_problem + NEW_PROMPT_SUFFIX, 'role': 'user'}
            ],
            'ability': 'MATH',
            'reward_model': {
                'ground_truth': answer,
                'style': 'rule',
            },
            'extra_info': {
                'index': index,
                'original_id': "",
                'raw_problem': raw_problem,
                'split': None,
                'difficulty': "",  # not available
            }
        }
        return data

    test_dataset = test_dataset.map(convert_data_format, num_proc=args.num_proc, with_indices=True, remove_columns=test_dataset.column_names)
    print(f"Length of test dataset: {len(test_dataset)}")
    print(test_dataset[0])

    # repeat the dataset n_trials times
    test_dataset = test_dataset.repeat(args.n_trials)
    print(f"Length of test dataset after repeating: {len(test_dataset)}")
    print()

    print(f"Saving to {os.path.join(args.output_dir, args.output_name)}")
    test_dataset.to_parquet(os.path.join(args.output_dir, args.output_name))
    
