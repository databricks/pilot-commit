# Usage: 
# python aime_dataset.py --year 2025
"""
Preprocess the AIME dataset to parquet format
"""

import argparse
import os
import datasets
from prompts import OLD_PROMPT_PREFIX, OLD_PROMPT_SUFFIX, NEW_PROMPT_SUFFIX, DEFAULT_SYSTEM_PROMPT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/share/data/files/dapo_formatted")
    parser.add_argument("--output_name", type=str, default="aime-2025-nosystem-boxed.parquet")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--n_trials", type=int, default=32)
    parser.add_argument("--year", type=int, required=True, choices=[2022, 2023, 2024, 2025], 
                       help="AIME year to process")

    args = parser.parse_args()

    # Configure data source and split based on year
    if args.year in [2022, 2023, 2024]:
        data_source = "zwhe99/aime90"
        split_name = str(args.year)
        subset = None
    elif args.year == 2025:
        data_source = "yentinglin/aime_2025"
        split_name = "train"
        subset = "default"
    else:
        raise ValueError(f"Unsupported year: {args.year}")

    print(f"Loading AIME {args.year} dataset from {data_source}...", flush=True)
    
    if subset:
        dataset = datasets.load_dataset(data_source, subset, trust_remote_code=True)
    else:
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset[split_name]



    def convert_data_format(example, idx):
        raw_problem = example['problem']
        answer = example['answer'] if args.year == 2025 else example['expected_answer']
        original_id = str(example['id'])

        data = {
            'data_source': f'aime_{args.year}',
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
                'index': idx,
                'original_id': original_id,
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
    
