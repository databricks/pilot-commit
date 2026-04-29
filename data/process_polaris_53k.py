# python polaris_53k.py --local_dir /share/data/files/polaris_53k
import os
import argparse
import random

from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/polaris_53k')
    parser.add_argument('--num_proc', type=int, default=16)
    args = parser.parse_args()

    # load dataset
    dataset_name = "POLARIS-Project/Polaris-Dataset-53K"
    data = load_dataset(dataset_name)

    # Counter for unique numeric indices
    global_counter = 0

    # process train and test data
    def process_fn_train(example, idx):
        instruction_following = "\n\nLet's think step by step and output the final answer within \\boxed{}."
        
        raw_problem = example["problem"]
        question = raw_problem + instruction_following
        data = {
            "data_source": "polaris_53k",
            "prompt": [
                {'content': '', 'role': 'system'},
                {'content': question, 'role': 'user'}
            ],
            "ability": "math",
            "reward_model": {
                "ground_truth": example["answer"],
                "style": "rule",
            },
            "extra_info": {
                'index': idx,
                'original_id': "",
                'raw_problem': raw_problem,
                'split': "train",
                'difficulty': example["difficulty"],  # already a string
            }
        }
        return data
    train_data = data['train'].map(function=process_fn_train, num_proc=args.num_proc, with_indices=True, remove_columns=data['train'].column_names)
    
    # Print an example
    print(train_data)
    print(train_data[0])

    print(f"Train set size: {len(train_data)}")
    
    # save train and test data
    train_data.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    print(f"Saved train and test data to {args.local_dir}")