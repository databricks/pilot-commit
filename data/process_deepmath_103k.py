# Without binary filter
# python data/process_deepmath_103k.py --local_dir /share/data/files/deepmath_103k
# With binary filter
# python data/process_deepmath_103k.py --local_dir /share/data/files/deepmath_103k_filtered --filtered
# With binary filter and subsample
# python data/process_deepmath_103k.py --local_dir /share/data/files/deepmath_103k_filtered_0.1 --filtered --subsample 0.1
import os
import argparse
import random

from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/deepmath_103k')
    parser.add_argument('--num_proc', type=int, default=16)
    parser.add_argument('--filtered', action='store_true', help='whether to use filtered dataset')
    parser.add_argument('--subsample', type=int, default=-1, help='subsample the dataset, int or float')
    parser.add_argument('--seed', type=int, default=42, help='seed for subsampling')
    args = parser.parse_args()

    # load dataset
    if args.filtered:
        dataset_name = "friendshipkim/DeepMath-103K-filtered"
    else:
        dataset_name = "friendshipkim/DeepMath-103K"
    try:
        data = load_dataset(dataset_name)
    except ValueError:
        print(f"Dataset {dataset_name} not found. Run prepare_deepmath_103k.py to preprocess the dataset.")
        exit(1)

    # process train and test data
    def process_fn_train(example, idx, split):
        instruction_following = "\n\nLet's think step by step and output the final answer within \\boxed{}."
        # if example["question"] doesn't end with a punctuation, add a period
        if not example["question"].endswith(('.', '?', '!', ':')):
            example["question"] = example["question"] + "."
        
        question = example["question"] + instruction_following
        data = {
            "data_source": "deepmath",
            "prompt": [
                {'content': '', 'role': 'system'},
                {'content': question, 'role': 'user'}
            ],
            "ability": "math",
            "reward_model": {
                "ground_truth": example["final_answer"],
                "style": "rule",
            },
            "extra_info": {
                'index': idx,  # int
                'original_id': str(example["original_index"]),  # str
                'raw_problem': example["question"],  # str
                'split': split,  # str
                'difficulty': f"{float(example['difficulty'])}/9.0",  # str
                # 'r1': example["r1_solution_1"]
            }
        }
        return data
    train_data = data['train'].map(function=process_fn_train, with_indices=True, num_proc=args.num_proc, remove_columns=data['train'].column_names, fn_kwargs={'split': 'train'})
    test_data = data['test'].map(function=process_fn_train, with_indices=True, num_proc=args.num_proc, remove_columns=data['test'].column_names, fn_kwargs={'split': 'test'})
    
    # Print an example
    print(train_data)
    print(test_data)
    print(train_data[0])

    # subsample train set
    if isinstance(args.subsample, float) and args.subsample < 1.0 and args.subsample > 0.0:
        random.seed(args.seed)
        train_data = train_data.select(random.sample(range(len(train_data)), int(len(train_data) * args.subsample)))
    elif isinstance(args.subsample, int) and args.subsample > 0:
        random.seed(args.seed)
        train_data = train_data.select(random.sample(range(len(train_data)), args.subsample))
    
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # save train and test data
    train_data.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_data.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    print(f"Saved train and test data to {args.local_dir}")