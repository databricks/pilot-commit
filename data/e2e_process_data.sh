export DATA_DIR=${DATA_DIR:-/root/data}

# download dapo_17k dataset
echo "Downloading dapo_17k dataset..."
bash data/prepare_dapo_data.sh

# preprocess dapo_17k dataset
# will save dapo-math-17k-nosystem-boxed-dedup.parquet to DATA_DIR/dapo_formatted
echo "Preprocessing dapo_17k dataset..."
python data/process_dapo_17k.py \
    --data_dir $DATA_DIR/dapo_formatted \
    --reformat_prompt \
    --instruction_location user \
    --new_file_suffix nosystem-boxed

# process filtered deepmath
echo "Processing filtered deepmath 103k..."
python data/process_deepmath_103k.py \
    --local_dir $DATA_DIR/deepmath_103k_filtered \
    --filtered

# process polaris 53k
echo "Processing polaris 53k..."
python data/process_polaris_53k.py \
    --local_dir $DATA_DIR/polaris_53k

# process aime 2025
# will save aime-2025-nosystem-boxed.parquet to DATA_DIR
echo "Processing aime 2025..."
python data/process_aime.py \
    --year 2025 \
    --output_dir $DATA_DIR/math_test \
    --output_name aime-2025-nosystem-boxed.parquet \
    --n_trials 32

# process aime 2024
echo "Processing aime 2024..."
python data/process_aime.py \
    --year 2024 \
    --output_dir $DATA_DIR/math_test \
    --output_name aime-2024-nosystem-boxed.parquet \
    --n_trials 32

# process amc 2023
echo "Processing amc 2023..."
python data/process_amc.py \
    --year 2023 \
    --output_dir $DATA_DIR/math_test \
    --output_name amc-2023-nosystem-boxed.parquet \
    --n_trials 32

# process math 500
echo "Processing math 500..."
python data/process_math.py \
    --output_dir $DATA_DIR/math_test \
    --output_name math-500-nosystem-boxed.parquet \
    --n_trials 1

# process oylmpiad bench
echo "Processing oylmpiad bench..."
python data/process_oylmpiad_bench.py \
    --output_dir $DATA_DIR/math_test \
    --output_name oylmpiad-bench-nosystem-boxed.parquet \
    --n_trials 1

# process minerva math
echo "Processing minerva math..."
python data/process_minerva_math.py \
    --output_dir $DATA_DIR/math_test \
    --output_name minerva-math-nosystem-boxed.parquet \
    --n_trials 1

# # for debugging
# echo "Processing filtered deepmath with subsample..."
# python data/process_deepmath_103k.py \
#     --local_dir $DATA_DIR/deepmath_103k_filtered_debug \
#     --filtered \
#     --subsample 600

echo "Done!"