#!/bin/bash

# Assert that all four arguments are provided
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <max_concurrent_scripts> <num_gpus> <start_gpu_id> <static/dynamic>"
    exit 1
fi

max_concurrent=$1
num_gpus=$2
start_gpu_id=$3
mask_type=$4

# If user passes an invalid or zero number of GPUs, quit
if [[ $num_gpus -lt 1 ]]; then
    echo "Error: num_gpus must be >= 1"
    exit 1
fi

# The Python script to execute
python_script="scripts/preprocessing/create_sam_dataset.py"
common_args="--indir data/creste --outdir data/creste/sam2 --camids cam0 --mask_type ${mask_type} --chunk_idx "
# python scripts/preprocessing/create_sam_dataset.py --indir data/creste --outdir data/creste/sam2 --camids cam0 --mask_type dynamic --chunk_idx 0 
# Array of argument sets to pass to the Python script
chunk_ids=(
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
)

# Counter for concurrent scripts
current_count=0

# Our "gpu_id" will increment for each chunk, 
# but we map it into the continuous range:
#   [start_gpu_id, start_gpu_id + num_gpus - 1]
gpu_id="$start_gpu_id"   # we'll increment this for each chunk

# Loop over all argument sets in the list
for chunk_id in "${chunk_ids[@]}"; do

    # The offset from start_gpu_id:
    #   offset = (gpu_id - start_gpu_id)
    # Then mod that offset by num_gpus
    # Add start_gpu_id back in to stay in [start_gpu_id, start_gpu_id+num_gpus-1].
    gpu_index=$(( start_gpu_id + ((gpu_id - start_gpu_id) % num_gpus) ))

    export CUDA_VISIBLE_DEVICES=$gpu_index
    echo "Starting $python_script $common_args $chunk_id on GPU $CUDA_VISIBLE_DEVICES"

    # Execute the Python script in the background with the current set of arguments
    python "$python_script" $common_args "$chunk_id" &

    # Increment the count of currently running scripts
    ((current_count++))
    ((gpu_id++))  # Advance to the next GPU for next chunk

    # Check if we've reached the max number of concurrent scripts
    if [[ $current_count -ge $max_concurrent ]]; then
        # Wait for all background jobs to finish
        wait
        echo "All scripts in the current batch have completed."
        # Reset the count
        current_count=0
    fi
done

# Wait for the last batch of scripts, if any are still running
wait
echo "All scripts have completed."
