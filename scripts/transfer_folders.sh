#!/bin/bash

remote_host="quanpham@grasp-login2.seas.upenn.edu"
base_remote_folder="/mnt/kostas-graid/datasets/quanpham/results/diffusion/cond_diff_24"
local_destination="/Users/anhquanpham/Developer/compositional-rl-synth-data/cluster_results/diffusion/cond_diff_24/test"

mkdir -p "$local_destination"

while IFS= read -r folder || [[ -n "$folder" ]]; do
    remote_folder="$base_remote_folder/$folder"
    folder_name=$(basename "$remote_folder")
    mkdir -p "$local_destination/$folder_name"
    rsync -avz -e ssh "$remote_host:$remote_folder/" "$local_destination/$folder_name/"
done < test_task_folders_to_move.txt

echo "Transfer complete!"
