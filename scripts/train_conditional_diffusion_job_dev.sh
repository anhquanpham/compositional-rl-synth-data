#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=slurm/%A_%a_train_conditional_diffusion.out
#SBATCH --mem=400G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=96:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med

source /home/quanpham/first_3.9.6/bin/activate

python /home/quanpham/compositional-rl-synth-data/scripts/train_conditional_diffusion.py \
    --base_data_path /mnt/kostas-graid/datasets/quanpham/dev \
    --base_results_folder /mnt/kostas-graid/datasets/quanpham/dev/results/diffusion \
    --gin_config_files /home/quanpham/compositional-rl-synth-data/config/diffusion.gin \
    --dataset_type expert \
    --experiment_type smallscale \
    --element IIWA \
    --num_train 2 \
    --seed 42
    
