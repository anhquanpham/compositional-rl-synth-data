#!/bin/bash
#SBATCH --job-name=orgt182diffusion_training
#SBATCH --output=Oslurm/%A_%a_train_conditional_diffusion.out
#SBATCH --mem=400G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=168:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med

source /home/quanpham/first_3.9.6/bin/activate

python /home/quanpham/compositional-rl-synth-data/scripts/train_conditional_diffusion.py \
    --base_data_path /mnt/kostas-graid/datasets/quanpham/full \
    --base_results_folder /home/quanpham/compositional-rl-synth-data/results/diffusion \
    --gin_config_files /home/quanpham/compositional-rl-synth-data/config/diffusion.gin \
    --compositional False \
    --dataset_type expert \
    --experiment_type default \
    --num_train 182 \
    --seed 42
    
