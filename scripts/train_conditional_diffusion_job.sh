#!/bin/bash
#SBATCH --job-name=64diffusion_training
#SBATCH --output=slurm/%A_%a_train_conditional_diffusion.out
#SBATCH --mem=150G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=96:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med
#SBATCH --exclude=ee-3090-0.grasp.maas,ee-3090-1.grasp.maas




source /home/quanpham/first_3.9.6/bin/activate

python /home/quanpham/compositional-rl-synth-data/scripts/train_conditional_diffusion.py \
    --base_data_path /mnt/kostas-graid/datasets/quanpham/full \
    --base_results_folder /home/quanpham/compositional-rl-synth-data/results/diffusion \
    --gin_config_files /home/quanpham/compositional-rl-synth-data/config/diffusion.gin \
    --dataset_type expert \
    --experiment_type default \
    --num_train 64 \
    --seed 42
    
