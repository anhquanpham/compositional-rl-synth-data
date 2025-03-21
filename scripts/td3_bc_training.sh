#!/bin/bash
#SBATCH --job-name=s1td3bc_training
#SBATCH --output=slurm/%A_%a_td3_bc_training.out
#SBATCH --mem=25G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=72:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med


source /home/quanpham/first_3.9.6/bin/activate

# Define parameters.
#2,   1,  14,  13,  3,    4,  11
#56, 98, 108, 120, 140, 182, 224
DATA_TYPE="agent"
SYNTHETIC_RUN_ID="non_comp_diff_11" 
ROBOT="Kinova3"
OBJ="Box"
OBST="GoalWall"
TASK="Shelf"
BASE_AGENT_DATA_PATH="/mnt/kostas-graid/datasets/quanpham/full"
BASE_SYNTHETIC_DATA_PATH="/home/quanpham/compositional-rl-synth-data/results/diffusion"
#BASE_SYNTHETIC_DATA_PATH="/mnt/kostas-graid/datasets/spatank/results/diffusion"
BASE_RESULTS_FOLDER="/home/quanpham/compositional-rl-synth-data/results/RL_Run"
DEVICE="cuda"
SEED=1001

# Run the training script with the parameters.
python /home/quanpham/compositional-rl-synth-data/scripts/td3_bc_training.py \
    --data_type ${DATA_TYPE} \
    --synthetic_run_id ${SYNTHETIC_RUN_ID} \
    --robot ${ROBOT} \
    --obj ${OBJ} \
    --obst ${OBST} \
    --task ${TASK} \
    --base_agent_data_path ${BASE_AGENT_DATA_PATH} \
    --base_synthetic_data_path ${BASE_SYNTHETIC_DATA_PATH} \
    --base_results_folder ${BASE_RESULTS_FOLDER} \
    --device ${DEVICE} \
    --seed ${SEED}