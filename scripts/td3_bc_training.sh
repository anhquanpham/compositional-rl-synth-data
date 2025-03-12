#!/bin/bash
#SBATCH --job-name=td3_bc_training
#SBATCH --output=slurm/%A_%a_td3_bc_training.out
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=72:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med


source /home/quanpham/first_3.9.6/bin/activate

# Define parameters.
DATA_TYPE="synthetic"
SYNTHETIC_RUN_ID="comp_diff_23"
ROBOT="Kinova3"
OBJ="Dumbbell"
OBST="ObjectWall"
TASK="Push"
BASE_AGENT_DATA_PATH="/mnt/kostas-graid/datasets/quanpham/full"
BASE_SYNTHETIC_DATA_PATH="/home/quanpham/compositional-rl-synth-data/results/diffusion"
#BASE_SYNTHETIC_DATA_PATH="/mnt/kostas-graid/datasets/spatank/results/diffusion"
BASE_RESULTS_FOLDER="/home/quanpham/compositional-rl-synth-data/results/RL_Run"
DEVICE="cuda"
SEED=23

# Run the training script with the parameters.
python /home/quanpham/compositional-rl-synth-data/scripts/td3_bc.py \
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