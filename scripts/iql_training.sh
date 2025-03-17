#!/bin/bash
#SBATCH --job-name=logiql_training
#SBATCH --output=slurm/log%A_%a_iql_training.out
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=72:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-med

source /home/quanpham/first_3.9.6/bin/activate

# Define parameters.
DATA_TYPE="synthetic"
SYNTHETIC_RUN_ID="comp_diff_37"
# MODE=""  # e.g., train or test
ROBOT="IIWA"
OBJ="Hollowbox"
OBST="ObjectDoor"
SUBTASK="Trashcan"
BASE_AGENT_DATA_PATH="/mnt/kostas-graid/datasets/quanpham/full"
BASE_SYNTHETIC_DATA_PATH="/home/quanpham/compositional-rl-synth-data/results/diffusion"
#BASE_SYNTHETIC_DATA_PATH="/mnt/kostas-graid/datasets/spatank/results/diffusion"
BASE_RESULTS_FOLDER="/home/quanpham/compositional-rl-synth-data/results/RL_Run"
DEVICE="cuda"
SEED=0
MAX_TIMESTEPS=50000
N_EPISODES=10
BATCH_SIZE=1024

# Run the training script with the parameters.
python /home/quanpham/compositional-rl-synth-data/scripts/iql_training.py \
    --data_type ${DATA_TYPE} \
    --synthetic_run_id ${SYNTHETIC_RUN_ID} \
    --robot ${ROBOT} \
    --obj ${OBJ} \
    --obst ${OBST} \
    --subtask ${SUBTASK} \
    --base_agent_data_path ${BASE_AGENT_DATA_PATH} \
    --base_synthetic_data_path ${BASE_SYNTHETIC_DATA_PATH} \
    --base_results_folder ${BASE_RESULTS_FOLDER} \
    --device ${DEVICE} \
    --seed ${SEED} \
    --max_timesteps ${MAX_TIMESTEPS} \
    --n_episodes ${N_EPISODES} \
    --batch_size ${BATCH_SIZE}
