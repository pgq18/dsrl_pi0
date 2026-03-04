#!/bin/bash
# DSRL Training with Walloss Policy on Libero Environment
#
# This script runs DSRL training using Walloss model instead of π0.
# Make sure to set the correct WALLOSS_MODEL_PATH before running.

proj_name=DSRL_Walloss_Libero
device_id=6

# Environment setup
export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Get the project root directory (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Add project root and LIBERO to PYTHONPATH
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/LIBERO:$PYTHONPATH

# Change to project root directory
cd ${PROJECT_ROOT}

# Train config path (depends on PROJECT_ROOT)
WALLOSS_TRAIN_CONFIG_PATH=${WALLOSS_TRAIN_CONFIG_PATH:-"${PROJECT_ROOT}/wall-x/workspace/libero/config_qact.yml"}
# Walloss model paths - UPDATE THESE TO YOUR MODEL LOCATION
WALLOSS_MODEL_PATH=${WALLOSS_MODEL_PATH:-"${PROJECT_ROOT}/wall-x/workspace/libero/finetuned"}
WALLOSS_PROCESSOR_PATH=${WALLOSS_PROCESSOR_PATH:-"${PROJECT_ROOT}/wall-x/workspace/libero/finetuned"}
WALLOSS_NORM_STATS_PATH=${WALLOSS_NORM_STATS_PATH:-"${PROJECT_ROOT}/wall-x/workspace/libero/lerobot/libero_goal_image/norm_stats.json"}

# Ensure mujoco is installed
pip install mujoco==3.3.1

# Create log directory and set log file
LOG_DIR=${PROJECT_ROOT}/logs/${proj_name}
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).txt
echo "Logging to: ${LOG_FILE}"

# Run DSRL training with Walloss policy
xvfb-run -a -s "-screen 0 1280x720x24" \
    python3 -m examples.launch_train_sim \
        --algorithm pixel_sac \
        --env libero \
        --task_id 7 \
        --policy_type walloss \
        --walloss_model_path ${WALLOSS_MODEL_PATH} \
        --walloss_processor_path ${WALLOSS_PROCESSOR_PATH} \
        --walloss_norm_stats_path ${WALLOSS_NORM_STATS_PATH} \
        --walloss_train_config_path ${WALLOSS_TRAIN_CONFIG_PATH} \
        --walloss_predict_mode diffusion \
        --action_dim 7 \
        --agent_pos_dim 8 \
        --prefix dsrl_walloss_libero \
        --wandb_project ${proj_name} \
        --batch_size 256 \
        --discount 0.999 \
        --seed 0 \
        --max_steps 500000  \
        --eval_interval 1000 \
        --checkpoint_interval 3000 \
        --log_interval 500 \
        --eval_episodes 10 \
        --multi_grad_step 20 \
        --start_online_updates 500 \
        --resize_image 256 \
        --action_magnitude 1.0 \
        --query_freq 32 \
        --hidden_dims 256 \
    2>&1 | tee ${LOG_FILE}
