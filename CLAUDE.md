# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conda Environment
- Environment Name: `dsrl_pi0`
- Python Path: `/data/pgq/miniconda3/envs/dsrl_pi0/bin/python`
- Python Version: 3.11.11
- Usage: `/data/pgq/miniconda3/envs/dsrl_pi0/bin/python <script>` or activate with `conda activate dsrl_pi0`

## Overview

This repository implements DSRL (Diffusion Steering via Reinforcement Learning) for steering the pre-trained π₀ policy across various robotic manipulation environments. The implementation is JAX-based and builds upon jaxrl2 and PTR architectures.

**Key Innovation**: Uses latent space reinforcement learning to "steer" a pre-trained generalist diffusion policy (π₀) for specific tasks, rather than training from scratch.

## Training Commands

### Simulation Training

Libero environment:
```bash
bash examples/scripts/run_libero.sh
```

Aloha environment:
```bash
bash examples/scripts/run_aloha.sh
```

### Real Robot Training

Franka robot with DROID:
```bash
# On remote server (for pi0 hosting):
cd openpi && python scripts/serve_policy.py --env=DROID

# On robot client:
bash examples/scripts/run_real.sh
```

**Note**: Real robot training requires setting environment variables in the script:
- `LEFT_CAMERA_ID`, `RIGHT_CAMERA_ID`, `WRIST_CAMERA_ID`: Camera IDs
- `remote_host`, `remote_port`: π₀ remote hosting configuration

## Project Structure

### Core Components

**`jaxrl2/`** - Main RL framework
- `agents/pixel_sac/`: Pixel-based SAC algorithm implementation
  - `pixel_sac_learner.py`: Main learner class with actor, critic, and temperature updates
  - `actor_updater.py`: Actor network update logic
  - `critic_updater.py`: Critic ensemble update logic
  - `temperature_updater.py`: Automatic entropy temperature tuning
- `networks/`: Neural network architectures
  - `encoders/`: Various CNN encoders (ResNet, IMPALA, spatial softmax)
  - `learned_std_normal_policy.py`: Policy network with learned std
  - `values.py`: State-action value functions (Q-functions)
- `data/`: Replay buffer and dataset handling
- `utils/`: WandB logging, visualization, experiment management

**`examples/`** - Training entry points and utilities
- `launch_train_sim.py`: Simulation training entry point (CLI interface)
- `train_sim.py`: Main simulation training logic
- `train_utils_sim.py`: Simulation-specific training utilities (observation conversion, trajectory collection)
- `launch_train_real.py`: Real robot training entry point
- `train_real.py`: Real robot training logic (uses DROID environment)
- `train_utils_real.py`: Real robot training utilities

### Submodules

**`openpi/`** - π₀ policy implementation (submodule)
- Pre-trained diffusion policy for robotic manipulation
- Used as the base policy that DSRL steers
- Supports remote hosting for real robot deployment

**`LIBERO/`** - Libero simulation environment (submodule)
- Benchmark suite for robotic manipulation tasks

## Architecture

### DSRL Algorithm Flow

1. **π₀ Policy Query**: The pre-trained π₀ diffusion policy is queried to get a base action
2. **Steering Action**: DSRL learns a residual/steering action in the latent space
3. **Combined Action**: The final action is π₀ action + steering action (scaled by `action_magnitude`)
4. **Training**: Uses pixel-based SAC with:
   - Actor: Outputs steering actions conditioned on visual observations
   - Critic: Ensemble Q-functions with pixel encoders
   - Temperature: Auto-tuned for entropy regularization

### Key Hyperparameters

- `query_freq`: How often to query π₀ (every N steps)
- `action_magnitude`: Scaling factor for steering actions
- `multi_grad_step`: Number of gradient steps per environment step (UTD ratio)
- `latent_dim`: Dimension of the latent space (default: 50 for π₀ noise)

### Observation Processing

Observations are converted differently for π₀ vs DSRL:

- **For π₀**: Full-resolution images (224x224) with language prompt
- **For DSRL**: Resized images (64x128 depending on env) as input to actor/critic

See `train_utils_sim.py`:
- `obs_to_pi_zero_input()`: Converts env obs to π₀ input format
- `obs_to_img()`: Converts env obs to DSRL pixel observations
- `obs_to_qpos()`: Extracts proprioceptive state

## Environment Variables

- `EXP`: Experiment output directory (default: `./logs/`)
- `OPENPI_DATA_HOME`: OpenPI data directory
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `XLA_PYTHON_CLIENT_PREALLOCATE`: JAX memory allocation (set to `false`)
- `DISPLAY`, `MUJOCO_GL`, `MUJOCO_EGL_DEVICE_ID`: Rendering settings for sim

## Common Issues

- **JAX compilation**: First run is slow due to JIT compilation. Cache stored in `~/jax_compilation_cache/`
- **Multi-GPU**: Batch size must be divisible by number of devices
- **π₀ downloading**: Checkpoints are auto-downloaded from S3 on first run
- **Libero rendering**: Uses EGL for headless rendering; ensure proper GPU drivers
