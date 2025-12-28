#!/bin/bash

# Define your games and seeds
GAMES=("ALE/Seaquest-v5" "ALE/Pong-v5" "ALE/SpaceInvaders-v5")  # Add more games as needed
# "ALE/Amidar-v5" "ALE/Freeway-v5" "ALE/Boxing-v5"
SEEDS=($1)  # Add more seeds as needed

# Common parameters
OBS_MODE="dqn"
ARCHITECTURE="PPO"
BACKEND="OCAtari"
TOTAL_TIMESTEPS=10000000
NUM_ENVS=10
WANDB_ENTITY="jan-landgrafe-tu-darmstadt"
WANDB_PROJECT_NAME="mthesis"
EXP_NAME="obs_mode_dqn"
BASE_DIR="shared_dqn"
TRACK="--track"

# Loop over games and seeds
for GAME in "${GAMES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "Running $GAME with seed $SEED..."
    python cleanrl/ppo_atari_oc_rename2.py \
      --env-id "$GAME" \
      --obs_mode "$OBS_MODE" \
      --architecture "$ARCHITECTURE" \
      --backend "$BACKEND" \
      --total_timesteps "$TOTAL_TIMESTEPS" \
      --num_envs "$NUM_ENVS" \
      --wandb_entity "$WANDB_ENTITY" \
      --wandb_project_name "$WANDB_PROJECT_NAME" \
      --seed "$SEED" \
      --exp_name "$EXP_NAME" \
      --base_dir "$BASE_DIR" \
      $TRACK
  done
done
