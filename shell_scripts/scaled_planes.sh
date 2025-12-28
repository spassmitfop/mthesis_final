#!/bin/bash

# Define your games and seeds
GAMES=("ALE/Amidar-v5" "ALE/Freeway-v5" "ALE/Boxing-v5")  # Add more games as needed
# "ALE/Seaquest-v5" "ALE/Pong-v5" "ALE/SpaceInvaders-v5"
SEEDS=($1)  # Add more seeds as needed

# Common parameters
OBS_MODE="obj"
ARCHITECTURE="PPO"
BACKEND="OCAtari"
TOTAL_TIMESTEPS=10000000
NUM_ENVS=10
WANDB_ENTITY="jan-landgrafe-tu-darmstadt"
WANDB_PROJECT_NAME="mthesis"
MASKED_WRAPPER="masked_dqn_planes_scaled"
EXP_NAME="planes_scaled_kr_0point87"
BASE_DIR="shared_scaled"
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
      --masked_wrapper "$MASKED_WRAPPER" \
      --seed "$SEED" \
      --exp_name "$EXP_NAME" \
      --base_dir "$BASE_DIR" \
      --keep_ratio \
      --scale_w 0.87 \
      --scale_h 0.87 \
      $TRACK
  done
done
