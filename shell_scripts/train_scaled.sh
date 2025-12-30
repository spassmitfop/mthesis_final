#!/bin/bash

# Define your games and seeds
GAMES=("ALE/Seaquest-v5" "ALE/Pong-v5" "ALE/SpaceInvaders-v5" "ALE/Amidar-v5" "ALE/Freeway-v5" "ALE/Boxing-v5")  # 
SEEDS=($1)  # Add more seeds as needed

# Common parameters
OBS_MODE="obj"
ARCHITECTURE="PPO"
BACKEND="OCAtari"
TOTAL_TIMESTEPS=1000000
NUM_ENVS=10
MASKED_WRAPPER="masked_dqn_bin_scaled"
EXP_NAME="bin_scaled_kr_0point87"
BASE_DIR="new_agents"

# Loop over games and seeds
for GAME in "${GAMES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "Running $GAME with seed $SEED..."
    python cleanrl/ppo_atari_oc.py \
      --env-id "$GAME" \
      --obs_mode "$OBS_MODE" \
      --architecture "$ARCHITECTURE" \
      --backend "$BACKEND" \
      --total_timesteps "$TOTAL_TIMESTEPS" \
      --num_envs "$NUM_ENVS" \
      --masked_wrapper "$MASKED_WRAPPER" \
      --seed "$SEED" \
      --exp_name "$EXP_NAME" \
      --base_dir "$BASE_DIR" \
      --keep_ratio \
      --scale_w 0.87 \
      --scale_h 0.87
  done
done
