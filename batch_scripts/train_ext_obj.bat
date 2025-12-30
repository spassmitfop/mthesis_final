@echo off
setlocal enabledelayedexpansion

REM List of Atari games
set games=Pong SpaceInvaders Boxing Freeway Amidar Seaquest

REM Loop through each game and run the command
for %%G in (%games%) do (
    echo Running PPO training for %%G ...
    python cleanrl/ppo_atari_oc.py ^
        --env-id ALE/%%G-v5 ^
        --obs_mode obj ^
        --masked_wrapper ext_obj ^
        --architecture PPO_OBJ ^
        --backend OCAtari ^
        --total_timesteps 10000000 ^
        --num_envs 10 ^
        --seed 6 ^
        --exp_name use_distances2 ^
        --base_dir new_agents ^
        --use_distances
)

echo All runs completed!
pause
