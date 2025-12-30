@echo off
setlocal enabledelayedexpansion

REM === List of games (separated by spaces) ===
set games=SpaceInvaders-v5 Seaquest-v5 Pong-v5  Amidar-v5 Boxing-v5 Freeway-v5

REM === List of seeds ===
set seeds=1 2 3

REM === Loop over games ===
for %%G in (%games%) do (
    REM Loop over seeds
    for %%S in (%seeds%) do (
        echo Running %%G with seed %%S
        python cleanrl/ppo_atari_oc.py ^
            --env-id ALE/%%G ^
            --obs_mode obj ^
            --architecture PPOCombi2Big ^
            --backend OCAtari ^
            --total_timesteps 10000 ^
            --masked_wrapper masked_dqn_bin_plus_og_obj ^
            --base_dir new_agents ^
            --exp_name bin_plus_obj_no_bnorm ^
            --seed %%S
    )
)

echo All runs completed!
pause
