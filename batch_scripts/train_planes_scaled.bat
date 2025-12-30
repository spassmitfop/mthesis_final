@echo off

REM Define your games
REM set GAMES=ALE/Seaquest-v5 ALE/Pong-v5
set GAMES= ALE/SpaceInvaders-v5 ALE/Amidar-v5 ALE/Freeway-v5 ALE/Boxing-v5

REM Get seeds from command line arguments
set SEEDS=%*

REM Common parameters
set OBS_MODE=obj
set ARCHITECTURE=PPO
set BACKEND=OCAtari
set TOTAL_TIMESTEPS=10000
set NUM_ENVS=10
set MASKED_WRAPPER=masked_dqn_planes_scaled
set EXP_NAME=planes_scaled_kr_0point87
set BASE_DIR=new_agents

REM Loop over games
for %%G in (%GAMES%) do (
    REM Loop over seeds
    for %%S in (%SEEDS%) do (
        echo Running %%G with seed %%S...
        python cleanrl/ppo_atari_oc.py ^
            --env-id %%G ^
            --obs_mode %OBS_MODE% ^
            --architecture %ARCHITECTURE% ^
            --backend %BACKEND% ^
            --total_timesteps %TOTAL_TIMESTEPS% ^
            --num_envs %NUM_ENVS% ^
            --masked_wrapper %MASKED_WRAPPER% ^
            --seed %%S ^
            --exp_name %EXP_NAME% ^
            --base_dir %BASE_DIR% ^
            --keep_ratio ^
            --scale_w 0.87 ^
            --scale_h 0.87
    )
)
