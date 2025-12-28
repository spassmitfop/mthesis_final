@echo off
REM Save this as run_games.bat

REM Define your games
set GAMES=ALE/Boxing-v5 ALE/Seaquest-v5 ALE/Pong-v5 
REM set GAMES=ALE/Amidar-v5 ALE/Freeway-v5 ALE/SpaceInvaders-v5

REM Get seeds from command line arguments
set SEEDS=%*

REM Common parameters
set OBS_MODE=obj
set ARCHITECTURE=PPO
set BACKEND=OCAtari
set TOTAL_TIMESTEPS=10000000
set NUM_ENVS=10
set WANDB_ENTITY=jan-landgrafe-tu-darmstadt
set WANDB_PROJECT_NAME=mthesis
set MASKED_WRAPPER=masked_dqn_planes_scaled
set EXP_NAME=planes_scaled_1point2
set BASE_DIR=shared_scaled
set TRACK=--track

REM Loop over games
for %%G in (%GAMES%) do (
    REM Loop over seeds
    for %%S in (%SEEDS%) do (
        echo Running %%G with seed %%S...
        python cleanrl/ppo_atari_oc_rename2.py ^
            --env-id %%G ^
            --obs_mode %OBS_MODE% ^
            --architecture %ARCHITECTURE% ^
            --backend %BACKEND% ^
            --total_timesteps %TOTAL_TIMESTEPS% ^
            --num_envs %NUM_ENVS% ^
            --wandb_entity %WANDB_ENTITY% ^
            --wandb_project_name %WANDB_PROJECT_NAME% ^
            --masked_wrapper %MASKED_WRAPPER% ^
            --seed %%S ^
            --exp_name %EXP_NAME% ^
            --base_dir %BASE_DIR% ^
            --scale_w 1.2 ^
            --scale_h 1.2 ^
            %TRACK%
    )
)
