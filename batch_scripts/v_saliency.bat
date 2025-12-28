REM List of Atari games, methods, wrappers
set games=pong seaquest spaceinvaders amidar freeway boxing
set methods=Saliency GuidedBackprop
set wrappers=obs_mode_dqn masked_dqn_bin masked_dqn_planes parallelplanes
set exts=0 10
set aggregations=mean

REM masked_dqn_bin masked_dqn_planes parallelplanes
REM set exts=0 10

REM Loop through each game
for %%G in (%games%) do (
    REM Loop through each wrapper
    for %%W in (%wrappers%) do (
        REM Loop through each saliency method
        for %%M in (%methods%) do (
            REM Loop through each extension value
            for %%E in (%exts%) do (
                REM Loop through each aggregation value
                for %%A in (%aggregations%) do (
                    echo Running saliency comparison for %%G %%W %%M ext=%%E am=%%A ...

                    python cleanrl/ppo_atari_object_focus_comparison.py ^
                        -g %%G ^
                        -a agents\%%G\1\%%W.cleanrl_model agents\%%G\2\%%W.cleanrl_model agents\%%G\3\%%W.cleanrl_model ^
                        -ca %%M ^
                        -mw %%W ^
                        -e 3 ^
                        -n %%W ^
                        -ext %%E ^
                        -am %%A ^
                        -out verify_saliency_jsons_ext%%E/%%W_%%M_%%A.json
                )
            )
        )
    )
)

echo All tasks completed!
pause
