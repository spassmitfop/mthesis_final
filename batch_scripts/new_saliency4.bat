REM List of Atari games, methods, wrappers
set games=pong seaquest spaceinvaders amidar freeway boxing
set methods=Saliency GuidedBackprop
set wrappers=obs_mode_dqn masked_dqn_bin masked_dqn_planes parallelplanes
set exts=0 10 20
set aggregations=max

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

                    python cleanrl/saliency_comparison6.py ^
                        -g %%G ^
                        -a saliency_agents\rename_test\%%G\1\%%W.cleanrl_model saliency_agents\rename_test\%%G\2\%%W.cleanrl_model saliency_agents\rename_test\%%G\3\%%W.cleanrl_model ^
                        -ca %%M ^
                        -mw %%W ^
                        -e 3 ^
                        -n %%W ^
                        -ext %%E ^
                        -am %%A ^
                        -out copy2_saliency_jsons_ext%%E/%%W_%%M_%%A.json
                )
            )
        )
    )
)

echo All tasks completed!
pause
