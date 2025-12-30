
set games=pong seaquest spaceinvaders amidar freeway boxing
set methods=Saliency GuidedBackprop
set wrappers=obs_mode_dqn masked_dqn_bin masked_dqn_planes parallelplanes
set exts=0 10
set aggregations=mean
REM set aggregations=mean max

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

                    python python_scripts/object_focus_comparison.py ^
                        -g %%G ^
                        -a agents\%%G\1\%%W.cleanrl_model agents\%%G\2\%%W.cleanrl_model agents\%%G\3\%%W.cleanrl_model ^
                        -ca %%M ^
                        -mw %%W ^
                        -e 3 ^
                        -n %%W ^
                        -ext %%E ^
                        -am %%A ^
                        -out object_focus_results2/ext%%E/%%W_%%M_%%A.json
                )
            )
        )
    )
)

echo All tasks completed!
pause
