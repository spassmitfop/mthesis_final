#!/usr/bin/env bash

# Base directories
AGENTS_DIR="shared_j/agents"
OUT_DIR="shared_j/object_focus_results2"

# List of Atari games, methods, wrappers
games="pong seaquest spaceinvaders amidar freeway boxing"
methods="Saliency GuidedBackprop"
wrappers="obs_mode_dqn masked_dqn_bin masked_dqn_planes parallelplanes"
exts="0 10"
aggregations="mean max"

# Loop through each game
for G in $games; do
    for W in $wrappers; do
        for M in $methods; do
            for E in $exts; do
                for A in $aggregations; do
                    echo "Running saliency comparison for $G $W $M ext=$E am=$A ..."

                    python python_scripts/object_focus_comparison.py \
                        -g "$G" \
                        -a "$AGENTS_DIR/$G/1/$W.cleanrl_model" \
                           "$AGENTS_DIR/$G/2/$W.cleanrl_model" \
                           "$AGENTS_DIR/$G/3/$W.cleanrl_model" \
                        -ca "$M" \
                        -mw "$W" \
                        -e 3 \
                        -n "$W" \
                        -ext "$E" \
                        -am "$A" \
                        -out "$OUT_DIR/ext$E/${W}_${M}_${A}.json"
                done
            done
        done
    done
done

echo "All tasks completed!"
read -p "Press Enter to exit..."
