#!/bin/bash

BASE="/workspace/datasets"

DATASETS=(
#    "Circular-bench"
#    "chair"
#    "4logs-earlham-park"
#    "plastic-cart-130-vert-shade"
#    "robot-under-over-exposure"
#    "Rocks2"
#    "robot-recapture"
#    "plastic-cart-130deg"
#    "Log-hicontrast"
#    "log"
      "rocks"
      "split-in-half-tree"
      "small-tree"
      "small-tree-2"
)

for DATA in "${DATASETS[@]}"; do
    echo "Processing: $DATA"
    DIR="$BASE/$DATA"

    # 1. Standard gsplat
    python simple_trainer.py default --data-dir "$DIR" --result-dir "$DIR/results/gsplat"

    # 2. Bilateral Grid
    python simple_trainer.py default --data-dir "$DIR" --result-dir "$DIR/results/bilateral_grid" --post-processing bilateral_grid

    # 3. PPISP
    python simple_trainer.py default --data-dir "$DIR" --result-dir "$DIR/results/ppisp" --post-processing ppisp

    # 4. GS-W (App Opt)
    python simple_trainer.py default --data-dir "$DIR" --result-dir "$DIR/results/gs-w" --app-opt
done