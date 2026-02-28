#!/bin/bash

BASE="/workspace/datasets"

DATASETS=(
    "Circular-bench"
    "chair"
    "4logs-earlham-park"
#    "plastic-cart-130-vert-shade"
    "robot-under-over-exposure"
    "Rocks2"
    "robot-recapture"
    "plastic-cart-130deg"
    "Log-hicontrast"
#    "log"
    "rocks"
    "split-in-half-tree"
    "small-tree"
#    "small-tree-2"
    "log-earlham-park"
)

for DATA in "${DATASETS[@]}"; do
    echo "Processing: $DATA"
    DIR="$BASE/$DATA"

    python simple_trainer.py --data-dir "$DIR" --result-dir "$DIR/results/3d"
done