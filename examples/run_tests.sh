#!/bin/bash

BASE="/workspace/datasets"

DATASETS=(
    "Circular-bench"
    "chair"
    "4logs-earlham-park"
    "plastic-cart-130-vert-shade"
    "robot-under-over-exposure"
    "Rocks2"
    "robot-recapture"
    "plastic-cart-130deg"
    "Log-hicontrast"
    "log"
)

for DATA in "${DATASETS[@]}"; do
    echo "Processing: $DATA"
    DIR="$BASE/$DATA"

    python simple_trainer.py default --data-dir "$DIR" --result-dir "$DIR/results/3d"
done