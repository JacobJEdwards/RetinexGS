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
      "rocks"
      "split-in-half-tree"
      "small-tree"
      "small-tree-2"
      "log-earlham-park"
)

TRAIN_IMAGE_DIRS = (
      "retinexformer_FiveK"
      "retinexformer_MST_Plus_Plus_4x1800"
      "retinexformer_SDSD_outdoor"
      "retinexformer_LOL_v1"
      "retinexformer_MST_Plus_Plus_8x1150"
      "retinexformer_SID"
      "retinexformer_LOL_v2_real"
      "retinexformer_NTIRE"
      "retinexformer_SMID"
      "retinexformer_LOL_v2_synthetic"
      "retinexformer_SDSD_indoor"
)

for DATA in "${DATASETS[@]}"; do
  for TRAIN_IMAGE_DIR in "${TRAIN_IMAGE_DIRS[@]}"; do
    echo "Processing: $DATA with $TRAIN_IMAGE_DIR"

    DIR="$BASE/$DATA"
    TRAIN_DIR = "$DIR/retinex/$TRAIN_IMAGE_DIR"
    TEST_DIR = "$DIR/images"

    # 1. Standard gsplat
    python simple_trainer.py default --data-dir "$DIR" --result-dir "$DIR/results/retinexformer/$TRAIN_IMAGE_DIR" --train-image-dir "$TRAIN_DIR" --test-image-dir "$TEST_DIR"
done