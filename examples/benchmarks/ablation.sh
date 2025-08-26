SCENE_DIR="/workspace/360_v2"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
POSTFIXES="_variance _multiexposure _contrast"
CONFIG_OPTIONS=(
  "--no-loss_adaptive_curve"
  "--no-loss_exposure"
  "--no-loss_reflectance_spa"
  "--no-loss_smooth_edge_aware"
  "--no-loss_exposure_local"
  "--no-loss_exclusion"
  "--no-loss_white_preservation"
  "--no-loss_histogram"
  "--no-loss_perceptual_color"
)

CONFIG_NAMES=(
  "no_loss_adaptive_curve"
  "no_loss_exposure"
  "no_loss_reflectance_spa"
  "no_loss_smooth_edge_aware"
  "no_loss_exposure_local"
  "no_loss_exclusion"
  "no_loss_white_preservation"
  "no_loss_histogram"
  "no_loss_perceptual_color"
)

for POSTFIX in $POSTFIXES; do
    RESULT_DIR_BASE="/workspace/2d/ablation/$POSTFIX"

    for SCENE in $SCENE_LIST;
    do
        if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
            DATA_FACTOR=2
        else
            DATA_FACTOR=4
        fi

        for i in "${!CONFIG_OPTIONS[@]}"; do
            echo "Running $SCENE with ${CONFIG_NAMES[$i]}"
            RESULT_DIR="$RESULT_DIR_BASE/${CONFIG_NAMES[$i]}"
            CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
                --data_dir $SCENE_DIR/"$SCENE"/ \
                --postfix $POSTFIX \
                --result_dir $RESULT_DIR/"$SCENE"/ \
                ${CONFIG_OPTIONS[$i]}
        done
    done
done