SCENE_DIR="/workspace/360_v2"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
POSTFIXES="variance multiexposure contrast"
RESULT_DIR="/workspace/2d"

CONFIG_OPTIONS=(
  "--no-loss_adaptive_curve"
  "--no-loss_smooth_edge_aware"
  "--no-loss_white_preservation"
  "--allow_chromatic_illumination"
  "--uncertainty_weighting"
  "--learnt_weighting"
)

CONFIG_NAMES=(
  "no_loss_adaptive_curve"
  "no_loss_smooth_edge_aware"
  "no_loss_white_preservation"
  "coloured_illumination"
  "uncertainty_weighting"
  "learnt_weighting"
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
            echo "Running $SCENE with ${CONFIG_NAMES[$i]} on $POSTFIX"
            RESULT_DIR="$RESULT_DIR_BASE/${CONFIG_NAMES[$i]}"
            CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
                --data_dir $SCENE_DIR/"$SCENE"/ \
                --postfix $POSTFIX \
                --result_dir $RESULT_DIR/"$SCENE"/ \
                --no-tb_save_image \
                --tb_every 1000 \
                ${CONFIG_OPTIONS[$i]}
        done
    done
done