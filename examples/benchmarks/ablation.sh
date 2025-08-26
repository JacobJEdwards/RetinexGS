SCENE_DIR="/workspace/360_v2"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
POSTFIXES="_variance _multiexposure" # todo:  repo or continue some contrast that didnt finish
CONFIG_OPTIONS=(
    "--no-appearance-embeddings"
    "--no-use-view-dirs"
    "--no-use-normals"
    "--no-use-camera-response-network"
    "--lambda_illum_smoothness 0.0"
    "--lambda_exclusion 0.0"
    "--lambda_shn_reg 0.0"
    "--lambda_tv_loss 0.0"
)
CONFIG_NAMES=(
    "no_appearance"
    "no_view_dirs"
    "no_normals"
    "no_crf"
    "no_illum_smoothness"
    "no_exclusion"
    "no_shn_reg"
    "no_tv_loss"
)

for POSTFIX in $POSTFIXES; do
    RESULT_DIR_BASE="/workspace/3d/ablation/$POSTFIX"

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