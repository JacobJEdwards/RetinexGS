SCENE_DIR="/workspace/360_v2"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
POSTFIXES="_variance _multiexposure _contrast"
CONFIG_OPTIONS=(
    "--no-use-normals"
    "--no-use-view-dirs"
)
CONFIG_NAMES=(
    "no_normals"
    "no_view_dirs"
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