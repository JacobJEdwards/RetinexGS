SCENE_DIR="../../360_v2"
RESULT_DIR="../../results/benchmark"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

RESULT_DIR_BASE="../../results/benchmark"

CONFIG_OPTIONS=(
    "--option1 value1"                  # example config 1
    "--option2 value2 --flag"          # example config 2
    "--option3"                         # example config 3 (flag only)
)

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

#     train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/"$SCENE"/ \
        --result_dir $RESULT_DIR/"$SCENE"/
done

CONFIG_INDEX=1

for CONFIG in "${CONFIG_OPTIONS[@]}"; do
    CONFIG_RESULT_DIR="$RESULT_DIR_BASE/result$CONFIG_INDEX"

    for SCENE in $SCENE_LIST; do
            if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
                DATA_FACTOR=2
            else
                DATA_FACTOR=4
            fi

            echo "Running scene: $SCENE with config $CONFIG (result dir: result$CONFIG_INDEX)"

            CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
                --render_traj_path $RENDER_TRAJ_PATH \
                --data_dir "$SCENE_DIR/$SCENE/" \
                --result_dir "$CONFIG_RESULT_DIR/$SCENE/" \
                "$CONFIG"
        done

        ((CONFIG_INDEX++))
done