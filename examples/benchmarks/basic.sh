SCENE_DIR="../../360_v2"
RESULT_DIR="../../results/benchmark"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/"$SCENE"/ \
        --result_dir $RESULT_DIR/"$SCENE"/
done