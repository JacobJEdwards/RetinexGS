SCENE_DIR="../../360_v2"
RESULT_DIR="../../results/benchmark"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RESULT_DIR_BASE="../../results/benchmark"
RENDER_TRAJ_PATH="ellipse"

CONFIG_OPTIONS=(
    "--learn_edge_aware_gamma"                  # example config 1
    "--no-learn_spatial_contrast --no-learn_adaptive_curve_lambdas --no-learn_local_exposure --no-learn_global_exposure"
    "--use_dynamic_weights"
    "--predictive_adaptive_curve"
    "--illum_opt_type base"
    "--no-use_illum_opt"
)

#for SCENE in $SCENE_LIST;
#do
#    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "garden" ]; then
#      continue
#    fi
#
#    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
#        DATA_FACTOR=2
#    else
#        DATA_FACTOR=4
#    fi
#
#    echo "Running $SCENE"
#
#    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
#        --render_traj_path $RENDER_TRAJ_PATH \
#        --data_dir $SCENE_DIR/"$SCENE"/ \
#        --result_dir $RESULT_DIR/"$SCENE"/
#
#done

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
            $CONFIG
    done

    ((CONFIG_INDEX++))
done