SCENE_DIR="../../360_v2"
RESULT_DIR="../../results/benchmark"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

RESULT_DIR_BASE="../../results/benchmark"

CONFIG_OPTIONS=(
    "strategy:default-strategy --lambda_shn_reg 0 --lambda_gray_world 0"
    "strategy:default-strategy --lambda_shn_reg 0 --lambda_gray_world 0 --lambda_tv_loss 0 --lambda_exclusion 0"
    "strategy:default-strategy --lambda_shn_reg 0 --lambda_gray_world 0 --appearance_embeddings"
    "strategy:default-strategy --lambda_shn_reg 0 --lambda_gray_world 0 --decomposed_field"
    "strategy:default-strategy --lambda_shn_reg 0 --lambda_gray_world 0 --decomposed_field --appearance_embeddings"
    "strategy:default-strategy --lambda_shn_reg 0 --lambda_gray_world 0 --lambda_tv_loss 0"
    "strategy:default-strategy --lambda_shn_reg 0 --lambda_gray_world 0 --lambda_tv_loss 0 --lambda_illum_smoothness 0.001"
    "strategy:default-strategy --lambda_shn_reg 0 --lambda_tv_loss 0"
    "strategy:default-strategy --lambda_gray_world 0"
    "strategy:default-strategy --lambda_gray_world 0 --lambda_tv_loss 0 --lambda_illum_smoothness 0.001"
)

#for SCENE in $SCENE_LIST;
#do
#    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "garden" ] || [ "$SCENE" = "stump" ]; then
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
##     train without eval
#    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
#        --render_traj_path $RENDER_TRAJ_PATH \
#        --data_dir $SCENE_DIR/"$SCENE"/ \
#        --result_dir $RESULT_DIR/"$SCENE"/
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
                "$CONFIG"
        done

        ((CONFIG_INDEX++))
done