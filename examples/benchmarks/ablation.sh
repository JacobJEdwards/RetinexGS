SCENE_DIR="../../360_v2"
RESULT_DIR="../../3d/contrast/results/benchmark"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"
POSTFIXES="_contrast _variance _multiexposure"
CONFIG_OPTIONS=(
    "--no_appearance_embeddings"
    "--no-use_view_dirs"
    "--no-use_normals"
    "--no-use_camera_response_network"
    "--lambda_illum_smoothness 0.0"
    "--lambda_exclusion 0.0"
    "--lambda_shn_reg 0.0"
    "--lambda_tv_loss 0.0"
    "--learning_steps 2000"
    "--learning_steps 5000"
    "--learning_steps 8000"
    "--learning_steps 10000"
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
    "steps_2k"
    "steps_5k"
    "steps_8k"
    "steps_10k"
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
                --render_traj_path $RENDER_TRAJ_PATH \
                --data_dir $SCENE_DIR/"$SCENE"/ \
                --result_dir $RESULT_DIR/"$SCENE"/ \
                ${CONFIG_OPTIONS[$i]}
        done
    done
done