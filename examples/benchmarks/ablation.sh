SCENE_DIR="/workspace/360_v2"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
POSTFIXES="variance multiexposure contrast"

RESULT_DIR="/workspace/3d"

#for POSTFIX in $POSTFIXES; do
#  for SCENE in $SCENE_LIST;
#  do
#
#      if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
#          DATA_FACTOR=2
#      else
#          DATA_FACTOR=4
#      fi
#
#      echo "Running $SCENE on $POSTFIX"
#
#      CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
#          --data_dir $SCENE_DIR/"$SCENE"/ \
#          --postfix "$POSTFIX" \
#          --save_images \
#          --tb_save_image \
#          --result_dir $RESULT_DIR/"$POSTFIX"/"$SCENE"/
#  done
#done

CONFIG_OPTIONS=(
    "--lambda-exclusion 0.0 --lambda-shn-reg 0.0"
    "--no-use-normals"
    "--no-use-view-dirs"
    "--no-use-camera-response-network"
    "--lambda_illum_smoothness 1.0"
    "--lambda_tv_loss 1.0"

)
CONFIG_NAMES=(
    "no_loss_terms"
    "no_normals"
    "no_view_dirs"
    "no_crn"
    "with_smoothness"
    "with_tv_loss"
)

for POSTFIX in $POSTFIXES; do
    RESULT_DIR_BASE="/workspace/3d/ablation/$POSTFIX"

    for SCENE in $SCENE_LIST;
    do
        for i in "${!CONFIG_OPTIONS[@]}"; do
            echo "Running $SCENE with ${CONFIG_NAMES[$i]} on $POSTFIX"
            RESULT_DIR="$RESULT_DIR_BASE/${CONFIG_NAMES[$i]}"
            CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer \
                --data_dir $SCENE_DIR/"$SCENE"/ \
                --postfix $POSTFIX \
                --no_save_images \
                --no_save_ckpt \
                --no_tb_save_image \
                --tb_every 1000 \
                --result_dir $RESULT_DIR/"$SCENE"/ \
                ${CONFIG_OPTIONS[$i]}
        done
    done
done