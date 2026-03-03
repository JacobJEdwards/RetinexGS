export PYTORCH_ALLOC_CONF=expandable_segments:True

SCENE_DIR="/workspace/360_v2"
RESULT_DIR="/workspace/retinexgs"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
POSTFIXES="contrast variance multiexposure"

for POSTFIX in $POSTFIXES;
  do
  for SCENE in $SCENE_LIST;
  do
      echo "Running $SCENE"

      NEW_RESULT_DIR=$RESULT_DIR/"$POSTFIX"/"$SCENE"
      DATADIR=$SCENE_DIR/"$SCENE"/

      CUDA_VISIBLE_DEVICES=0 python simple_trainer.py --disable_viewer \
          --data_dir $DATADIR \
          --postfix $POSTFIX \
          --result_dir $NEW_RESULT_DIR
  done
done
