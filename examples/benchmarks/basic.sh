SCENE_DIR="/workspace/360_v2"
RESULT_DIR_PPISP="/workspace/ppisp"
RESULT_DIR_BILAT="/workspace/bilateral_grid"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
POSTFIXES="contrast variance multiexposure"

for POSTFIX in $POSTFIXES;
  do
  for SCENE in $SCENE_LIST;
  do
      echo "Running $SCENE"

      NEW_RESULT_DIR_PPISP=$RESULT_DIR_PPSIP/"$POSTFIX"/"$SCENE"
      DATADIR=$SCENE_DIR/"$SCENE"/

      CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer \
          --data_dir $DATADIR \
          --postfix $POSTFIX \
          --result_dir $NEW_RESULT_DIR_PPISP \
          --postprocessing ppisp

      NEW_RESULT_DIR_BILAT=$RESULT_DIR_BILAT/"$POSTFIX"/"$SCENE"

      CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --disable_viewer \
          --data_dir $DATADIR \
          --postfix $POSTFIX \
          --result_dir $NEW_RESULT_DIR_BILAT \
          --postprocessing bilateral_grid

  done
done
