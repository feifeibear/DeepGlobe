#!/bin/bash
# set -x
# set -e

export CUDA_VISIBLE_DEVICES=5
export IS_RESTART=0
export START_EPOCH=0
export STOP_EPOCH=75
export PROJ_BASE_PATH="/root"

MODEL="AOI_3_Paris"
TRAIN_PATH_LIST="
$PROJ_BASE_PATH/data/train/${MODEL}_Train
"

mkdir -p $PROJ_BASE_PATH/data/working
# echo ">>> CLEAN UP" && echo rm -rf $PROJ_BASE_PATH/data/working && rm -rf $PROJ_BASE_PATH/data/working && mkdir -p $PROJ_BASE_PATH/data/working

source activate py35 && for train_path in $TRAIN_PATH_LIST; do
    echo ">>> PREPROCESSING STEP"
    echo python v5_im.py preproc_train $train_path
    python $PROJ_BASE_PATH/code/v5_im.py preproc_train $train_path || exit 1

    echo python v12_im.py preproc_train $train_path
    python $PROJ_BASE_PATH/code/v12_im.py preproc_train $train_path || exit 1

    echo ">>> TRAINING v9s model"
    echo python v9s.py validate $train_path
    python $PROJ_BASE_PATH/code/v9s.py validate $train_path | tee -a $PROJ_BASE_PATH/data/v9s-validate.log || exit 1

    echo python v9s.py evalfscore $train_path
    python $PROJ_BASE_PATH/code/v9s.py evalfscore $train_path | tee -a $PROJ_BASE_PATH/data/v9s-evalfscore.log|| exit 1

    ### v13 --------------
    echo ">>>>>>>>>> v13.py: Training for v13 model"
    python $PROJ_BASE_PATH/code/v13.py validate $train_path | tee -a $PROJ_BASE_PATH/data/v13-validate.log || exit 1

    echo ">>>>>>>>>> v13.py: Parametr optimization for v13 model"
    python $PROJ_BASE_PATH/code/v13.py evalfscore $train_path | tee -a $PROJ_BASE_PATH/data/v13-evalfscore.log || exit 1

    ### v17 --------------
    echo ">>>>>>>>>> v17.py"
    python $PROJ_BASE_PATH/code/v17.py evalfscore $train_path || exit 1
done
