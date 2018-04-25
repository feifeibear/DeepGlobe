#!/bin/bash
# set -x
# set -e

export CUDA_VISIBLE_DEVICES=3

export RUN_TRAIN=0
export RUN_TEST=1

export IS_RESTART=0
export START_EPOCH=0
export STOP_EPOCH=75
export FIT_BATCH_SIZE=2

export PROJ_BASE_PATH="/root"

MODEL="AOI_3_Paris"
TRAIN_PATH=$PROJ_BASE_PATH/data/train/${MODEL}_Train
TEST_PATH=$PROJ_BASE_PATH/data/test/${MODEL}_Test_public

mkdir -p $PROJ_BASE_PATH/data/working || exit
source activate py35  || exit 1

#-=-=-=-=-=-=-=--=-=-=- TRAINNING -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if [[ $RUN_TRAIN -eq 1 ]]; then
  # echo ">>> clean up"
  # rm -rf $PROJ_BASE_PATH/data/working && mkdir -p $PROJ_BASE_PATH/data/working

  echo ">>> RUN TRAIN"
  echo ">>> PREPROCESSING STEP"
  echo python v5_im.py preproc_train $TRAIN_PATH
  python $PROJ_BASE_PATH/code/v5_im.py preproc_train $TRAIN_PATH || exit 1

  echo python v12_im.py preproc_train $TRAIN_PATH
  python $PROJ_BASE_PATH/code/v12_im.py preproc_train $TRAIN_PATH || exit 1

  echo ">>> TRAINING v9s model"
  echo python v9s.py validate $TRAIN_PATH
  python $PROJ_BASE_PATH/code/v9s.py validate $TRAIN_PATH | tee -a $PROJ_BASE_PATH/data/v9s-validate.log || exit 1

  echo python v9s.py evalfscore $TRAIN_PATH
  python $PROJ_BASE_PATH/code/v9s.py evalfscore $TRAIN_PATH | tee -a $PROJ_BASE_PATH/data/v9s-evalfscore.log|| exit 1

  ### v13 --------------
  echo ">>>>>>>>>> v13.py: Training for v13 model"
  python $PROJ_BASE_PATH/code/v13.py validate $TRAIN_PATH | tee -a $PROJ_BASE_PATH/data/v13-validate.log || exit 1

  echo ">>>>>>>>>> v13.py: Parametr optimization for v13 model"
  python $PROJ_BASE_PATH/code/v13.py evalfscore $TRAIN_PATH | tee -a $PROJ_BASE_PATH/data/v13-evalfscore.log || exit 1

  ### v17 --------------
  echo ">>>>>>>>>> v17.py"
  python $PROJ_BASE_PATH/code/v17.py evalfscore $TRAIN_PATH || exit 1
fi

#-=-=-=-=-=-=-=--=-=-=- TESTING -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if [[ $RUN_TEST -eq 1 ]]; then
  # echo "clean up"
  # rm -f $PROJ_BASE_PATH/data/working/images/v5/test_AOI_*_im.h5
  # rm -f $PROJ_BASE_PATH/data/working/images/v5/test_AOI_*_mul.h5
  # rm -f $PROJ_BASE_PATH/data/working/images/v12/test_AOI_*_mul.h5
  # rm -f $PROJ_BASE_PATH/data/working/images/v16/test_AOI_*_osm.h5

  echo "run test"
  echo ">>> PREPROCESSING STEP"
  echo ">>>" python v5_im.py preproc_test $TEST_PATH
  python $PROJ_BASE_PATH/code/v5_im.py preproc_test $TEST_PATH || exit 1

  echo ">>>" python v12_im.py preproc_test $TEST_PATH
  python $PROJ_BASE_PATH/code/v12_im.py preproc_test $TEST_PATH || exit 1

  echo ">>> INFERENCE STEP"
  echo ">>>" python v9s.py testproc $TEST_PATH
  python $PROJ_BASE_PATH/code/v9s.py testproc $TEST_PATH || exit 1

  echo ">>>" python v13.py testproc $TEST_PATH
  python $PROJ_BASE_PATH/code/v13.py testproc $TEST_PATH || exit 1

  echo ">>>" python v17.py testproc $TEST_PATH
  python $PROJ_BASE_PATH/code/v17.py testproc $TEST_PATH || exit 1
fi
