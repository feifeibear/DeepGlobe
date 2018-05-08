#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export MODEL="AOI_4_Shanghai"

export RUN_TRAIN=1
export RUN_TEST=1

export IS_RESTART=0
export START_EPOCH=0
export STOP_EPOCH=75
export FIT_BATCH_SIZE=8
export PRED_BATCH_SIZE=8

#-=-=-=-=-=-=-=-= No need to modify -=-=-=-=-=-=-=-=-=--=-=-
TRAIN_PATH=/root/data/train/${MODEL}_Train
TEST_PATH=/root/data/test/${MODEL}_Test_public

mkdir -p /root/data/working || exit
source activate py35  || exit 1

#-=-=-=-=-=-=-=--=-=-=- TRAINNING -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if [[ $RUN_TRAIN -eq 1 ]]; then
  # echo ">>> clean up"
  # rm -rf /root/data/working && mkdir -p /root/data/working

  echo ">>> RUN TRAIN"
  echo ">>> PREPROCESSING STEP"
  echo python v5_im.py preproc_train $TRAIN_PATH
  # python /root/code/v5_im.py preproc_train $TRAIN_PATH || exit 1

  echo python v12_im.py preproc_train $TRAIN_PATH
  # python /root/code/v12_im.py preproc_train $TRAIN_PATH || exit 1

  # echo python v16.py preproc_train $TRAIN_PATH
  # python v16.py preproc_train $TRAIN_PATH

  echo ">>> TRAINING v9s model"
  echo python v9s.py validate $TRAIN_PATH
  python /root/code/v9s.py validate $TRAIN_PATH  || exit 1

  echo python v9s.py evalfscore $TRAIN_PATH
  python /root/code/v9s.py evalfscore $TRAIN_PATH || exit 1

  ### v13 --------------
  echo ">>>>>>>>>> v13.py: Training for v13 model"
  python /root/code/v13.py validate $TRAIN_PATH  || exit 1

  echo ">>>>>>>>>> v13.py: Parametr optimization for v13 model"
  python /root/code/v13.py evalfscore $TRAIN_PATH || exit 1

  ### v16 --------------
  #echo ">>>>>>>>>> v16.py"
  #python /root/code/v16.py validate $TRAIN_PATH

  #echo ">>>>>>>>>> v16.py"
  #python /root/code/v16.py evalfscore $TRAIN_PATH

  ### v17 --------------
  echo ">>>>>>>>>> v17.py"
  python /root/code/v17.py evalfscore $TRAIN_PATH || exit 1
fi

#-=-=-=-=-=-=-=--=-=-=- TESTING -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if [[ $RUN_TEST -eq 1 ]]; then
  # echo "clean up"
  # rm -f /root/data/working/images/v5/test_AOI_*_im.h5
  # rm -f /root/data/working/images/v5/test_AOI_*_mul.h5
  # rm -f /root/data/working/images/v12/test_AOI_*_mul.h5
  # rm -f /root/data/working/images/v16/test_AOI_*_osm.h5

  echo "run test"
  echo ">>> PREPROCESSING STEP"
  echo ">>>" python v5_im.py preproc_test $TEST_PATH
  python /root/code/v5_im.py preproc_test $TEST_PATH || exit 1

  echo ">>>" python v12_im.py preproc_test $TEST_PATH
  python /root/code/v12_im.py preproc_test $TEST_PATH || exit 1

  #echo ">>>" python v16.py preproc_test $TEST_PATH
  #python /root/code/v16.py preproc_test $TEST_PATH

  echo ">>> INFERENCE STEP"
  echo ">>>" python v17.py testproc $TEST_PATH
  python /root/code/v17.py testproc $TEST_PATH || exit 1

  echo ">>> INFERENCE v9s and v13 after v17"
  echo ">>>" python v9s.py testproc $TEST_PATH
  python /root/code/v9s.py testproc $TEST_PATH || exit 1

  echo ">>>" python v13.py testproc $TEST_PATH
  python /root/code/v13.py testproc $TEST_PATH || exit 1
fi
