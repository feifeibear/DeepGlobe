#!/bin/bash
# set -x
set -e

export CUDA_VISIBLE_DEVICES=6

export PROJ_BASE_PATH="/root"

MODEL="AOI_3_Paris"

TEST_PATH_LIST="
$PROJ_BASE_PATH/data/test/${MODEL}_Test_public
"

# clean up
mkdir -p $PROJ_BASE_PATH/data/output $PROJ_BASE_PATH/data/working
# rm -f $PROJ_BASE_PATH/data/working/images/v5/${MODEL}_test_ImageId.csv
# rm -f $PROJ_BASE_PATH/data/working/images/v5/test_${MODEL}_im.h5
# rm -f $PROJ_BASE_PATH/data/working/images/v5/test_${MODEL}_im.h5
# rm -f $PROJ_BASE_PATH/data/working/images/v5/test_${MODEL}_mul.h5
# rm -f $PROJ_BASE_PATH/data/working/images/v12/test_${MODEL}_mul.h5

source activate py35 && for test_path in $TEST_PATH_LIST; do
    echo ">>> PREPROCESSING STEP"
    echo ">>>" python v5_im.py preproc_test $test_path
    python $PROJ_BASE_PATH/code/v5_im.py preproc_test $test_path || exit 1

    echo ">>>" python v12_im.py preproc_test $test_path
    python $PROJ_BASE_PATH/code/v12_im.py preproc_test $test_path || exit 1

    echo ">>> INFERENCE STEP"
    echo ">>>" python v9s.py testproc $test_path
    python $PROJ_BASE_PATH/code/v9s.py testproc $test_path || exit 1

    echo ">>>" python v13.py testproc $test_path
    python $PROJ_BASE_PATH/code/v13.py testproc $test_path || exit 1

    echo ">>>" python v17.py testproc $test_path
    python $PROJ_BASE_PATH/code/v17.py testproc $test_path || exit 1
done
