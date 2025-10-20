set -ex
MODEL='bicycle_gan'
CLASS='sony_wieTotal_data'
NZ=8
NO_FLIP='--no_flip'
DIRECTION='BtoA'
LOAD_SIZE=286
CROP_SIZE=256
INPUT_NC=1
NITER=30
NITER_DECAY=30

CHECKPOINTS_DIR=./checkpoints/${CLASS}/
NAME=${CLASS}_${MODEL}


  python ./train.py \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --direction ${DIRECTION} \
  --display_id -1\