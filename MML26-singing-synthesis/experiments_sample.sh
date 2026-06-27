#!/bin/bash

# train on Synthetic, validate/test on Klangio singing dataset
OUTPUT_DIR="./BASIC_PITCH_CHALLENGE/"
COMMON="--output-dir $OUTPUT_DIR --logger tensorboard"

python3 -m src.train \
    --train-dataset Synthetic \
    --val-dataset Klangio \
    --sequence-length 8 \
    --batch-size 32 \
    --frame-weight 9.0 \
    --onset-weight 18.0 \
    --learning-rate 1e-4 \
    --eval-metric COnPOff_f1 \
    --precision 32 \
    --experiment-name "basic_pitch_training" \
    $COMMON
