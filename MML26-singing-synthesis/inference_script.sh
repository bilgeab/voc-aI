#!/bin/bash

CHECKPOINT_PATH=""
INPUT_PATH=""
OUTPUT_PATH=""

echo "Running Basic Pitch inference"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Input: $INPUT_PATH"
echo "  Output: $OUTPUT_PATH"

python -m src.inference \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --input-path "$INPUT_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --save-activations
