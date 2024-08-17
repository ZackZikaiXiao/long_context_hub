#!/bin/bash

# 设置参数
BASE_PATH="/Volumes/main/default/default_volume/erikyzzhang/long_context"
MODEL_NAME_OR_PATH="${BASE_PATH}/models/llama3-8B-32k-ft/checkpoint-700"
MODEL_MAX_LENGTH=$((16384 * 2))
OUTPUT_DIR="${BASE_PATH}/models/llama3-8B-32k-ft-sft"
DATA_PATH="${BASE_PATH}/data/LongAlpaca-12k.json"

# 如果输出目录不存在，创建它
if [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
    mkdir -p "$(dirname "$OUTPUT_DIR")"
fi

# 运行 supervised-fine-tune.py 并传递参数
torchrun --nproc_per_node=8 --master_port=29499 supervised-fine-tune.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATA_PATH \
    --model_max_length $MODEL_MAX_LENGTH \
    --output_dir $OUTPUT_DIR
