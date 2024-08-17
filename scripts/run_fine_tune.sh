#!/bin/bash

# 得到通过redpajama ft预训练的模型
# 参数设置

MODEL_NAME_OR_PATH="./models/TinyLlama-1.1B-Chat-v1.0"
MODEL_MAX_LENGTH=$((16384 * 1)) 
OUTPUT_DIR="./tinyllama_weights/tinyllama-1.1B-16k-ft-mlp"
DATASET_PATH="./RedPajama-Data-1T-Sample"

# MODEL_NAME_OR_PATH="./models/llama3"
# MODEL_MAX_LENGTH=$((16384 * 2)) 
# OUTPUT_DIR="./llama3_weights/TinyLlama-8B-32k-ft"

# 如果输出目录不存在，创建它
if [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
    mkdir -p "$(dirname "$OUTPUT_DIR")"
fi

# 运行 fine-tune.py 并传递参数
torchrun --nproc_per_node=8 fine-tune.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATASET_PATH \
    --model_max_length $MODEL_MAX_LENGTH \
    --output_dir $OUTPUT_DIR