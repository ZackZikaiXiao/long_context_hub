#!/bin/bash


# 参数设置
MODEL_NAME_OR_PATH="./tinyllama_weights/tinyllama-1.1B-16k-ft"

MODEL_MAX_LENGTH=$((16384 * 1)) 
OUTPUT_DIR="./tinyllama_weights/tinyllama-1.1B-16k-ft-sft-loramel-repeat-2epoch"
# OUTPUT_DIR="./tinyllama_weights/tinyllama-1.1B-16k-ft-sft-random50-lds-sqrt2"
# DATASET_PATH="./data_augument/aeda_LongAlpaca-12k.json"
DATASET_PATH="./dataset/LongAlpaca-12k.json"




# 如果输出目录不存在，创建它
if [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
    mkdir -p "$(dirname "$OUTPUT_DIR")"
fi


# 运行 supervised-fine-tune.py 并传递参数
torchrun --nproc_per_node=8 --master_port=29501 supervised-fine-tune-LoRAMEL.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATASET_PATH \
    --model_max_length $MODEL_MAX_LENGTH \
    --output_dir $OUTPUT_DIR
