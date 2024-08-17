#!/bin/bash
export NCCL_P2P_LEVEL=NVL
# 参数设置
MODEL_MAX_LENGTH=$((16384 * 1)) 

# 创建输出目录函数
create_output_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# # Stage 1:
# # 100%
# MODEL_NAME_OR_PATH="./tinyllama_weights/tinyllama-1.1B-16k-ft"
# OUTPUT_DIR="./tinyllama_weights_increased_length_upsampling/tinyllama-1.1B-16k-ft-sft-random30-quad-root-sampler-increased_length_1"
# FILTER_MODE="increased_length_1" 
# NUM_TRAIN_EPOCHS=1

# create_output_dir "$OUTPUT_DIR"

# # 运行 supervised-fine-tune.py 并传递参数
# torchrun --nproc_per_node=8 --master_port=29501 supervised-fine-tune-sampler.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --model_max_length $MODEL_MAX_LENGTH \
#     --filter_mode $FILTER_MODE \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $NUM_TRAIN_EPOCHS

# # 检查第一个阶段是否成功完成
# if [ $? -ne 0 ]; then
#     echo "Stage 1 failed. Exiting."
#     exit 1
# fi

# Stage 2:
# MODEL_NAME_OR_PATH="./tinyllama_weights_increased_length_upsampling/tinyllama-1.1B-16k-ft-sft-random30-quad-root-sampler-increased_length_1"
# OUTPUT_DIR="./tinyllama_weights_increased_length_upsampling/tinyllama-1.1B-16k-ft-sft-random30-quad-root-sampler-increased_length_2"
# FILTER_MODE="increased_length_2" 
# NUM_TRAIN_EPOCHS=2
# create_output_dir "$OUTPUT_DIR"

# torchrun --nproc_per_node=8 --master_port=29502 supervised-fine-tune-sampler.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --model_max_length $MODEL_MAX_LENGTH \
#     --filter_mode $FILTER_MODE \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $NUM_TRAIN_EPOCHS

# if [ $? -ne 0 ]; then
#     echo "Stage 2 failed. Exiting."
#     exit 1
# fi

# # Stage 3:
# MODEL_NAME_OR_PATH="./tinyllama_weights_increased_length_upsampling/tinyllama-1.1B-16k-ft-sft-random30-quad-root-sampler-increased_length_2"
# OUTPUT_DIR="./tinyllama_weights_increased_length_upsampling/tinyllama-1.1B-16k-ft-sft-random30-quad-root-sampler-increased_length_3"
# FILTER_MODE="increased_length_3" 
# NUM_TRAIN_EPOCHS=3
# create_output_dir "$OUTPUT_DIR"

# torchrun --nproc_per_node=8 --master_port=29503 supervised-fine-tune-sampler.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --model_max_length $MODEL_MAX_LENGTH \
#     --filter_mode $FILTER_MODE \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs $NUM_TRAIN_EPOCHS

# if [ $? -ne 0 ]; then
#     echo "Stage 3 failed. Exiting."
#     exit 1
# fi

# Stage 4:
MODEL_NAME_OR_PATH="./tinyllama_weights_increased_length_upsampling/tinyllama-1.1B-16k-ft-sft-random30-quad-root-sampler-increased_length_3/checkpoint-141"
OUTPUT_DIR="./tinyllama_weights_increased_length_upsampling/tinyllama-1.1B-16k-ft-sft-random30-quad-root-sampler-increased_length_4"
FILTER_MODE="increased_length_4" 
NUM_TRAIN_EPOCHS=4
create_output_dir "$OUTPUT_DIR"

torchrun --nproc_per_node=8 --master_port=29504 supervised-fine-tune-sampler.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_max_length $MODEL_MAX_LENGTH \
    --filter_mode $FILTER_MODE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS

if [ $? -ne 0 ]; then
    echo "Stage 4 failed. Exiting."
    exit 1
fi
