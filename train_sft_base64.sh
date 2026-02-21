#!/bin/bash

DATETIME=$(date '+%Y-%m-%d-%H')
RUN_NAME="SFT_60frame_base64" ## 必须，不然OOM
OUTPUT_DIR=/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/$RUN_NAME/$DATETIME
mkdir -p $OUTPUT_DIR

# # === 环境变量设置 (关键修改部分) ===

# # 1. [核心修复] 指定用于握手的网卡
# # 根据你的 ifconfig，bond0.3102 是主通信网卡
# export NCCL_SOCKET_IFNAME=bond0.3102

# # 2. [推荐] 单机训练禁用 IB，防止 IB 驱动报错干扰 NCCL 初始化
# export NCCL_IB_DISABLE=1

# # 3. [可选] P2P 设置
# # 如果你的显卡之间有 NVLink 桥接，请保留下面这行 NVL 设置，并注释掉 DISABLE
# export NCCL_P2P_LEVEL=NVL
# # export NCCL_P2P_DISABLE=1  # 除非遇到 P2P 报错或卡死，否则建议注释掉这行以获得更好性能

# # 4. 调试日志 (如果再次报错，取消下面两行的注释看详细信息)
# # export NCCL_DEBUG=INFO
# # export TORCH_DISTRIBUTED_DEBUG=DETAIL

# # 牺牲速度换稳定性
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=bond0.3102

# === 其他常规设置 ===
export WANDB_PROJECT=TimeSearch-R-SFT
export WANDB_NAME=$RUN_NAME
export LOG_PATH=${OUTPUT_DIR}/log.txt
export DEBUG=true
export PYTHONPATH=".:$PYTHONPATH"

# 这是一个可能导致冲突的变量，通常不需要手动设置 SIGLIP_URL 除非代码显式依赖
# 如果代码里没用到 GRPC 连接这个地址，可以考虑注释掉
export SIGLIP_URL=grpc://127.0.0.1:52000 

export WANDB_API_KEY="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"
export DECORD_EOF_RETRY_MAX=2048001 ## 增加次数
# Local training configuration
NUM_GPUS=8
MASTER_PORT=28510
echo "Local training mode: ${NUM_GPUS} GPUs on localhost:${MASTER_PORT}"

TRAIN_PATH=configs/dataset_sft.yaml
VIDEO_ROOT=/xuhongbo/shuimu.chen/LongVideoBench/videos_480p_noaudio
MODEL_BASE=/scratch/prj0000000262-bucket/ocr/ec/models/Qwen2.5-VL-7B-Instruct

# 启动命令
torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=${MASTER_PORT} \
    time_r1/sft_base64.py \
    --deepspeed /scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/scripts/zero2_offload.json \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_BASE \
    --dataset_name dummy \
    --train_data_path $TRAIN_PATH \
    --video_folder $VIDEO_ROOT \
    --prompt_template v3 \
    --tool_name_list seek_video_frames \
    --total_video_tokens  24000 \
    --max_frames 60 \
    --min_per_frame_tokens 4 \
    --max_per_frame_tokens 192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 0 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing True \
    --attn_implementation flash_attention_2 \
    --learning_rate 1e-6 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --report_to wandb \
    --save_steps 3000 \
    --save_only_model true
