#!/bin/bash

DATETIME=$(date '+%Y-%m-%d-%H')
RUN_NAME="SFT_Video_R1_cyt_60_frame_epoch1_zero2_fix_work1——pilance-16-128" ## 必须，不然OOM
OUTPUT_DIR=/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/$RUN_NAME/$DATETIME
mkdir -p $OUTPUT_DIR


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
accelerate launch --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=${MASTER_PORT} \
    time_r1/sft.py \
    --deepspeed /scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/scripts/zero2_offload.json \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_BASE \
    --train_data_path $TRAIN_PATH \
    --video_folder $VIDEO_ROOT \
    --prompt_template v3 \
    --tool_name_list seek_video_frames \
    --total_video_tokens  24000 \
    --max_frames 60 \
    --min_per_frame_tokens 4 \
    --max_per_frame_tokens 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 1 \
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
    --save_steps 1000 \
    --save_only_model true