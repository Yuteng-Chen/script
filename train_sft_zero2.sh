DATETIME=$(date '+%Y-%m-%d-%H')
RUN_NAME="SFT_Video_R1_cyt_60_frame_epoch10_zero2"
OUTPUT_DIR=/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/$RUN_NAME/$DATETIME
mkdir -p $OUTPUT_DIR
export WANDB_PROJECT=TimeSearch-R-SFT
export WANDB_NAME=$RUN_NAME
export LOG_PATH=${OUTPUT_DIR}/log.txt
export DEBUG=true
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=".:$PYTHONPATH"



# === 在这里设置环境变量 ===
# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_TIMEOUT=22
# export TORCH_NCCL_BLOCKING_WAIT=0
# export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口
# export NCCL_P2P_DISABLE=1
# export MAX_JOBS=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1  # 同步CUDA操作






export PYTHONPATH=".:$PYTHONPATH"
export SIGLIP_URL=grpc://127.0.0.1:52000
export WANDB_API_KEY="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"
# export LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1

# Local training configuration
NUM_GPUS=8
MASTER_PORT=28509
echo "Local training mode: ${NUM_GPUS} GPUs on localhost:${MASTER_PORT}"


TRAIN_PATH=configs/dataset_sft.yaml

VIDEO_ROOT=/xuhongbo/shuimu.chen/LongVideoBench/videos_480p_noaudio


MODEL_BASE=/scratch/prj0000000262-bucket/ocr/ec/models/Qwen2.5-VL-7B-Instruct


torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=${MASTER_PORT} \
    time_r1/sft.py \
    --deepspeed /scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/scripts/zero2.json \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_BASE \
    --train_data_path $TRAIN_PATH \
    --video_folder $VIDEO_ROOT \
    --prompt_template v3 \
    --tool_name_list seek_video_frames \
    --total_video_tokens 10240 \
    --max_frames 60 \
    --min_per_frame_tokens 4 \
    --max_per_frame_tokens 192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing True \
    --attn_implementation flash_attention_2 \
    --learning_rate 1e-6 \
    --num_train_epochs 10 \
    --run_name $RUN_NAME \
    --report_to wandb \
    --save_steps 1000 \
    --save_only_model true 

