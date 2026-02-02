DATETIME=$(date '+%Y-%m-%d-%H')
RUN_NAME="training"
OUTPUT_DIR=/data/shuimu.chen/TimeSearch-R/experiment/$RUN_NAME/$DATETIME
mkdir -p $OUTPUT_DIR
export WANDB_PROJECT=TimeSearch-R-ColdStart
export WANDB_NAME=$RUN_NAME
export LOG_PATH=${OUTPUT_DIR}/log.txt
export DEBUG=true
export CUDA_VISIBLE_DEVICES=1,2,3
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=".:$PYTHONPATH"
export SIGLIP_URL=grpc://127.0.0.1:51000
export LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1

# Local training configuration
NUM_GPUS=3
MASTER_PORT=29500

echo "Local training mode: ${NUM_GPUS} GPUs on localhost:${MASTER_PORT}"

TRAIN_PATH=configs/dataset.yaml

VIDEO_ROOT=/data/shuimu.chen/LongVideoBench/LongVideoHaystack/videos

MODEL_BASE=/data/shuimu.chen/Video-R1/Qwen2.5-VL_COT_SFT_offitial

torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=${MASTER_PORT} \
    time_r1/train.py \
    --deepspeed scripts/zero3.json\
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_BASE \
    --train_data_path $TRAIN_PATH \
    --video_folder $VIDEO_ROOT \
    --reward_func v7 \
    --prompt_template v4 \
    --tool_name_list seek_video_frames \
    --max_interaction_turns 8 \
    --max_prompt_length 18000 \
    --max_completion_length 16000 \
    --max_completion_length_per_turn 256 \
    --total_video_tokens 10240 \
    --max_frames 768 \
    --min_per_frame_tokens 12 \
    --max_per_frame_tokens 256 \
    --num_generations 3 \
    --scale_rewards false \
    --beta 0.005 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 1 \
    --dataloader_num_workers 0 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --report_to wandb \
    --save_steps 200 \
    --save_only_model true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.8 \
    --shuffle_dataset true \
    --replay_buffer_type dapo \
    --use_counterfactual_reasoning true
