DATETIME=$(date '+%Y-%m-%d-%H')
RUN_NAME="check600_v_1_video-r1_stage2_cyt"
OUTPUT_DIR=/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/$RUN_NAME/$DATETIME
mkdir -p $OUTPUT_DIR
export WANDB_PROJECT=timesearch-R-stage_2
export WANDB_NAME=$RUN_NAME
export LOG_PATH=${OUTPUT_DIR}/log.txt
# export DEBUG=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=".:$PYTHONPATH"
export SIGLIP_URL=grpc://127.0.0.1:52000
# export LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1
export WANDB_API_KEY="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"
# Local training configuration
NUM_GPUS=8
MASTER_PORT=29500

echo "Local training mode: ${NUM_GPUS} GPUs on localhost:${MASTER_PORT}"

TRAIN_PATH=configs/dataset.yaml

VIDEO_ROOT=/xuhongbo/shuimu.chen/LongVideoBench/LongVideoHaystack/videos_480p_noaudio

MODEL_BASE=/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/check600_v_1_video-r1_stage1_test_cyt/2026-01-28-01/Qwen2.5-VL-1200

# MODEL_BASE=/xuhongbo//shuimu.chen/Qwen2.5-VL-3B-Instruct
# MODEL_BASE=/data/shuimu.chen/Qwen2.5-VL-3B-Instruct
# MODEL_BASE=/xuhongbo/shuimu.chen/TimeSearch-R/Qwen2.5-VL-GRPO

    # --max_prompt_length 18000 \
    # --max_completion_length 16000 \
torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=${MASTER_PORT} \
    time_r1/train_VLLM_stage_2.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_BASE \
    --train_data_path $TRAIN_PATH \
    --video_folder $VIDEO_ROOT \
    --reward_func v10 \
    --prompt_template v3 \
    --tool_name_list seek_video_frames \
    --max_interaction_turns 4 \
    --max_prompt_length 24000 \
    --max_completion_length 16000 \
    --max_completion_length_per_turn 256 \
    --total_video_tokens 10240 \
    --max_frames 700 \
    --min_per_frame_tokens 4 \
    --max_per_frame_tokens 256 \
    --num_generations 8 \
    --scale_rewards false \
    --beta 0.005 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 1 \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --report_to wandb \
    --save_steps 400 \
    --save_only_model true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --shuffle_dataset true \
    --replay_buffer_type dapo \
    --lr_scheduler_type "cosine" \
    --log_completions true \
    --use_counterfactual_reasoning true
