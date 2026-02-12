DATETIME=$(date '+%Y-%m-%d-%H')
RUN_NAME="split_tool_s3_1e-6_wandb_no_reasoning"
OUTPUT_DIR=/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/$RUN_NAME/$DATETIME
mkdir -p $OUTPUT_DIR
module load cuda/12.4.1
export WANDB_PROJECT=timesearch-R-stage_1
export WANDB_NAME=$RUN_NAME
export LOG_PATH=${OUTPUT_DIR}/log.txt
# export DEBUG=true
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export TOKENIZERS_PARALLELISM=false
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

# MODEL_BASE=/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/SFT_Video_R1_cyt_60_frame/2026-01-27-14/Qwen2.5-VL-800-SFT
MODEL_BASE=/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/SFT_Video_R1_cyt_60_frame_epoch10_zero2/2026-02-08-12/Qwen2.5-VL-2000

# PREVIOUS_CKPT="/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/$RUN_NAME/2026-01-27-10/checkpoint-50"
# MODEL_BASE=/xuhongbo//shuimu.chen/Qwen2.5-VL-3B-Instruct
# MODEL_BASE=/data/shuimu.chen/Qwen2.5-VL-3B-Instruct
# MODEL_BASE=/xuhongbo/shuimu.chen/TimeSearch-R/Qwen2.5-VL-GRPO

    # --max_prompt_length 18000 \
    # --max_completion_length 16000 \
torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=${MASTER_PORT} \
    time_r1/train_VLLM_stage_1_split.py \
    --deepspeed /scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/scripts/zero3.json \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_BASE \
    --train_data_path $TRAIN_PATH \
    --video_folder $VIDEO_ROOT \
    --reward_func v11_valid_tool_split_S3_wandb_no_reasoning \
    --prompt_template v3 \
    --tool_name_list seek_video_frames \
    --max_interaction_turns 4 \
    --max_prompt_length 24000 \
    --max_completion_length 16000 \
    --max_completion_length_per_turn 256 \
    --total_video_tokens 10240 \
    --max_frames 734 \
    --min_per_frame_tokens 4 \
    --max_per_frame_tokens 192 \
    --num_generations 8 \
    --scale_rewards false \
    --beta 0.005 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 1 \
    --dataloader_num_workers 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --report_to wandb \
    --save_steps 1000 \
    --save_only_model true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --shuffle_dataset true \
    --replay_buffer_type dapo \
    --lr_scheduler_type "cosine" \
    --log_completions true \
    --use_counterfactual_reasoning true
