# The IP address from the above step
export SIGLIP_URL=grpc://127.0.0.1:51000
# 1. 确保这里列出了所有你想用的卡，比如 4 张卡
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=$PYTHONPATH:/data/shuimu.chen/TimeSearch-R

# 2. 将 nproc_per_node 改为 4 (与上面的卡数一致)
torchrun \
    --nproc_per_node=1 \
    --master_port=24137 \
    time_r1/reflect_tool_three.py \
    --input_path /data/shuimu.chen/LongVideoBench/LongVideoHaystack_all.json \
    --save_path /data/shuimu.chen/TimeSearch-R/tool_output_reflect \
    --data_root /data/shuimu.chen/LongVideoBench \
    --model_base /data/shuimu.chen/TimeSearch-R/Qwen2.5-VL-GRPO \
    --prompt_template v4 \
    --use_env True \
    --use_vllm True \
    --batch_size 1 \
    --num_data_workers 0 \
    --total_video_tokens 24000 \
    --max_frames 768 \
    --max_tokens 256