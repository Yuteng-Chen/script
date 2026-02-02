# The IP address from the above step
export SIGLIP_URL=grpc://127.0.0.1:51000
# 1. 确保这里列出了所有你想用的卡，比如 4 张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=".:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# /xuhongbo/shuimu.chen/TimeSearch-R/experiment/timesearch-SFT_TEST/2026-01-15-07
# /xuhongbo/shuimu.chen/LongVideoBench/qwen2.5vl-Instruct-72b
# 不再使用 torchrun，直接用 python 启动
python time_r1/test_data.py \
    --input_path /xuhongbo/shuimu.chen/Video-R1-103K_sft_candidates_60k_id_xuhonbo_part2.json \
    --save_path /xuhongbo/shuimu.chen/TimeSearch-R/COT_Video-R1_60_2 \
    --data_root /xuhongbo/shuimu.chen/LongVideoBench/videos_480p_noaudio \
    --model_base /xuhongbo/shuimu.chen/LongVideoBench/qwen2.5vl-Instruct-72b \
    --prompt_template v3 \
    --use_env True \
    --use_vllm True \
    --tensor_parallel_size 8 \
    --batch_size 2 \
    --num_data_workers 0 \
    --total_video_tokens 24000 \
    --max_frames 60 \
    --max_tokens 192
