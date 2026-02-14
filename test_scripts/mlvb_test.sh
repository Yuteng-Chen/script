# The IP address from the above step
export SIGLIP_URL=grpc://127.0.0.1:52000

# source /scratch/prj0000000262-bucket/ocr/ec/env/bin/activate
# cd /scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/clip_as_service/server
# bash start.sh

# 【修改点 1】取消注释，并设置为 0 (明确使用第一张卡)
# 既然你有两张卡，设置为 0 或 1 都可以，这里用 0 比较稳妥
export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONPATH=$PYTHONPATH:/scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest

# 【修改点 2】注释掉这一行
# 既然你手动测试通过了，脚本里不要再重新加载 module，防止破坏现有环境
# module load cuda/12.4.1

# 运行命令
torchrun \
    --nproc_per_node=4 \
    --master_port=24137 \
    time_r1/reflect_tool_three.py \
    --input_path /scratch/prj0000000262-bucket/ocr/ec/MLVU_CYT_forder/json_test/MLVU_merged_all_cyt.json \
    --save_path /scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/test/MLVU/grpo_3000 \
    --data_root /data/shuimu.chen/LongVideoBench \
    --model_base /scratch/prj0000000262-bucket/ocr/ec/TimeSearch-R_latest/experiment/split_tool_s3_1e-6/2026-02-09-20/checkpoint-3000 \
    --prompt_template v3 \
    --use_env True \
    --use_vllm True \
    --batch_size 1 \
    --num_data_workers 4 \
    --total_video_tokens 24000 \
    --max_frames 768 \
    --max_tokens 256