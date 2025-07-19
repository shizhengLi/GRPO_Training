#! /bin/bash

USER_ENV=`whoami`
#set -x
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1

# 设置环境变量
export PROJECT_NAME=verl_train
export WANDB_MODE=offline
export WANDB_OFFICIAL=1
export WANDB_DIR=/data/lishizheng/code/simpleRL-reason/results/wandb

export VLLM_MAX_MODEL_LEN=4096  # 限制VLLM最大序列长度
export CUDA_VISIBLE_DEVICES=0,1  # 明确指定使用的GPU
export TORCH_CUDA_MEMORY_STATS=1  # 启用更详细的内存统计
export TORCH_USE_CUDA_DSA=0       # 禁用CUDA设备端断言，避免与InferenceMode冲突
export OMP_NUM_THREADS=4          # 限制CPU线程数
export TORCH_DISTRIBUTED_DEBUG=INFO  # 减少分布式调试信息等级
export HDFS_DATA_PATH=/data/lishizheng/code/simpleRL-reason/results/data
export HDFS_MODEL_PATH=/data/lishizheng/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987
export HDFS_CHECKPOINT_PATH=/data/lishizheng/code/simpleRL-reason/results/checkpoints
export HDFS_LOG_PATH=/data/lishizheng/code/simpleRL-reason/results/logs
export RUN_NAME=verl-grpo
export ARNOLD_WORKER_NUM=1  # 1个节点，2张GPU



# 创建结果目录
mkdir -p /data/lishizheng/code/simpleRL-reason/results/data
mkdir -p /data/lishizheng/code/simpleRL-reason/results/models
mkdir -p /data/lishizheng/code/simpleRL-reason/results/checkpoints
mkdir -p /data/lishizheng/code/simpleRL-reason/results/logs

LOG_FILE_PATH="/data/lishizheng/code/simpleRL-reason/results/logs/$RUN_NAME.log"

export WORKING_DIR=$(pwd)

# 启动Ray任务
ray job submit --address=127.0.0.1:6379 \
  --entrypoint-num-cpus=1 \
  --runtime-env-json='{
        "working_dir": "'${WORKING_DIR}'",
        "env_vars": {
          "http_proxy": "",
          "https_proxy": "",
          "WANDB_MODE": "'${WANDB_MODE}'",
          "WANDB_DIR": "'${WANDB_DIR}'",
          "WANDB_OFFICIAL": "'${WANDB_OFFICIAL}'",
          "VLLM_MAX_MODEL_LEN": "'${VLLM_MAX_MODEL_LEN}'",
          "CUDA_VISIBLE_DEVICES": "'${CUDA_VISIBLE_DEVICES}'",
          "OMP_NUM_THREADS": "'${OMP_NUM_THREADS}'",
          "TORCH_CUDA_MEMORY_STATS": "'${TORCH_CUDA_MEMORY_STATS}'",
          "TORCH_USE_CUDA_DSA": "'${TORCH_USE_CUDA_DSA}'",
          "TORCH_DISTRIBUTED_DEBUG": "'${TORCH_DISTRIBUTED_DEBUG}'"
        }
    }' \
  -- python -m verl.trainer.main_ppo 2>&1 | tee -a $LOG_FILE_PATH
