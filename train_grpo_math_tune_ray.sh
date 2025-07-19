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
export VLLM_ENFORCE_EAGER=0     # 禁用eager模式，解决InferenceMode冲突
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

# 添加Wandb run ID继承
# 检查是否存在之前的wandb run id
CHECKPOINT_DIR="/data/lishizheng/code/simpleRL-reason/results/checkpoints/verl-grpo_Qwen-2.5-0.5B_max_response2048_batch4_rollout2_klcoef0.0001_entcoef0.001_simplelr_math_35"
WANDB_ID_FILE="${CHECKPOINT_DIR}/wandb_run_id.txt"

if [ -f "$WANDB_ID_FILE" ]; then
  export WANDB_RUN_ID=$(cat "$WANDB_ID_FILE")
  echo "继续使用之前的Wandb运行ID: $WANDB_RUN_ID"
else
  echo "没有找到现有的Wandb ID，将创建新的运行"
fi

# 创建结果目录
# mkdir -p /data/lishizheng/code/simpleRL-reason/results/data
# mkdir -p /data/lishizheng/code/simpleRL-reason/results/models
# mkdir -p /data/lishizheng/code/simpleRL-reason/results/checkpoints
# mkdir -p /data/lishizheng/code/simpleRL-reason/results/logs

# Default values
TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=4  # 减少验证批次大小，避免OOM
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=3072
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=2
# per GPU
PPO_MICRO_BATCH_SIZE=1
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=2
ROLLOUT_N=4
KL_COEF=0.001
TOTAL_EPOCHS=10
DATASET_NAME=simplelr_math_35
ROLLOUT_GPU_MEMORY_UTIL=0.5  # 降低GPU内存使用率，避免OOM
MODEL_NAME=Qwen2.5-Math-7B
SAVE_FREQ=20
TEST_FREQ=5
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2

#############很重要，一直报错：#################
# train bs=4：
# rollout bs 报错：4，8
# 默认值改小，避免验证时内存占用过大
MICRO_ROLLOUT_BATCH_SIZE=16

###########################################

REMOVE_PREVIOUS_CKPT=False
# 增加验证超时设置，防止验证阶段无限等待
VALIDATION_TIMEOUT=600  # 10分钟超时，避免过长等待
# 增加VLLM内存优化参数
VLLM_MAX_NUM_BATCHED_TOKENS=4096  # 限制批处理中token数量，必须大于max_model_len （3072）
VLLM_ENFORCE_EAGER=False  # 禁用eager模式，避免InferenceMode冲突
DISABLE_VAL_GEN=False  # 是否禁用验证生成
MAX_VAL_SEQ_LEN=1024  # 限制验证生成的最大长度
# 添加验证集采样大小参数，限制验证样本数量
VAL_SAMPLE_SIZE=10  # 默认只验证10个样本，加快验证速度

generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --val_batch_size) suffix+="_valbatch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --ppo_micro_batch_size) shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --log_prob_micro_batch_size) suffix+="_logprobbatch$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --total_epochs) suffix+="_epochs$2"; shift 2 ;;
      --rollout_gpu_memory_util) shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      --remove_clip) suffix+="_remove_clip$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    suffix+="_$MODEL_NAME"
  fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi
  
  echo "$suffix"
}

echo "Arguments received: $@"

# Generate a unique suffix based on the input arguments
SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val_batch_size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --micro_rollout_batch_size) MICRO_ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --remove_clip) REMOVE_CLIP="$2"; shift 2 ;;
    --remove_previous_ckpt) REMOVE_PREVIOUS_CKPT="$2"; shift 2 ;;
    --validation_timeout) VALIDATION_TIMEOUT="$2"; shift 2 ;;
    --vllm_max_batched_tokens) VLLM_MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --disable_val_gen) DISABLE_VAL_GEN="$2"; shift 2 ;;
    --vllm_enforce_eager) VLLM_ENFORCE_EAGER="$2"; shift 2 ;;
    --max_val_seq_len) MAX_VAL_SEQ_LEN="$2"; shift 2 ;;
    --val_sample_size) VAL_SAMPLE_SIZE="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Val Batch Size: $VAL_BATCH_SIZE" 
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" 
echo "Max Response Length: $MAX_RESPONSE_LENGTH" 
echo "Learning Rate: $LEARNING_RATE" 
echo "--------------------------------"
echo "Rollout N: $ROLLOUT_N" 
echo "Micro Rollout Batch Size: $MICRO_ROLLOUT_BATCH_SIZE"
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" 
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE" 
echo "--------------------------------"
echo "KL Loss Coefficient: $KL_LOSS_COEF" 
echo "KL Loss Type: $KL_LOSS_TYPE" 
echo "Temperature: $TEMPERATURE" 
echo "KL Coefficient: $KL_COEF" 
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Dataset Name: $DATASET_NAME"
echo "Model Name: $MODEL_NAME"
echo "Remove Clip: $REMOVE_CLIP"
echo "Remove Previous Ckpt: $REMOVE_PREVIOUS_CKPT"
echo "Validation Timeout: $VALIDATION_TIMEOUT"
echo "VLLM Max Batched Tokens: $VLLM_MAX_NUM_BATCHED_TOKENS"
echo "Disable Val Gen: $DISABLE_VAL_GEN"
echo "VLLM Enforce Eager: $VLLM_ENFORCE_EAGER"
echo "Max Val Sequence Length: $MAX_VAL_SEQ_LEN"
echo "Val Sample Size: $VAL_SAMPLE_SIZE"
echo "LOG FILE PATH: $LOG_FILE_PATH"

max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)

export WORKING_DIR=$(pwd)

#ray job submit --address=${HEAD_IP}:${HEAD_PORT} \

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
  -- python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
  data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  actor_rollout_ref.model.path=$HDFS_MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
  actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
  actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS \
  actor_rollout_ref.rollout.micro_rollout_batch_size=$MICRO_ROLLOUT_BATCH_SIZE \
  actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  +critic.model.fsdp_config.model_dtype=bfloat16 \
  algorithm.kl_ctrl.kl_coef=$KL_COEF \
  critic.ppo_micro_batch_size_per_gpu=1 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$PROJECT_NAME \
  trainer.remove_previous_ckpt=$REMOVE_PREVIOUS_CKPT \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=$ARNOLD_WORKER_NUM \
  trainer.remove_clip=$REMOVE_CLIP \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
  trainer.total_epochs=$TOTAL_EPOCHS \
  ++trainer.validation_timeout=$VALIDATION_TIMEOUT \
  actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS \
  actor_rollout_ref.rollout.enable_chunked_prefill=$DISABLE_VAL_GEN \
  ++actor_rollout_ref.rollout.max_model_len=$MAX_VAL_SEQ_LEN \
  ++data.val_batch_size=$VAL_BATCH_SIZE \
  ++data.val_sample_size=$VAL_SAMPLE_SIZE 2>&1 | tee -a $LOG_FILE_PATH
