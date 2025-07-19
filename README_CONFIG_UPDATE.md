# 配置更新说明

## 修改内容

1. **配置文件迁移**
   - 将`train_grpo_math_tune_ray.sh`中的训练参数移动到`verl/trainer/config/ppo_trainer.yaml`
   - 原始配置文件已备份为`verl/trainer/config/ppo_trainer.yaml.bak`
   - 新配置文件包含了所有必要的参数，特别是针对Qwen-2.5-0.5B模型的设置

2. **禁用eager模式**
   - 在配置文件中设置`actor_rollout_ref.rollout.enforce_eager: True`
   - 在环境变量中设置`VLLM_ENFORCE_EAGER=0`

3. **超时处理改进**
   - 修改了`ray_trainer.py`中的超时处理代码，使用简单的时间检查方法
   - 设置了验证超时参数`trainer.validation_timeout: 600`（10分钟）

4. **脚本简化**
   - 简化了`train_grpo_math_tune_ray.sh`脚本，只保留必要的环境变量设置和Ray启动命令
   - 添加了Wandb run ID继承功能，确保训练中断后可以继续使用相同的Wandb运行ID

## 使用方法

### 启动训练

直接运行脚本即可，无需额外参数：

```bash
RAY_ADDRESS='http://127.0.0.1:8265' proxychains4 bash train_grpo_math_tune_ray.sh
```

### 修改配置

如需修改训练参数，请直接编辑`verl/trainer/config/ppo_trainer.yaml`文件。主要参数包括：

- **数据相关**
  - `data.max_prompt_length`: 输入提示的最大长度（1024）
  - `data.max_response_length`: 响应的最大长度（2048）
  - `data.train_batch_size`: 训练批次大小（4）
  - `data.val_batch_size`: 验证批次大小（4）
  - `data.val_sample_size`: 验证样本数量（10）

- **模型相关**
  - `actor_rollout_ref.model.path`: 模型路径
  - `actor_rollout_ref.rollout.tensor_model_parallel_size`: 张量并行大小（2）
  - `actor_rollout_ref.rollout.n`: 每个样本生成的响应数量（2）

- **优化器相关**
  - `actor_rollout_ref.actor.optim.lr`: 学习率（5e-7）
  - `actor_rollout_ref.actor.kl_loss_coef`: KL损失系数（0.0001）
  - `actor_rollout_ref.actor.entropy_coeff`: 熵系数（0.001）

- **训练控制**
  - `trainer.total_epochs`: 总训练轮数（10）
  - `trainer.save_freq`: 保存检查点频率（5）
  - `trainer.test_freq`: 验证频率（5）
  - `trainer.validation_timeout`: 验证超时时间（600秒）

### 注意事项

1. 确保Ray服务已启动：`ray start --head --port=6379`
2. 确保环境变量正确设置，特别是GPU相关变量
3. 如果训练中断，脚本会自动尝试从上一个检查点恢复
4. 验证过程有超时保护，防止验证阶段无限等待 