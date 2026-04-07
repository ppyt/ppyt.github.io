#!/bin/bash
# run_all_rl.sh
# 顺序执行 PPO -> GRPO -> DAPO
# WandB 统一 project，不同 experiment_name
set -eo pipefail

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

export WANDB_API_KEY=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
export CUDA_VISIBLE_DEVICES=2,3,5,6
export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1

# 日志目录（带时间戳，方便追溯）
LOG_DIR="logs/rl_compare_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "日志目录: $LOG_DIR"

MODEL=/mnt/interaction-sh/qwen/Qwen2.5-7B
TRAIN_DATA=/mnt/z4/lumenpeng/verl/data/gsm8k/train.parquet
VAL_DATA=/mnt/z4/lumenpeng/verl/data/gsm8k/test.parquet
WANDB_PROJECT="verl_gsm8k_rl_compare"

COMMON_ARGS="
data.train_files=$TRAIN_DATA
data.val_files=$VAL_DATA
data.train_batch_size=256
data.max_prompt_length=512
data.max_response_length=512
actor_rollout_ref.model.path=$MODEL
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=64
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
algorithm.kl_ctrl.kl_coef=0.001
trainer.logger=['console','wandb']
trainer.val_before_train=False
trainer.n_gpus_per_node=4
trainer.nnodes=1
trainer.save_freq=-1
trainer.test_freq=10
trainer.total_epochs=15
trainer.project_name=$WANDB_PROJECT
"

echo "================== Start PPO ==================" | tee -a "$LOG_DIR/run.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PPO 开始训练" | tee -a "$LOG_DIR/run.log"
python3 -m verl.trainer.main_ppo \
$COMMON_ARGS \
critic.optim.lr=1e-5 \
critic.model.path=/mnt/interaction-sh/qwen/Qwen2.5-0.5B-Instruct \
critic.ppo_micro_batch_size_per_gpu=4 \
trainer.experiment_name='ppo_run' \
2>&1 | tee "$LOG_DIR/verl_ppo.log"
PPO_EXIT=${PIPESTATUS[0]}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PPO 训练结束, exit_code=$PPO_EXIT" | tee -a "$LOG_DIR/run.log"
if [ $PPO_EXIT -ne 0 ]; then echo "PPO 训练失败，终止执行" | tee -a "$LOG_DIR/run.log"; exit 1; fi
sleep 10 && nvidia-smi >> "$LOG_DIR/run.log" 2>&1

echo "================== Start GRPO ==================" | tee -a "$LOG_DIR/run.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO 开始训练" | tee -a "$LOG_DIR/run.log"
python3 -m verl.trainer.main_ppo \
$COMMON_ARGS \
algorithm.adv_estimator=grpo \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
trainer.experiment_name='grpo_run' \
2>&1 | tee "$LOG_DIR/verl_grpo.log"
GRPO_EXIT=${PIPESTATUS[0]}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO 训练结束, exit_code=$GRPO_EXIT" | tee -a "$LOG_DIR/run.log"
if [ $GRPO_EXIT -ne 0 ]; then echo "GRPO 训练失败，终止执行" | tee -a "$LOG_DIR/run.log"; exit 1; fi
sleep 10 && nvidia-smi >> "$LOG_DIR/run.log" 2>&1

echo "================== Start DAPO ==================" | tee -a "$LOG_DIR/run.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DAPO 开始训练" | tee -a "$LOG_DIR/run.log"
python3 -m verl.trainer.main_ppo \
$COMMON_ARGS \
algorithm.adv_estimator=grpo \
actor_rollout_ref.rollout.n=16 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
reward_model.reward_manager=dapo \
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
+reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
+reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
+reward_model.reward_kwargs.max_resp_len=1024 \
trainer.experiment_name='dapo_run' \
2>&1 | tee "$LOG_DIR/verl_dapo.log"
DAPO_EXIT=${PIPESTATUS[0]}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DAPO 训练结束, exit_code=$DAPO_EXIT" | tee -a "$LOG_DIR/run.log"
if [ $DAPO_EXIT -ne 0 ]; then echo "DAPO 训练失败" | tee -a "$LOG_DIR/run.log"; exit 1; fi

echo "================== All Done ==================" | tee -a "$LOG_DIR/run.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部训练完成" | tee -a "$LOG_DIR/run.log"
echo "日志保存在: $LOG_DIR"