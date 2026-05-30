# StealthRL 本地 MAGE + RoBERTa 版本

本项目当前只需要关注训练检测器为 **MAGE + RoBERTa** 的最新两组本地实验结果：

- **A+C 前**：`mage-rob-baseline-100-v1`
- **A+C 后**：`mage-rob-ac-100-v1`

仓库已清理掉旧论文评估管线、历史实验结果、临时日志和无关检测器组合。

## 主要目录

```text
Qwen2.5-0.5B-Instruct/                  本地基础模型
AI_Text Evasion/TextGenAdvTrack-2026Spring/
                                         UCAS 数据
data/processed/                         训练数据、评估数据和阈值缓存
checkpoints/mage-rob-baseline-100-v1/   A+C 前最新 checkpoint
checkpoints/mage-rob-ac-100-v1/         A+C 后最新 checkpoint
results/mage-rob-baseline-100-v1/       A+C 前评估结果
results/mage-rob-ac-100-v1/             A+C 后评估结果
train_local.py                          本地 GRPO 训练入口
train_dpo_warmup.py                     A+C 的 DPO warmup 入口
evaluate.py                             评估入口
run_inference.py                        推理生成 submission.csv
```

## 环境

```bash
pip install -r requirements.txt
```

服务器常用环境变量：

```bash
export STEALTHRL_PROJECT_ROOT="$PWD"
export STEALTHRL_BASE_MODEL="$PWD/Qwen2.5-0.5B-Instruct"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

## A+C 前训练

```bash
export STEALTHRL_REWARD_MODE=linear
export STEALTHRL_INIT_ADAPTER=
export STEALTHRL_OUTPUT_DIR=checkpoints/mage-rob-baseline-100-v1
export STEALTHRL_NUM_SAMPLES=10000
export STEALTHRL_BATCH_SIZE=8
export STEALTHRL_NUM_GENERATIONS=8
export STEALTHRL_MAX_STEPS=100
export STEALTHRL_ROBERTA_BATCH=32
export STEALTHRL_SEMANTIC_BATCH=512
export STEALTHRL_MAX_COMPLETION=768
export STEALTHRL_TOKEN_MAX=350
python train_local.py
```

## A+C 后训练

如需重新生成 DPO warmup adapter：

```bash
export STEALTHRL_REWARD_MODE=margin_worst
export STEALTHRL_DPO_OUTPUT_DIR=checkpoints/stealthrl-qwen2.5-0.5b-dpo-warmup
export STEALTHRL_DPO_PREF_SAMPLES=1200
export STEALTHRL_DPO_CANDIDATES=4
export STEALTHRL_DPO_MAX_STEPS=300
python train_dpo_warmup.py
```

从 warmup adapter 继续 GRPO：

```bash
export STEALTHRL_REWARD_MODE=margin_worst
export STEALTHRL_INIT_ADAPTER=checkpoints/stealthrl-qwen2.5-0.5b-dpo-warmup
export STEALTHRL_OUTPUT_DIR=checkpoints/mage-rob-ac-100-v1
export STEALTHRL_NUM_SAMPLES=10000
export STEALTHRL_BATCH_SIZE=8
export STEALTHRL_NUM_GENERATIONS=8
export STEALTHRL_MAX_STEPS=100
export STEALTHRL_ROBERTA_BATCH=32
export STEALTHRL_SEMANTIC_BATCH=512
export STEALTHRL_MAX_COMPLETION=768
export STEALTHRL_TOKEN_MAX=350
python train_local.py
```

## 评估

A+C 前：

```bash
mkdir -p results/mage-rob-baseline-100-v1
export STEALTHRL_LORA_PATH="$PWD/checkpoints/mage-rob-baseline-100-v1"
export STEALTHRL_MAX_EVAL=200
python evaluate.py 2>&1 | tee results/mage-rob-baseline-100-v1/eval.log
cp results/eval_results_full.json results/mage-rob-baseline-100-v1/eval_results_full.json
cp results/samples_mage.jsonl results/mage-rob-baseline-100-v1/samples_mage.jsonl
cp results/samples_ucas.jsonl results/mage-rob-baseline-100-v1/samples_ucas.jsonl
```

A+C 后：

```bash
mkdir -p results/mage-rob-ac-100-v1
export STEALTHRL_LORA_PATH="$PWD/checkpoints/mage-rob-ac-100-v1"
export STEALTHRL_MAX_EVAL=200
python evaluate.py 2>&1 | tee results/mage-rob-ac-100-v1/eval.log
cp results/eval_results_full.json results/mage-rob-ac-100-v1/eval_results_full.json
cp results/samples_mage.jsonl results/mage-rob-ac-100-v1/samples_mage.jsonl
cp results/samples_ucas.jsonl results/mage-rob-ac-100-v1/samples_ucas.jsonl
```

## 推理

默认使用 A+C 后模型：

```bash
python run_inference.py
```

输出文件：

```text
submission.csv
```

## 奖励设置

训练关注的检测器组合：

```text
MAGE + RoBERTa
```

baseline 对应 A+C 前；`margin_worst` + DPO warmup 对应 A+C 后。
