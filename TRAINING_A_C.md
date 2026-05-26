# A+C Training Route

This project now supports a new comparable training route:

- Baseline: original linear detector reward + GRPO
- A+C: DPO warmup, then GRPO with threshold/margin + worst-detector reward

## 1. Install

```bash
pip install -r requirements.txt
```

The local route requires `trl` and `peft`.

## 2. Common Server Variables

Set these to match the server layout:

```bash
cd /data/StealthRL-main/StealthRL-main
export STEALTHRL_PROJECT_ROOT="$PWD"
export STEALTHRL_BASE_MODEL="$PWD/Qwen2.5-0.5B-Instruct"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

## 3. Baseline Run

This keeps the old reward formula and does not use DPO warmup.

```bash
export STEALTHRL_REWARD_MODE=linear
export STEALTHRL_INIT_ADAPTER=
export STEALTHRL_OUTPUT_DIR=checkpoints/baseline-linear-grpo
export STEALTHRL_MAX_STEPS=100

python train_local.py
```

For a full run, increase `STEALTHRL_MAX_STEPS` and optionally set:

```bash
export STEALTHRL_NUM_GENERATIONS=8
export STEALTHRL_MAX_STEPS=2000
```

## 4. A+C Run

First build preference pairs and train the DPO warmup adapter:

```bash
export STEALTHRL_REWARD_MODE=margin_worst
export STEALTHRL_DPO_OUTPUT_DIR=checkpoints/stealthrl-qwen2.5-0.5b-dpo-warmup
export STEALTHRL_DPO_PREF_SAMPLES=1200
export STEALTHRL_DPO_CANDIDATES=4
export STEALTHRL_DPO_MAX_STEPS=300

python train_dpo_warmup.py
```

Then continue with GRPO from the DPO adapter:

```bash
export STEALTHRL_REWARD_MODE=margin_worst
export STEALTHRL_INIT_ADAPTER=checkpoints/stealthrl-qwen2.5-0.5b-dpo-warmup
export STEALTHRL_OUTPUT_DIR=checkpoints/stealthrl-qwen2.5-0.5b-margin-grpo
export STEALTHRL_MAX_STEPS=100

python train_local.py
```

For a full run:

```bash
export STEALTHRL_NUM_GENERATIONS=8
export STEALTHRL_MAX_STEPS=2000
python train_local.py
```

## 5. Evaluation

Evaluate any checkpoint by setting `STEALTHRL_LORA_PATH`.

```bash
export STEALTHRL_LORA_PATH="$PWD/checkpoints/stealthrl-qwen2.5-0.5b-margin-grpo"
export STEALTHRL_MAX_EVAL=500
python evaluate.py
```

For baseline comparison:

```bash
export STEALTHRL_LORA_PATH="$PWD/checkpoints/baseline-linear-grpo"
python evaluate.py
```

## 6. Submission Inference

```bash
export STEALTHRL_LORA_PATH="$PWD/checkpoints/stealthrl-qwen2.5-0.5b-margin-grpo"
python run_inference.py
```

## Reward Difference

Old linear reward:

```text
R = alpha * (1 - (0.6 * roberta + 0.4 * fdgpt)) + beta * semantic
```

New margin/worst reward:

```text
margin_i = sigmoid((threshold_i - detector_score_i) / tau)
R_det = (1 - worst_weight) * weighted_mean(margin_i) + worst_weight * min(margin_i)
R = alpha * R_det + semantic_barrier - length_penalty
```

`threshold_i` is calibrated from human MAGE samples at `target_fpr=0.01` and cached at:

```text
data/processed/detector_thresholds.json
```
