"""
data2.xlsx 8:2 分割 + RoBERTa & Fast-DetectGPT 检测器评估
label: 0=AI, 1=人类

用于判断：
  1. 两个检测器在该数据集上方向是否正确（AUROC > 0.5）
  2. 训练集和测试集分布是否一致
  3. 是否适合作为训练数据集
"""

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ============================================================
# 0. 读取数据 & 8:2 分割
# ============================================================
print("="*60)
print("读取 data2.xlsx...")
df = pd.read_excel('data2.xlsx')
print(f"  总行数: {len(df)}")
print(f"  AI(0): {(df['label']==0).sum()}  人类(1): {(df['label']==1).sum()}")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"\n  训练集: {len(train_df)} 条  (AI={( train_df['label']==0).sum()}, 人类={(train_df['label']==1).sum()})")
print(f"  测试集: {len(test_df)}  条  (AI={(test_df['label']==0).sum()}, 人类={(test_df['label']==1).sum()})")

def compute_auroc(scores, labels):
    """labels: 0=AI, 1=人类；scores 越高越像 AI"""
    # y_true=1 表示 AI（正类），y_true=0 表示人类
    y_true = [1 if l == 0 else 0 for l in labels]
    auroc = roc_auc_score(y_true, scores)
    # 若 < 0.5，说明方向反了，自动翻转报告
    flipped = auroc < 0.5
    if flipped:
        auroc = 1 - auroc
    return auroc, flipped

def eval_split(name, split_df, rob_fn, fdgpt_fn, sample_n=300):
    """在一个分割上评估两个检测器"""
    # 采样（加速）
    ai_rows  = split_df[split_df['label']==0].sample(min(sample_n, (split_df['label']==0).sum()), random_state=42)
    hum_rows = split_df[split_df['label']==1].sample(min(sample_n, (split_df['label']==1).sum()), random_state=42)
    subset   = pd.concat([ai_rows, hum_rows]).reset_index(drop=True)
    texts  = subset['text'].tolist()
    labels = subset['label'].tolist()

    print(f"\n  [{name}] 评估 {len(subset)} 条（AI={len(ai_rows)}, 人类={len(hum_rows)}）")

    # RoBERTa
    rob_scores = rob_fn(texts)
    rob_auroc, rob_flip = compute_auroc(rob_scores, labels)
    print(f"    RoBERTa      AUROC={rob_auroc:.4f}  {'⚠️ 方向翻转' if rob_flip else '✅ 方向正确'}")

    # Fast-DetectGPT（负 loss 版）
    fdgpt_scores = fdgpt_fn(texts)
    fdgpt_auroc, fdgpt_flip = compute_auroc(fdgpt_scores, labels)
    print(f"    Fast-DGPT    AUROC={fdgpt_auroc:.4f}  {'⚠️ 方向翻转' if fdgpt_flip else '✅ 方向正确'}")

    return rob_auroc, fdgpt_auroc

# ============================================================
# 1. 加载 RoBERTa
# ============================================================
print("\n" + "="*60)
print("加载 RoBERTa...")
rob_pipe = pipeline(
    "text-classification",
    model="openai-community/roberta-large-openai-detector",
    device=0 if torch.cuda.is_available() else -1,
    truncation=True, max_length=512,
)
print("  ✅ RoBERTa 加载完成")

def score_roberta(texts, batch_size=32):
    scores = []
    for i in range(0, len(texts), batch_size):
        res = rob_pipe(texts[i:i+batch_size])
        for r in res:
            scores.append(r['score'] if r['label']=='FAKE' else 1.0-r['score'])
    return scores

# ============================================================
# 2. 加载 Fast-DetectGPT（负 loss，简单版）
# ============================================================
print("\n加载 Fast-DetectGPT (gpt-neo-2.7B)...")
fdgpt_name = "EleutherAI/gpt-neo-2.7B"
fdgpt_tok = AutoTokenizer.from_pretrained(fdgpt_name)
if fdgpt_tok.pad_token is None:
    fdgpt_tok.pad_token = fdgpt_tok.eos_token
fdgpt_model = AutoModelForCausalLM.from_pretrained(
    fdgpt_name, torch_dtype=torch.float16, device_map="auto"
).eval()
print("  ✅ Fast-DetectGPT 加载完成")

def score_fdgpt_negloss(texts):
    """负困惑度：loss 低 → 文本对模型可预测 → 越像AI → 高分"""
    scores = []
    for text in texts:
        try:
            inputs = fdgpt_tok(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(fdgpt_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                loss = fdgpt_model(**inputs, labels=inputs['input_ids']).loss
            scores.append(-loss.item())
        except Exception:
            scores.append(0.0)
    return scores

def score_fdgpt_entropy(texts):
    """条件熵：entropy 低 → 分布更尖锐 → 越像AI → 高分"""
    scores = []
    for text in texts:
        try:
            inputs = fdgpt_tok(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(fdgpt_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = fdgpt_model(**inputs).logits
            log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
            probs     = log_probs.exp()
            entropy   = -(probs * log_probs).sum(dim=-1).mean().item()
            scores.append(-entropy)
        except Exception:
            scores.append(0.0)
    return scores

# ============================================================
# 3. 评估
# ============================================================
print("\n" + "="*60)
print("评估 RoBERTa（负loss版 Fast-DetectGPT）")
print("="*60)

for split_name, split_df in [("训练集(80%)", train_df), ("测试集(20%)", test_df)]:
    eval_split(split_name, split_df, score_roberta, score_fdgpt_negloss)

print("\n" + "="*60)
print("评估 Fast-DetectGPT 条件熵版（对比）")
print("="*60)

for split_name, split_df in [("训练集(80%)", train_df), ("测试集(20%)", test_df)]:
    ai_rows  = split_df[split_df['label']==0].sample(min(300, (split_df['label']==0).sum()), random_state=42)
    hum_rows = split_df[split_df['label']==1].sample(min(300, (split_df['label']==1).sum()), random_state=42)
    subset   = pd.concat([ai_rows, hum_rows]).reset_index(drop=True)
    texts    = subset['text'].tolist()
    labels   = subset['label'].tolist()
    scores   = score_fdgpt_entropy(texts)
    auroc, flip = compute_auroc(scores, labels)
    print(f"\n  [{split_name}] Fast-DGPT 条件熵: AUROC={auroc:.4f}  {'⚠️ 方向翻转' if flip else '✅ 方向正确'}")

print("\n" + "="*60)
print("结论：")
print("  AUROC > 0.7 且方向正确 → 该检测器适合作为该数据集的训练信号")
print("  AUROC < 0.6 或方向翻转 → 不可靠，建议换检测器或数据集")
print("="*60)