"""
StealthRL 评估脚本（完整版，对齐论文）
数据集：MAGE test split + UCAS val.csv（双数据集）
检测器：RoBERTa / Fast-DetectGPT / Binoculars / MAGE detector（4个）
指标：AUROC、TPR@1%FPR、TPR@5%FPR、ASR、E5语义相似度
"""

import json, os, torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.environ.get("STEALTHRL_PROJECT_ROOT", os.getcwd())
BASE_MODEL = os.environ.get(
    "STEALTHRL_BASE_MODEL",
    os.path.join(PROJECT_ROOT, "Qwen2.5-0.5B-Instruct"),
)
LORA_PATH = os.environ.get(
    "STEALTHRL_LORA_PATH",
    os.path.join(PROJECT_ROOT, "checkpoints/stealthrl-qwen2.5-0.5b-margin-grpo"),
)
UCAS_VAL = os.environ.get(
    "STEALTHRL_UCAS_VAL",
    os.path.join(
        PROJECT_ROOT,
        "AI_Text Evasion/TextGenAdvTrack-2026Spring/UCAS_AISAD_TEXT-val.csv",
    ),
)
RESULTS    = "results"
MAX_EVAL   = int(os.environ.get("STEALTHRL_MAX_EVAL", "500"))
os.makedirs(RESULTS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
MODEL_DTYPE = torch.bfloat16 if USE_BF16 else DTYPE

# ============================================================
# 1. 加载两个数据集
# ============================================================
print("=" * 65)
print("加载数据集")
print("=" * 65)

# MAGE test split（label=0=AI，label=1=人类）
print("\n[数据集1] MAGE test split...")
ds = load_dataset('yaful/MAGE', split='test')
ai_mage    = [x['text'] for x in ds if x['label'] == 0][:MAX_EVAL]
human_mage = [x['text'] for x in ds if x['label'] == 1][:MAX_EVAL]
print(f"  AI: {len(ai_mage)} 条 | 人类: {len(human_mage)} 条")

# UCAS val.csv（label=0=机器生成，label=1=人类）
print("\n[数据集2] UCAS val.csv...")
df = pd.read_csv(UCAS_VAL)
df = df[df['text'].notna() & (df['text'].str.strip() != '')]
ai_ucas    = df[df['label'] == 0]['text'].tolist()[:MAX_EVAL]
human_ucas = df[df['label'] == 1]['text'].tolist()[:MAX_EVAL]
print(f"  AI: {len(ai_ucas)} 条 | 人类: {len(human_ucas)} 条")

# ============================================================
# 2. 加载四个检测器（对齐论文 Section 4.2）
# ============================================================
print("\n" + "=" * 65)
print("加载检测器")
print("=" * 65)

detectors = {}

# ── 检测器1：RoBERTa（训练集成，权重0.6）──────────────────
print("\n[1/4] RoBERTa（训练集成）...")
rob_tok = AutoTokenizer.from_pretrained("openai-community/roberta-large-openai-detector")
rob_model = AutoModelForSequenceClassification.from_pretrained(
    "openai-community/roberta-large-openai-detector",
    torch_dtype=DTYPE,
)
rob_model.to(DEVICE).eval()
def score_roberta(texts):
    scores = []
    for i in range(0, len(texts), 16):
        batch = list(texts)[i:i+16]
        inputs = rob_tok(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(rob_model(**inputs).logits, dim=-1)
        scores.extend(probs[:, 0].detach().float().cpu().tolist())
    return scores
detectors['RoBERTa'] = score_roberta
print("  ✅ RoBERTa 加载完成")

# ── 检测器2：Fast-DetectGPT（训练集成，权重0.4）───────────
print("\n[2/4] Fast-DetectGPT（训练集成）...")
try:
    # 论文用 gpt-neo-2.7B，显存不足可降级为 gpt2
    fdgpt_name = "EleutherAI/gpt-neo-2.7B"
    fdgpt_tok   = AutoTokenizer.from_pretrained(fdgpt_name)
    fdgpt_model = AutoModelForCausalLM.from_pretrained(
        fdgpt_name, torch_dtype=DTYPE, device_map="auto" if torch.cuda.is_available() else None
    ).eval()
    if fdgpt_tok.pad_token is None:
        fdgpt_tok.pad_token = fdgpt_tok.eos_token
    print(f"  ✅ {fdgpt_name} 加载完成")

    def score_fast_detectgpt(texts):
        scores = []
        for text in texts:
            try:
                inputs = fdgpt_tok(
                    text, return_tensors='pt',
                    truncation=True, max_length=512
                )
                inputs = {k: v.to(fdgpt_model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    loss = fdgpt_model(**inputs, labels=inputs['input_ids']).loss
                scores.append(-loss.item())   # 负困惑度，越高越像AI
            except Exception:
                scores.append(0.0)
        arr = torch.sigmoid(torch.tensor(scores, dtype=torch.float32) * 0.5)
        return arr.tolist()
    detectors['Fast-DetectGPT'] = score_fast_detectgpt

except Exception as e:
    print(f"  ⚠️ gpt-neo-2.7B 加载失败: {e}，降级为 gpt2")
    fdgpt_name  = "gpt2"
    fdgpt_tok   = AutoTokenizer.from_pretrained(fdgpt_name)
    fdgpt_model = AutoModelForCausalLM.from_pretrained(fdgpt_name).to(DEVICE).eval()
    if fdgpt_tok.pad_token is None:
        fdgpt_tok.pad_token = fdgpt_tok.eos_token

    def score_fast_detectgpt(texts):
        scores = []
        for text in texts:
            try:
                inputs = fdgpt_tok(text, return_tensors='pt',
                                   truncation=True, max_length=512)
                inputs = {k: v.to(fdgpt_model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    loss = fdgpt_model(**inputs, labels=inputs['input_ids']).loss
                scores.append(-loss.item())
            except Exception:
                scores.append(0.0)
        arr = torch.sigmoid(torch.tensor(scores, dtype=torch.float32) * 0.5)
        return arr.tolist()
    detectors['Fast-DetectGPT(gpt2)'] = score_fast_detectgpt

# ── 检测器3：Binoculars（论文 held-out，迁移测试）──────────
print("\n[3/4] Binoculars（held-out 检测器）...")
try:
    from transformers import AutoModelForCausalLM as CausalLM
    bino_tok    = AutoTokenizer.from_pretrained("gpt2-medium")
    bino_ref    = CausalLM.from_pretrained("gpt2-medium",
                      torch_dtype=DTYPE).to(DEVICE).eval()
    bino_score  = CausalLM.from_pretrained("gpt2-large",
                      torch_dtype=DTYPE).to(DEVICE).eval()
    if bino_tok.pad_token is None:
        bino_tok.pad_token = bino_tok.eos_token

    def score_binoculars(texts):
        scores = []
        for text in texts:
            try:
                inputs = bino_tok(text, return_tensors='pt',
                                  truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    ce  = bino_score(**inputs, labels=inputs['input_ids']).loss
                    ppl = bino_ref(**inputs,   labels=inputs['input_ids']).loss
                scores.append((ppl - ce).item())   # 正值=AI
            except Exception:
                scores.append(0.0)
        arr = torch.sigmoid(torch.tensor(scores, dtype=torch.float32))
        return arr.tolist()
    detectors['Binoculars'] = score_binoculars
    print("  ✅ Binoculars (gpt2-medium + gpt2-large) 加载完成")
except Exception as e:
    print(f"  ⚠️ Binoculars 加载失败: {e}")

# ── 检测器4：MAGE detector（论文 held-out，迁移测试）────────
print("\n[4/4] MAGE detector（held-out 检测器）...")
try:
    mage_tok = AutoTokenizer.from_pretrained('yaful/MAGE')
    mage_det = AutoModelForSequenceClassification.from_pretrained('yaful/MAGE')
    mage_det.config.id2label = {0: 'AI', 1: 'HUMAN'}
    if torch.cuda.is_available():
        mage_det = mage_det.cuda()
    mage_det.eval()

    def score_mage(texts):
        results = []
        for i in range(0, len(texts), 4):
            batch = texts[i:i+4]
            inputs = mage_tok(batch, return_tensors='pt',
                              truncation=True, max_length=4096, padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                probs = torch.softmax(mage_det(**inputs).logits, dim=-1)
            results.extend(probs[:, 0].cpu().tolist())  # index 0 = AI
        return results
    detectors['MAGE'] = score_mage
    print("  ✅ MAGE detector 加载完成")
except Exception as e:
    print(f"  ⚠️ MAGE detector 加载失败: {e}")

print(f"\n共加载 {len(detectors)} 个检测器: {list(detectors.keys())}")

# ============================================================
# 3. E5 语义相似度
# ============================================================
print("\n加载 E5 语义相似度模型...")
sem_model = SentenceTransformer("intfloat/e5-large-v2",
                                 device="cuda" if torch.cuda.is_available() else "cpu")

def e5_similarity(texts_a, texts_b, batch=64):
    emb_a = sem_model.encode(texts_a, normalize_embeddings=True,
                              convert_to_tensor=True, batch_size=batch)
    emb_b = sem_model.encode(texts_b, normalize_embeddings=True,
                              convert_to_tensor=True, batch_size=batch)
    return F.cosine_similarity(emb_a, emb_b).mean().item()

# ============================================================
# 4. 评估函数
# ============================================================
def compute_metrics(ai_scores, human_scores):
    y_true = [1]*len(ai_scores) + [0]*len(human_scores)
    y_pred = ai_scores + human_scores
    auroc  = roc_auc_score(y_true, y_pred)
    thr1   = np.percentile(human_scores, 99)   # 1%FPR
    thr5   = np.percentile(human_scores, 95)   # 5%FPR
    tpr1   = float(np.mean(np.array(ai_scores) >= thr1))
    tpr5   = float(np.mean(np.array(ai_scores) >= thr5))
    return {"auroc": auroc, "tpr_1": tpr1, "asr_1": 1-tpr1,
            "tpr_5": tpr5, "asr_5": 1-tpr5, "thr1": thr1}

def print_result(label, base, attacked):
    print(f"\n    ── {label} ──")
    print(f"    {'指标':<14} {'基线':<10} {'攻击后':<10} {'变化'}")
    print(f"    {'-'*44}")
    for k, name in [("auroc","AUROC"), ("tpr_1","TPR@1%FPR"),
                     ("asr_1","ASR@1%FPR"), ("tpr_5","TPR@5%FPR"),
                     ("asr_5","ASR@5%FPR")]:
        b, a = base[k], attacked[k]
        print(f"    {name:<14} {b:<10.4f} {a:<10.4f} {a-b:+.4f}")

# ============================================================
# 5. 基线评估
# ============================================================
print("\n" + "=" * 65)
print("计算基线（原始 AI 文本，无攻击）")
print("=" * 65)

baselines = {"mage": {}, "ucas": {}}
thresholds = {"mage": {}, "ucas": {}}

for det_name, det_fn in detectors.items():
    print(f"\n  {det_name}...")
    m = compute_metrics(det_fn(ai_mage), det_fn(human_mage))
    u = compute_metrics(det_fn(ai_ucas), det_fn(human_ucas))
    baselines["mage"][det_name] = m
    baselines["ucas"][det_name] = u
    thresholds["mage"][det_name] = m["thr1"]
    thresholds["ucas"][det_name] = u["thr1"]
    print(f"    MAGE test: AUROC={m['auroc']:.4f}  ASR@1%={m['asr_1']:.4f}")
    print(f"    UCAS val:  AUROC={u['auroc']:.4f}  ASR@1%={u['asr_1']:.4f}")

# ============================================================
# 6. 加载 StealthRL 模型并改写
# ============================================================
print("\n" + "=" * 65)
print(f"加载 StealthRL 模型")
print("=" * 65)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base_lm = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, trust_remote_code=True,
    torch_dtype=MODEL_DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
)
lora_model = PeftModel.from_pretrained(base_lm, LORA_PATH)
lora_model.eval()
print(f"  ✅ 模型加载完成: {LORA_PATH}")

def paraphrase(texts, batch=4):
    results = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        prompts = [
            f"Paraphrase the following text while preserving its meaning. "
            f"Write approximately {len(t.split())} words.\n\nOriginal: {t}\n\nParaphrase:"
            for t in chunk
        ]
        inputs = tok(prompts, return_tensors='pt', padding=True,
                     truncation=True, max_length=512).to(lora_model.device)
        with torch.no_grad():
            outs = lora_model.generate(
                **inputs, max_new_tokens=256,
                temperature=1.0, top_p=0.9, do_sample=True,
                pad_token_id=tok.pad_token_id,
            )
        for j, out in enumerate(outs):
            gen = tok.decode(out[inputs['input_ids'].shape[1]:],
                             skip_special_tokens=True).strip()
            results.append(gen if len(gen.split()) >= 5 else chunk[j])
        if (i // batch) % 25 == 0:
            print(f"  进度: {min(i+batch, len(texts))}/{len(texts)}")
    return results

print("\n生成 MAGE test 改写文本...")
para_mage = paraphrase(ai_mage)
e5_mage   = e5_similarity(ai_mage, para_mage)
print(f"  E5 语义相似度: {e5_mage:.4f}")

print("\n生成 UCAS val 改写文本...")
para_ucas = paraphrase(ai_ucas)
e5_ucas   = e5_similarity(ai_ucas, para_ucas)
print(f"  E5 语义相似度: {e5_ucas:.4f}")

# 保存改写样例
for fname, orig, para in [("samples_mage.jsonl", ai_mage, para_mage),
                           ("samples_ucas.jsonl", ai_ucas, para_ucas)]:
    with open(f"{RESULTS}/{fname}", 'w') as f:
        for o, p in zip(orig[:10], para[:10]):
            json.dump({"original": o, "paraphrase": p}, f, ensure_ascii=False)
            f.write('\n')

# ============================================================
# 7. 攻击后评估 + 对比
# ============================================================
print("\n计算攻击后得分...")
attacked = {"mage": {}, "ucas": {}}
for det_name, det_fn in detectors.items():
    attacked["mage"][det_name] = compute_metrics(
        det_fn(para_mage), det_fn(human_mage))
    attacked["ucas"][det_name] = compute_metrics(
        det_fn(para_ucas), det_fn(human_ucas))

# ============================================================
# 8. 最终输出（对齐论文表格格式）
# ============================================================
print("\n" + "=" * 65)
print("最终评估结果（对齐论文 Table 2/3 格式）")
print("=" * 65)

print("\n【训练集成检测器：RoBERTa + Fast-DetectGPT】")
print("【Held-out 检测器：Binoculars + MAGE（测试迁移效果）】")

for ds_name, ds_label in [("mage", "MAGE test split"), ("ucas", "UCAS val.csv")]:
    print(f"\n{'─'*65}")
    print(f"  数据集：{ds_label}")
    print(f"{'─'*65}")
    for det_name in detectors.keys():
        if det_name in baselines[ds_name]:
            print_result(det_name,
                         baselines[ds_name][det_name],
                         attacked[ds_name][det_name])

print(f"\n{'─'*65}")
print(f"  语义相似度（E5）")
print(f"{'─'*65}")
print(f"    MAGE test: {e5_mage:.4f}")
print(f"    UCAS val:  {e5_ucas:.4f}")

print("\n" + "=" * 65)
print("AUROC 越低、ASR 越高 = 攻击越成功")
print("Binoculars/MAGE 下降 = 攻击能迁移到未见过的检测器")

# 保存完整结果
res = {
    "model": LORA_PATH,
    "e5_similarity": {"mage_test": e5_mage, "ucas_val": e5_ucas},
    "baselines": baselines,
    "attacked":  attacked,
}
with open(f"{RESULTS}/eval_results_full.json", 'w') as f:
    json.dump(res, f, indent=2)
print(f"\n完整结果已保存: {RESULTS}/eval_results_full.json")
