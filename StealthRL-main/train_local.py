"""
StealthRL 训练脚本
模型：Qwen2.5-0.5B-Instruct
奖励：MAGE_detector(0.6) + RoBERTa(0.4) + E5语义相似度
算法：GRPO + LoRA

改动说明：
  - 奖励集成：原 RoBERTa(0.6)+FastDetectGPT(0.4) → MAGE(0.6)+RoBERTa(0.4)
  - MAGE detector 信号更强（AUROC=0.99），方向稳定
  - RoBERTa 降为 0.4 权重，作为辅助信号
  - Binoculars + Fast-DetectGPT 作为 held-out 评估迁移效果
  - token_max=350，max_completion=768，截断惩罚=0.3
  - 随机抽样 seed=42，10000条
"""

import os, json, torch, random
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification, pipeline
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
import shutil

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.environ.get("STEALTHRL_PROJECT_ROOT", os.getcwd())
USE_CUDA = torch.cuda.is_available()
USE_BF16 = USE_CUDA and torch.cuda.is_bf16_supported()
MODEL_DTYPE = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_CUDA else torch.float32)

CFG = {
    "base_model":   os.environ.get(
        "STEALTHRL_BASE_MODEL",
        os.path.join(PROJECT_ROOT, "Qwen2.5-0.5B-Instruct"),
    ),
    "output_dir":   os.environ.get(
        "STEALTHRL_OUTPUT_DIR",
        "checkpoints/stealthrl-qwen2.5-0.5b-mage-grpo",
    ),
    "init_adapter": os.environ.get(
        "STEALTHRL_INIT_ADAPTER", ""
    ),
    "train_data":   "data/processed/train.jsonl",

    # LoRA
    "lora_rank":    16,
    "lora_alpha":   32,
    "lora_dropout": 0.05,

    # 训练超参
    "learning_rate":         float(os.environ.get("STEALTHRL_LR", "2.8e-4")),
    "batch_size":            int(os.environ.get("STEALTHRL_BATCH_SIZE", "8")),
    "num_generations":       int(os.environ.get("STEALTHRL_NUM_GENERATIONS", "8")),
    "max_completion_length": int(os.environ.get("STEALTHRL_MAX_COMPLETION", "768")),
    "max_steps":             int(os.environ.get("STEALTHRL_MAX_STEPS", "2000")),
    "kl_coef":               0.05,

    # 奖励权重（新集成：MAGE 0.6 + RoBERTa 0.4）
    "alpha":        1.0,   # 检测逃逸权重
    "beta":         0.1,   # 语义相似度权重
    "mage_weight":  0.6,   # MAGE detector 权重
    "rob_weight":   0.4,   # RoBERTa 权重

    # 截断惩罚
    "truncation_penalty": float(os.environ.get("STEALTHRL_TRUNCATION_PENALTY", "0.3")),

    # 数据
    "num_samples": int(os.environ.get("STEALTHRL_NUM_SAMPLES", "10000")),
    "token_min":   int(os.environ.get("STEALTHRL_TOKEN_MIN", "100")),
    "token_max":   int(os.environ.get("STEALTHRL_TOKEN_MAX", "350")),
    "random_seed": 42,
}

os.makedirs(CFG["output_dir"], exist_ok=True)
_SENTENCE_ENDINGS = {".", "!", "?", '"', "\u201d", "'"}

def make_prompt(text):
    return f"Paraphrase the following text while preserving its meaning: {text}"

# ============================================================
# 0. 准备训练数据
# ============================================================
if not os.path.exists(CFG["train_data"]):
    print(f"[0/5] 准备训练数据（seed={CFG['random_seed']}，{CFG['num_samples']} 条）...")
    os.makedirs("data/processed", exist_ok=True)

    filter_tok = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=True)
    ds = load_dataset('yaful/MAGE', split='train')
    ai_samples = [x for x in ds if x['label'] == 0]
    print(f"  AI 样本总数: {len(ai_samples)}")

    filtered = []
    for x in ai_samples:
        n_tok = len(filter_tok.encode(x['text']))
        if CFG["token_min"] <= n_tok <= CFG["token_max"]:
            filtered.append(x)
    print(f"  过滤后共 {len(filtered)} 条")

    random.seed(CFG["random_seed"])
    random.shuffle(filtered)
    filtered = filtered[:CFG["num_samples"]]
    print(f"  随机抽取（seed={CFG['random_seed']}）: {len(filtered)} 条")

    split = int(len(filtered) * 0.9)
    for name, data in [('train', filtered[:split]), ('eval', filtered[split:])]:
        with open(f"data/processed/{name}.jsonl", 'w', encoding='utf-8') as f:
            for x in data:
                text = x['text']
                json.dump({
                    'prompt':        make_prompt(text),
                    'original_text': text,
                    'label':         0
                }, f, ensure_ascii=False)
                f.write('\n')
        print(f"  {name}.jsonl: {len(data)} 条")

    shutil.copy("data/processed/eval.jsonl", "data/processed/esl_validation.jsonl")
    shutil.copy("data/processed/eval.jsonl", "data/processed/native_validation.jsonl")
    print("  数据准备完成")
else:
    print("[0/5] 训练数据已存在，跳过")

# ============================================================
# 1. 加载策略模型
# ============================================================
print(f"[1/5] 加载模型: {CFG['base_model']}")
tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"], trust_remote_code=True,
        torch_dtype=MODEL_DTYPE, device_map="auto",
        attn_implementation="sdpa",
    )
    print("  ✅ SDPA 已启用")
except Exception:
    model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"], trust_remote_code=True,
        torch_dtype=MODEL_DTYPE, device_map="auto",
    )

eos_ids = [tokenizer.eos_token_id]
for token in ["<|im_end|>", "<|endoftext|>"]:
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid and tid != tokenizer.unk_token_id:
        eos_ids.append(tid)
eos_ids = list(set(filter(None, eos_ids)))
model.config.eos_token_id = eos_ids
model.generation_config.eos_token_id = eos_ids
model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"  EOS token IDs: {eos_ids}")

if CFG["init_adapter"] and os.path.exists(CFG["init_adapter"]):
    print(f"  加载 adapter: {CFG['init_adapter']}")
    model = PeftModel.from_pretrained(model, CFG["init_adapter"], is_trainable=True)
else:
    if CFG["init_adapter"]:
        print(f"  未找到 adapter，从 base 初始化 LoRA")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CFG["lora_rank"], lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("  ✅ TF32 已启用")

# ============================================================
# 2. 加载检测器（新集成：MAGE × 0.6 + RoBERTa × 0.4）
# ============================================================
print("[2/5] 加载检测器（MAGE×0.6 + RoBERTa×0.4）...")

# MAGE detector（权重 0.6，主信号）
mage_tok   = AutoTokenizer.from_pretrained("yaful/MAGE")
mage_model = AutoModelForSequenceClassification.from_pretrained(
    "yaful/MAGE"
).cuda().eval()
print("  ✅ MAGE detector 加载完成")

# RoBERTa（权重 0.4，辅助信号）
rob_pipe = pipeline(
    "text-classification",
    model="openai-community/roberta-large-openai-detector",
    device=0 if torch.cuda.is_available() else -1,
    truncation=True, max_length=512,
)
print("  ✅ RoBERTa 加载完成")


def score_mage(texts, batch_size=8):
    """
    MAGE detector：返回 AI 概率（index 0 = AI）
    Longformer，支持长文本，对 MAGE 数据集 AUROC=0.99
    """
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = mage_tok(
            batch, return_tensors='pt', truncation=True,
            max_length=4096, padding=True
        )
        inputs = {k: v.to(mage_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = mage_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, 0].cpu().tolist())  # index 0 = AI
    return scores


def score_roberta(texts, batch_size=16):
    """RoBERTa：返回 AI 概率（FAKE label）"""
    scores = []
    for i in range(0, len(texts), batch_size):
        res = rob_pipe(texts[i:i+batch_size])
        for r in res:
            scores.append(r['score'] if r['label'] == 'FAKE' else 1.0 - r['score'])
    return scores


def get_ai_prob(texts):
    """
    新集成奖励公式：
    pens(y) = 0.6 * P_MAGE(y) + 0.4 * P_RoBERTa(y)
    """
    mage  = score_mage(texts)
    rob   = score_roberta(texts)
    return [CFG["mage_weight"] * m + CFG["rob_weight"] * r
            for m, r in zip(mage, rob)]


# ============================================================
# 3. 加载 E5 语义相似度模型
# ============================================================
print("[3/5] 加载 E5 语义相似度模型...")
try:
    sem_model = SentenceTransformer(
        "intfloat/e5-large-v2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("  使用 intfloat/e5-large-v2")
except Exception:
    sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  降级使用 all-MiniLM-L6-v2")

# ============================================================
# 4. 加载数据 + 预计算 E5 缓存
# ============================================================
print("[4/5] 加载训练数据 + 预计算 E5 缓存...")
train_data = []
with open(CFG["train_data"], encoding='utf-8') as f:
    for line in f:
        train_data.append(json.loads(line.strip()))

train_dataset = Dataset.from_list(train_data)
print(f"  训练集: {len(train_dataset)} 条")

EMBED_CACHE = {}
all_orig = list(set(d['original_text'] for d in train_data))
for i in range(0, len(all_orig), 256):
    batch = all_orig[i:i+256]
    embs  = sem_model.encode(batch, convert_to_tensor=True,
                              normalize_embeddings=True, show_progress_bar=False)
    for text, emb in zip(batch, embs):
        EMBED_CACHE[text] = emb.cpu()
print(f"  E5 缓存完成：{len(EMBED_CACHE)} 条")


def get_similarity(texts_a, texts_b):
    emb_a = torch.stack([
        EMBED_CACHE.get(
            t, sem_model.encode(t, convert_to_tensor=True,
                                normalize_embeddings=True).cpu()
        ) for t in texts_a
    ])
    emb_b = sem_model.encode(texts_b, convert_to_tensor=True,
                              normalize_embeddings=True)
    if torch.cuda.is_available():
        emb_a = emb_a.cuda()
        if emb_b.device.type != 'cuda':
            emb_b = emb_b.cuda()
    return F.cosine_similarity(emb_a, emb_b).cpu().tolist()


# ============================================================
# 奖励函数
# R(x,y) = α*(1 - pens) + β*Rsem - truncation_penalty
# pens   = 0.6*P_MAGE + 0.4*P_RoBERTa
# α=1.0, β=0.1, KL λ=0.05（GRPO内置）
# ============================================================
def reward_fn(prompts, completions, original_text=None, **kwargs):
    rewards = [None] * len(completions)
    valid_completions, valid_originals, valid_idx = [], [], []

    for i, (comp, orig) in enumerate(
        zip(completions, original_text or ['']*len(completions))
    ):
        comp_clean = comp.strip()
        if len(comp_clean.split()) >= 5:
            valid_completions.append(comp_clean)
            valid_originals.append(str(orig))
            valid_idx.append(i)
        else:
            rewards[i] = -1.0

    if valid_completions:
        try:
            ai_probs = get_ai_prob(valid_completions)
        except Exception as e:
            print(f"  检测器报错: {e}")
            ai_probs = [0.5] * len(valid_completions)
        try:
            sim_scores = get_similarity(valid_originals, valid_completions)
        except Exception as e:
            print(f"  语义模型报错: {e}")
            sim_scores = [0.8] * len(valid_completions)

        for idx, ai_p, sim, comp in zip(
            valid_idx, ai_probs, sim_scores, valid_completions
        ):
            r = CFG["alpha"] * (1.0 - ai_p) + CFG["beta"] * sim
            if comp[-1] not in _SENTENCE_ENDINGS:
                r -= CFG["truncation_penalty"]
            rewards[idx] = r

    return rewards


# ============================================================
# 5. GRPO 训练
# ============================================================
print("[5/5] 初始化 GRPO 训练器...")
grpo_args = GRPOConfig(
    output_dir=CFG["output_dir"],
    learning_rate=CFG["learning_rate"],
    per_device_train_batch_size=CFG["batch_size"],
    num_generations=CFG["num_generations"],
    max_completion_length=CFG["max_completion_length"],
    temperature=1.0,
    top_p=0.9,
    max_steps=CFG["max_steps"],
    logging_steps=10,
    save_steps=500,
    save_total_limit=10,
    beta=CFG["kl_coef"],
    bf16=USE_BF16,
    fp16=USE_CUDA and not USE_BF16,
    report_to="none",
    gradient_accumulation_steps=1,
    warmup_steps=100,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    torch_compile=False,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    resume_from_checkpoint=True,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_args,
    processing_class=tokenizer,
    reward_funcs=[reward_fn],
    train_dataset=train_dataset,
)

print("\n开始训练...")
print(f"  模型：Qwen2.5-0.5B-Instruct")
print(f"  数据：MAGE train，seed=42 随机抽取 {CFG['num_samples']} 条")
print(f"  奖励：R = 1.0*(1-pens) + 0.1*Rsem")
print(f"  pens：0.6*MAGE + 0.4*RoBERTa（新集成）")
print(f"  截断惩罚：{CFG['truncation_penalty']}")
print(f"  max_completion：{CFG['max_completion_length']} tokens")
print(f"  总步数：{CFG['max_steps']} 步")
print(f"  init_adapter：{CFG['init_adapter'] or '无'}")
print("=" * 60)
trainer.train()

trainer.save_model(CFG["output_dir"])
tokenizer.save_pretrained(CFG["output_dir"])
print(f"\n✅ 训练完成，模型已保存到: {CFG['output_dir']}")