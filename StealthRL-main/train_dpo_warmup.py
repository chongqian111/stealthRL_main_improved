"""
DPO warmup for the local 0.5B StealthRL workflow.

Stage C:
1. Generate several paraphrase candidates for each MAGE training prompt.
2. Score them with the same margin/worst-detector reward used by GRPO.
3. Build (prompt, chosen, rejected) preference pairs.
4. Train a LoRA adapter with DPO.

After this finishes, run train_local.py. By default train_local.py will load:
    checkpoints/stealthrl-qwen2.5-0.5b-dpo-warmup
as its initial adapter and continue with GRPO.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from stealthrl.local_reward import LocalRewardScorer, make_prompt


PROJECT_ROOT = os.environ.get("STEALTHRL_PROJECT_ROOT", os.getcwd())
USE_CUDA = torch.cuda.is_available()
USE_BF16 = USE_CUDA and torch.cuda.is_bf16_supported()
MODEL_DTYPE = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_CUDA else torch.float32)

CFG: dict[str, Any] = {
    "base_model": os.environ.get(
        "STEALTHRL_BASE_MODEL",
        os.path.join(PROJECT_ROOT, "Qwen2.5-0.5B-Instruct"),
    ),
    "output_dir": os.environ.get(
        "STEALTHRL_DPO_OUTPUT_DIR",
        "checkpoints/stealthrl-qwen2.5-0.5b-dpo-warmup",
    ),
    "train_data": os.environ.get("STEALTHRL_TRAIN_DATA", "data/processed/train.jsonl"),
    "preference_data": os.environ.get(
        "STEALTHRL_PREF_DATA",
        "data/processed/dpo_preferences_margin.jsonl",
    ),
    "num_samples": int(os.environ.get("STEALTHRL_NUM_SAMPLES", "5000")),
    "token_min": int(os.environ.get("STEALTHRL_TOKEN_MIN", "100")),
    "token_max": int(os.environ.get("STEALTHRL_TOKEN_MAX", "500")),
    "max_preference_samples": int(os.environ.get("STEALTHRL_DPO_PREF_SAMPLES", "1200")),
    "num_candidates": int(os.environ.get("STEALTHRL_DPO_CANDIDATES", "4")),
    "candidate_batch_size": int(os.environ.get("STEALTHRL_DPO_GEN_BATCH", "2")),
    "max_new_tokens": int(os.environ.get("STEALTHRL_DPO_MAX_NEW_TOKENS", "384")),
    "temperature": float(os.environ.get("STEALTHRL_DPO_TEMPERATURE", "1.0")),
    "top_p": float(os.environ.get("STEALTHRL_DPO_TOP_P", "0.9")),
    "min_reward_gap": float(os.environ.get("STEALTHRL_DPO_MIN_GAP", "0.05")),
    "lora_rank": int(os.environ.get("STEALTHRL_LORA_RANK", "16")),
    "lora_alpha": int(os.environ.get("STEALTHRL_LORA_ALPHA", "32")),
    "lora_dropout": float(os.environ.get("STEALTHRL_LORA_DROPOUT", "0.05")),
    "learning_rate": float(os.environ.get("STEALTHRL_DPO_LR", "5e-5")),
    "batch_size": int(os.environ.get("STEALTHRL_DPO_BATCH", "4")),
    "grad_accum": int(os.environ.get("STEALTHRL_DPO_GRAD_ACCUM", "4")),
    "max_steps": int(os.environ.get("STEALTHRL_DPO_MAX_STEPS", "300")),
    "warmup_steps": int(os.environ.get("STEALTHRL_DPO_WARMUP", "30")),
    "save_steps": int(os.environ.get("STEALTHRL_DPO_SAVE_STEPS", "100")),
    "dpo_beta": float(os.environ.get("STEALTHRL_DPO_BETA", "0.1")),
    "max_length": int(os.environ.get("STEALTHRL_DPO_MAX_LENGTH", "1024")),
    "max_prompt_length": int(os.environ.get("STEALTHRL_DPO_MAX_PROMPT_LENGTH", "512")),
    # Shared reward config.
    "alpha": 1.0,
    "beta": float(os.environ.get("STEALTHRL_SEM_BETA", "0.1")),
    "roberta_weight": float(os.environ.get("STEALTHRL_ROBERTA_WEIGHT", "0.6")),
    "fdgpt_weight": float(os.environ.get("STEALTHRL_FDGPT_WEIGHT", "0.4")),
    "reward_mode": os.environ.get("STEALTHRL_REWARD_MODE", "margin_worst"),
    "roberta_model": os.environ.get(
        "STEALTHRL_ROBERTA_MODEL",
        "openai-community/roberta-large-openai-detector",
    ),
    "fdgpt_model": os.environ.get(
        "STEALTHRL_FDGPT_MODEL",
        "EleutherAI/gpt-neo-2.7B",
    ),
    "threshold_cache": os.environ.get(
        "STEALTHRL_THRESHOLD_CACHE",
        "data/processed/detector_thresholds.json",
    ),
    "threshold_dataset_split": os.environ.get("STEALTHRL_THRESHOLD_SPLIT", "test"),
    "threshold_num_human": int(os.environ.get("STEALTHRL_THRESHOLD_HUMAN", "1000")),
    "target_fpr": float(os.environ.get("STEALTHRL_TARGET_FPR", "0.01")),
    "margin_tau": float(os.environ.get("STEALTHRL_MARGIN_TAU", "0.05")),
    "worst_detector_weight": float(os.environ.get("STEALTHRL_WORST_WEIGHT", "0.5")),
    "semantic_model": os.environ.get("STEALTHRL_SEMANTIC_MODEL", "intfloat/e5-large-v2"),
    "semantic_floor": float(os.environ.get("STEALTHRL_SEMANTIC_FLOOR", "0.85")),
    "semantic_fail_penalty": float(os.environ.get("STEALTHRL_SEMANTIC_FAIL", "0.5")),
    "semantic_fallback": 0.8,
    "length_ratio_min": float(os.environ.get("STEALTHRL_LENGTH_MIN", "0.70")),
    "length_ratio_max": float(os.environ.get("STEALTHRL_LENGTH_MAX", "1.30")),
    "length_penalty_weight": float(os.environ.get("STEALTHRL_LENGTH_PENALTY", "0.35")),
    "min_completion_words": int(os.environ.get("STEALTHRL_MIN_WORDS", "5")),
    "invalid_reward": -1.0,
    "roberta_batch_size": int(os.environ.get("STEALTHRL_ROBERTA_BATCH", "16")),
    "fdgpt_batch_size": int(os.environ.get("STEALTHRL_FDGPT_BATCH", "1")),
    "semantic_batch_size": int(os.environ.get("STEALTHRL_SEMANTIC_BATCH", "128")),
}


def prepare_train_data(tokenizer: Any) -> None:
    if os.path.exists(CFG["train_data"]):
        print(f"[data] Existing train data found: {CFG['train_data']}")
        return

    print(f"[data] Preparing MAGE train data: {CFG['num_samples']} samples")
    os.makedirs("data/processed", exist_ok=True)
    ds = load_dataset("yaful/MAGE", split="train")
    ai_samples = [row for row in ds if row["label"] == 0]

    filtered = []
    for row in ai_samples:
        n_tok = len(tokenizer.encode(row["text"]))
        if CFG["token_min"] <= n_tok <= CFG["token_max"]:
            filtered.append(row)
        if len(filtered) >= CFG["num_samples"]:
            break

    split = int(len(filtered) * 0.9)
    for name, rows in (("train", filtered[:split]), ("eval", filtered[split:])):
        path = f"data/processed/{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                text = row["text"]
                json.dump(
                    {
                        "prompt": make_prompt(text),
                        "original_text": text,
                        "label": 0,
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
        print(f"  {path}: {len(rows)}")

    shutil.copy("data/processed/eval.jsonl", "data/processed/esl_validation.jsonl")
    shutil.copy("data/processed/eval.jsonl", "data/processed/native_validation.jsonl")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_generation(tokenizer: Any, output_ids: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()


def generate_candidates(model: Any, tokenizer: Any, prompts: list[str]) -> list[list[str]]:
    all_candidates: list[list[str]] = []
    model.eval()
    tokenizer.padding_side = "left"

    for i in range(0, len(prompts), CFG["candidate_batch_size"]):
        batch_prompts = prompts[i : i + CFG["candidate_batch_size"]]
        expanded = []
        owner = []
        for local_idx, prompt in enumerate(batch_prompts):
            for _ in range(CFG["num_candidates"]):
                expanded.append(prompt)
                owner.append(local_idx)

        inputs = tokenizer(
            expanded,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG["max_prompt_length"],
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CFG["max_new_tokens"],
                temperature=CFG["temperature"],
                top_p=CFG["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        grouped = [[] for _ in batch_prompts]
        for row_idx, output in enumerate(outputs):
            text = extract_generation(tokenizer, output, prompt_len)
            if text:
                grouped[owner[row_idx]].append(text)
        all_candidates.extend(grouped)

        done = min(i + CFG["candidate_batch_size"], len(prompts))
        if done % 50 == 0 or done == len(prompts):
            print(f"[prefs] Generated candidates: {done}/{len(prompts)}")

    return all_candidates


def build_preference_data() -> None:
    pref_path = Path(CFG["preference_data"])
    if pref_path.exists():
        print(f"[prefs] Existing preference data found: {pref_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepare_train_data(tokenizer)

    rows = load_jsonl(CFG["train_data"])[: CFG["max_preference_samples"]]
    prompts = [row["prompt"] for row in rows]
    originals = [row["original_text"] for row in rows]

    print(f"[prefs] Loading generator model: {CFG['base_model']}")
    gen_model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"],
        trust_remote_code=True,
        torch_dtype=MODEL_DTYPE,
        device_map="auto" if USE_CUDA else None,
    )
    candidates_by_prompt = generate_candidates(gen_model, tokenizer, prompts)
    del gen_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scorer = LocalRewardScorer(CFG, tokenizer=tokenizer)
    scorer.load_detectors()
    scorer.load_semantic_model()
    if CFG["reward_mode"] != "linear":
        scorer.load_or_calibrate_thresholds()
    scorer.precompute_original_embeddings(
        [{"original_text": original} for original in originals]
    )

    pref_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with pref_path.open("w", encoding="utf-8") as f:
        for row, original, candidates in zip(rows, originals, candidates_by_prompt, strict=True):
            unique_candidates = list(dict.fromkeys([c.strip() for c in candidates if c.strip()]))
            if len(unique_candidates) < 2:
                continue
            rewards, details = scorer.compute_rewards(
                [original] * len(unique_candidates),
                unique_candidates,
                return_details=True,
            )
            ranked = sorted(
                zip(unique_candidates, rewards, details, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
            chosen, chosen_reward, chosen_details = ranked[0]
            rejected, rejected_reward, rejected_details = ranked[-1]
            if chosen_reward - rejected_reward < CFG["min_reward_gap"]:
                continue

            json.dump(
                {
                    "prompt": row["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "original_text": original,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "chosen_details": chosen_details,
                    "rejected_details": rejected_details,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
            kept += 1

    scorer.close()
    print(f"[prefs] Saved {kept} preference pairs: {pref_path}")
    if kept == 0:
        raise RuntimeError("No preference pairs were created; lower STEALTHRL_DPO_MIN_GAP.")


def make_dpo_config() -> DPOConfig:
    kwargs = dict(
        output_dir=CFG["output_dir"],
        learning_rate=CFG["learning_rate"],
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        max_steps=CFG["max_steps"],
        warmup_steps=CFG["warmup_steps"],
        logging_steps=10,
        save_steps=CFG["save_steps"],
        save_total_limit=5,
        beta=CFG["dpo_beta"],
        bf16=USE_BF16,
        fp16=USE_CUDA and not USE_BF16,
        report_to="none",
        optim="adamw_torch_fused" if USE_CUDA else "adamw_torch",
        gradient_checkpointing=True,
        max_length=CFG["max_length"],
        max_prompt_length=CFG["max_prompt_length"],
    )
    try:
        return DPOConfig(**kwargs)
    except TypeError:
        # Older TRL versions do not accept max_length/max_prompt_length here.
        kwargs.pop("max_length", None)
        kwargs.pop("max_prompt_length", None)
        return DPOConfig(**kwargs)


def make_dpo_trainer(model: Any, tokenizer: Any, dataset: Dataset) -> DPOTrainer:
    args = make_dpo_config()
    try:
        return DPOTrainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )
    except TypeError:
        return DPOTrainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            train_dataset=dataset,
        )


def train_dpo() -> None:
    build_preference_data()
    pref_rows = load_jsonl(CFG["preference_data"])
    dataset = Dataset.from_list(
        [
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for row in pref_rows
        ]
    )
    print(f"[dpo] Preference pairs: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[dpo] Loading train model: {CFG['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"],
        trust_remote_code=True,
        torch_dtype=MODEL_DTYPE,
        device_map="auto" if USE_CUDA else None,
    )
    model.config.use_cache = False
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CFG["lora_rank"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    trainer = make_dpo_trainer(model, tokenizer, dataset)
    print("[dpo] Starting DPO warmup...")
    trainer.train()
    trainer.save_model(CFG["output_dir"])
    tokenizer.save_pretrained(CFG["output_dir"])
    print(f"[dpo] Saved DPO warmup adapter: {CFG['output_dir']}")


if __name__ == "__main__":
    train_dpo()
