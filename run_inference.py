import pandas as pd
import torch
import os
import re
import unicodedata
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

PROJECT_ROOT = os.environ.get("STEALTHRL_PROJECT_ROOT", os.getcwd())
BASE_MODEL = os.environ.get("STEALTHRL_BASE_MODEL", os.path.join(PROJECT_ROOT, "Qwen2.5-0.5B-Instruct"))
LORA_PATH = os.path.join(PROJECT_ROOT, "checkpoints/mage-rob-ac-100-v1")
TEST_CSV = os.environ.get("STEALTHRL_TEST_CSV", os.path.join(PROJECT_ROOT, "AI_Text Evasion/TextGenAdvTrack-2026Spring/UCAS_AISAD_TEXT-test1.csv"))

BATCH_SIZE = int(os.environ.get("STEALTHRL_INFER_BATCH", "128"))
MAX_NEW_TOKENS = int(os.environ.get("STEALTHRL_MAX_NEW_TOKENS", "512"))

USE_CUDA = torch.cuda.is_available()
USE_BF16 = USE_CUDA and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_CUDA else torch.float32)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, trust_remote_code=True,
    dtype=DTYPE, device_map="auto",
)
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()

df = pd.read_csv(TEST_CSV)
results = []
prompts = df['prompt'].tolist()

for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating"):
    batch_prompts = prompts[i:i+BATCH_SIZE]

    inputs = tok(
        batch_prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tok.pad_token_id,
            eos_token_id=model.config.eos_token_id,
        )

    for j, out in enumerate(outputs):
        gen = tok.decode(out[prompt_len:], skip_special_tokens=True).strip()
        gen = gen.encode('utf-8', errors='replace').decode('utf-8')
        gen = clean_text(gen)
        
        if len(gen.strip()) < 5:
            gen = ""
            
        results.append(gen)

submit_df = df.copy()
submit_df['text'] = results
submit_df.to_csv('submission.csv', index=False, encoding='utf-8-sig')