# 保存为 prepare_mage_data.py
from datasets import load_dataset
import json, os, random, shutil

os.makedirs('data/processed', exist_ok=True)

print("加载 MAGE 数据集...")
ds = load_dataset('yaful/MAGE', split='train')

# 取 AI 生成文本（label=0）
ai_samples = [x for x in ds if x['label'] == 0]
print(f"AI 生成样本总数: {len(ai_samples)}")

# 随机采样 10000 条（论文用量）
random.seed(42)
random.shuffle(ai_samples)
ai_samples = ai_samples[:10000]

# 9:1 分训练/验证
split = int(len(ai_samples) * 0.9)
train_data = ai_samples[:split]   # 9000 条
eval_data  = ai_samples[split:]   # 1000 条

def write_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for x in data:
            text = x['text']
            obj = {
                # prompt = 模型的输入指令（论文固定模板）
                'prompt': f'Paraphrase the following text while preserving its meaning: {text}',
                # original_text = 原始 AI 文本（奖励函数计算语义相似度用）
                'original_text': text,
                'label': 0   # 0=AI生成（MAGE 约定）
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f"写出: {path} ({len(data)} 条)")

write_jsonl(train_data, 'data/processed/train.jsonl')
write_jsonl(eval_data,  'data/processed/eval.jsonl')
shutil.copy('data/processed/eval.jsonl', 'data/processed/esl_validation.jsonl')
shutil.copy('data/processed/eval.jsonl', 'data/processed/native_validation.jsonl')
print("✅ MAGE 训练数据准备完成")