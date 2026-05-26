# convert_to_jsonl.py
import pandas as pd
import json
import os
import shutil

DATA_DIR = "/data/StealthRL-main/StealthRL-main/AI_Text Evasion/TextGenAdvTrack-2026Spring"
OUT_DIR  = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# 读取 val.csv（已含 prompt / text / label 三列）
val = pd.read_csv(os.path.join(DATA_DIR, "UCAS_AISAD_TEXT-val.csv"))
print(f"val.csv 总行数: {len(val)}")
print(f"列名: {list(val.columns)}")
print(f"label 分布:\n{val['label'].value_counts()}\n")

# 只取 label=1 的 AI 生成文本参与 RL 训练
ai_rows = val[val["label"] == 1].reset_index(drop=True)
split    = int(len(ai_rows) * 0.8)
train_df = ai_rows.iloc[:split]
eval_df  = ai_rows.iloc[split:]
print(f"训练集: {len(train_df)} 条")
print(f"验证集: {len(eval_df)} 条")

# 写出 JSONL
def write_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {
                "prompt": str(row["prompt"]),
                "text":   str(row["text"]),
                "label":  int(row["label"])
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"已写出: {path}  ({len(df)} 条)")

write_jsonl(train_df, f"{OUT_DIR}/train.jsonl")
write_jsonl(eval_df,  f"{OUT_DIR}/eval.jsonl")

# esl_validation / native_validation 暂用 eval 集代替
shutil.copy(f"{OUT_DIR}/eval.jsonl", f"{OUT_DIR}/esl_validation.jsonl")
shutil.copy(f"{OUT_DIR}/eval.jsonl", f"{OUT_DIR}/native_validation.jsonl")

print("\n✅ 数据转换完成！目录结构：")
for f in os.listdir(OUT_DIR):
    print(f"  data/processed/{f}")