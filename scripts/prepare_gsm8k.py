#!/usr/bin/env python3
"""
Step A 数据准备：GSM8K 前128条，编码为 token_ids
输出: /root/autodl-tmp/litedrafter/data/gsm8k_128.jsonl
"""
import json, transformers

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
SRC = "/root/autodl-tmp/archive/eagle3_repro/repos/DeepSpec/eval_datasets/gsm8k.jsonl"
OUT = "/root/autodl-tmp/litedrafter/data/gsm8k_128.jsonl"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

with open(SRC) as f:
    all_items = [json.loads(line) for line in f]

items = all_items[:128]
print(f"GSM8K 总条数: {len(all_items)}, 取前 {len(items)} 条")

lengths = []
records = []
for i, item in enumerate(items):
    question = item["turns"][0]
    token_ids = tokenizer.encode(question, add_special_tokens=True)
    lengths.append(len(token_ids))
    records.append({
        "idx": i,
        "question": question,
        "token_ids": token_ids,
        "input_len": len(token_ids),
    })

with open(OUT, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

lengths.sort()
print(f"\nToken 长度分布:")
print(f"  min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")
print(f"  p25={lengths[32]}, p50={lengths[64]}, p75={lengths[96]}, p95={lengths[121]}")
print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")
print(f"\n输出: {OUT}")
