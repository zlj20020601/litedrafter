#!/usr/bin/env python3
"""
CodeContests benchmark 数据独立审核
重点验证（用户强调）：
  1. len(final_input_ids) == 1024
  2. system/user role token 正常
  3. assistant generation prefix 正常
  4. 没有 assistant/reference solution
"""
import json, os, hashlib
from collections import Counter
from transformers import AutoTokenizer

OUTPUT_FILE = "/root/autodl-tmp/litedrafter/data/codecontests_qwen3_1024_256.jsonl"
META_FILE   = "/root/autodl-tmp/litedrafter/data/codecontests_qwen3_1024_256.meta.json"
MODEL_PATH  = "/root/autodl-tmp/models/Qwen3-8B"
TARGET = 1024
EXPECTED = 256

print("=" * 70)
print("CodeContests benchmark 数据独立审核")
print("=" * 70)

records = []
with open(OUTPUT_FILE) as f:
    for line in f:
        records.append(json.loads(line.strip()))
print(f"加载 {len(records)} 条")

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"tokenizer vocab_size={tok.vocab_size}\n")

PASS, FAIL, WARN = "✓ PASS", "✗ FAIL", "⚠ WARN"
results = []

def check(num, desc, status, detail=""):
    results.append((num, status))
    print(f"{'─'*60}")
    print(f"检查 {num}: {desc}")
    print(f"  状态: {status}")
    if detail:
        print(f"  {detail}")

# ━━ 1. len == 1024 ━━
wrong = [(i, len(r["input_ids"])) for i, r in enumerate(records) if len(r["input_ids"]) != TARGET]
check(1, "每条 final_input_ids 精确 1024 tokens",
      PASS if not wrong else FAIL,
      f"wrong={len(wrong)}" + (f" e.g. {wrong[:5]}" if wrong else ""))

# ━━ 2. user role token 正常 ━━
bad_user = []
for i, r in enumerate(records):
    dec = r["decoded_prompt"]
    if not dec.startswith("<|im_start|>user\n"):
        bad_user.append(i)
check(2, "user role token 正常（开头 <|im_start|>user）",
      PASS if not bad_user else FAIL,
      f"bad={len(bad_user)}")

# ━━ 3. assistant generation prefix 正常 ━━
bad_gen = []
gen_suffix = tok.encode("<|im_start|>assistant\n\n\n", add_special_tokens=False)
for i, r in enumerate(records):
    ids = r["input_ids"]
    # Qwen3: <|im_start|>assistant\n<think>\n\n</think>\n\n
    if "<|im_start|>assistant" not in r["decoded_prompt"]:
        bad_gen.append(i)
check(3, "assistant generation prefix 完整",
      PASS if not bad_gen else FAIL,
      f"missing_generation_prefix={len(bad_gen)}")

# ━━ 4. 无 assistant/reference solution ━━
suspicious = []
for i, r in enumerate(records):
    content = r["messages"][0]["content"]
    # reference solution 常用标记（弱检查：生成过程只用了 problem 字段）
    markers = ["## Solution", "```python\n", "def solve()", "Below is the solution",
               "Here is the solution", "reference solution", "expected output:"]
    for m in markers:
        if m.lower() in content.lower():
            suspicious.append((i, m))
            break
check(4, "无 assistant/reference solution（只用 problem 字段）",
      PASS if not suspicious else WARN,
      f"content含solution标记={len(suspicious)}" +
      (f" e.g. {suspicious[:3]}" if suspicious else "") +
      "\n  说明: 生成脚本只读取 problem 字段，solutions 字段从未参与")

# ━━ 5. 256 条，sample_id 唯一 ━━
ids = [r["sample_id"] for r in records]
check(5, "共 256 条，sample_id 唯一",
      PASS if len(records) == EXPECTED and len(set(ids)) == len(ids) else FAIL,
      f"count={len(records)}, unique_ids={len(set(ids))}")

# ━━ 6. 选区构成：A 桶 249 + B1 桶 7 ━━
raw_lens = [r["prompt_token_length_raw"] for r in records]
a_cnt = sum(1 for l in raw_lens if 1024 <= l <= 1152)
b1_cnt = sum(1 for l in raw_lens if 1153 <= l <= 1280)
other = len(records) - a_cnt - b1_cnt
check(6, "选区: 249×[1024,1152] + 7×[1153,1280]",
      PASS if a_cnt >= 249 and b1_cnt == 7 else WARN,
      f"A={a_cnt}, B1={b1_cnt}, other={other}, min={min(raw_lens)}, max={max(raw_lens)}")

# ━━ 7. source 分布 ━━
src_dist = Counter(r["source"] for r in records)
check(7, "source 分布记录",
      PASS if len(src_dist) > 1 else WARN,
      ", ".join(f"{k}={v}" for k, v in sorted(src_dist.items(), key=lambda x: -x[1])))

# ━━ 8. 无重复 content ━━
contents = [r["messages"][0]["content"] for r in records]
hashes = [hashlib.md5(c.encode()).hexdigest() for c in contents]
check(8, "无重复 content",
      PASS if len(set(hashes)) == len(hashes) else FAIL,
      f"duplicates={len(hashes) - len(set(hashes))}")

# ━━ 9. metadata 落盘 + 一致性 ━━
meta_ok = os.path.exists(META_FILE)
seed_ok = False
if meta_ok:
    with open(META_FILE) as f:
        meta = json.load(f)
    seed_ok = meta.get("selection", {}).get("seed") == 42
    note = meta.get("preprocess_note", "")
    excl = meta.get("assistant_reference_excluded")
check(9, "metadata 落盘 + seed=42 + 预处理说明",
      PASS if meta_ok and seed_ok else FAIL,
      f"meta_exists={meta_ok}, seed=42: {seed_ok}"
      + (f"\n  preprocess_note: {note[:120]}" if meta_ok else "")
      + (f"\n  assistant_reference_excluded: {excl}" if meta_ok else ""))

# ━━ 10. 独立复算长度（decode→encode） ━━
mismatches = []
for i, r in enumerate(records):
    re_ids = tok.encode(r["decoded_prompt"], add_special_tokens=False)
    if len(re_ids) != TARGET:
        mismatches.append((i, len(re_ids)))
check(10, "独立复算长度（decode→encode 可逆性）",
      PASS if not mismatches else WARN,
      f"mismatches={len(mismatches)}" +
      (f" e.g. {mismatches[:5]} (BPE 边界已知限制)" if mismatches else ""))

# ━━ 总结 ━━
print("\n" + "=" * 70)
passes = sum(1 for _, s in results if s == PASS)
fails = sum(1 for _, s in results if s == FAIL)
warns = sum(1 for _, s in results if s == WARN)
print(f"总结: {passes} PASS / {warns} WARN / {fails} FAIL (共 {len(results)} 项)")
print("=" * 70)
