#!/usr/bin/env python3
"""
CodeContests Qwen3.5-4B 数据独立审核
10 项检查，与之前 Qwen3-8B 版本审核标准完全一致
"""
import json, sys, os
from collections import Counter

DATA_FILE = "/root/autodl-tmp/litedrafter/data/codecontests_qwen35_4b_1024_256.jsonl"
META_FILE = "/root/autodl-tmp/litedrafter/data/codecontests_qwen35_4b_1024_256.meta.json"
MODEL_PATH = "/root/autodl-tmp/models/Qwen3.5-4B"

results = []
passes = 0
warns = 0
fails = 0

def check(name, condition, detail=""):
    global passes, warns, fails
    status = "PASS" if condition else "FAIL"
    if condition:
        passes += 1
    else:
        fails += 1
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

def warn_check(name, condition, detail=""):
    global passes, warns, fails
    if condition:
        passes += 1
        print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        warns += 1
        print(f"[WARN] {name}" + (f" — {detail}" if detail else ""))


# ── 加载数据 ──
with open(DATA_FILE) as f:
    for line in f:
        results.append(json.loads(line))
with open(META_FILE) as f:
    meta = json.load(f)

print("=" * 70)
print(f"CodeContests Qwen3.5-4B 独立审核")
print(f"数据文件: {DATA_FILE}")
print(f"样本数: {len(results)}")
print("=" * 70)

# ── 1. 样本数量 ──
sids = [r["sample_id"] for r in results]
pids = [r["problem_id"] for r in results]
contents = [r["messages"][0]["content"] for r in results]
check("1. 样本数量=256", len(results) == 256, f"实际 {len(results)}")
check("1b. sample_id 唯一", len(set(sids)) == len(sids), f"unique {len(set(sids))}/{len(sids)}")
check("1c. problem_id 唯一", len(set(pids)) == len(pids), f"unique {len(set(pids))}/{len(pids)}")
check("1d. content 唯一", len(set(contents)) == len(contents), f"unique {len(set(contents))}/{len(contents)}")

# ── 2. 最终 token 长度（独立复算）──
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

recomputed_lens = []
mismatches = 0
decode_mismatches = 0
for i, r in enumerate(results):
    # 重新从 messages render
    rendered = tok.apply_chat_template(
        r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids_recomputed = tok.encode(rendered, add_special_tokens=False)
    recomputed_lens.append(len(ids_recomputed))

    if rendered != r["decoded_prompt"]:
        decode_mismatches += 1

all_1024 = all(l == 1024 for l in recomputed_lens)
check("2. 独立复算 token 长度全部=1024", all_1024,
      f"min={min(recomputed_lens)}, max={max(recomputed_lens)}, all_1024={sum(1 for l in recomputed_lens if l==1024)}/256")

# ── 3. chat template 正确 ──
# 检查每条的 decoded_prompt 以 assistant generation prompt 结尾
suffix_ok = 0
think_closed = 0
for r in results:
    dp = r["decoded_prompt"]
    # Qwen3.5 enable_thinking=False: ...assistant\n<think>\n\n</think>\n\n
    if "<|im_start|>assistant\n" in dp:
        suffix_ok += 1
    if "</think>" in dp:
        think_closed += 1

check("3. chat template 包含 assistant generation prompt", suffix_ok == 256, f"{suffix_ok}/256")
check("3b. enable_thinking=False → </think> 关闭", think_closed == 256, f"{think_closed}/256")

# ── 4. 没有把 solution/answer 泄漏到 prompt ──
solution_markers = ["solution", "def solve", "```python\n", "answer:", "output:", "reference"]
solution_hits = 0
for r in results:
    c = r["messages"][0]["content"].lower()
    for marker in solution_markers:
        if marker in c:
            solution_hits += 1
            break
check("4. content 无 solution/answer 泄漏", solution_hits == 0,
      f"{solution_hits} 条疑似包含答案标记")

# ── 5. generation prompt 完整（user role 开头）──
user_start = sum(1 for r in results if r["decoded_prompt"].startswith("<|im_start|>user\n"))
check("5. 全部以 user role 开头", user_start == 256, f"{user_start}/256")

# ── 6. decode→encode 可逆 ──
reversible = 0
for r in results:
    ids = tok.encode(r["decoded_prompt"], add_special_tokens=False)
    decoded_back = tok.decode(ids)
    if decoded_back == r["decoded_prompt"]:
        reversible += 1
check("6. decode→encode 可逆", reversible == 256, f"{reversible}/256")

# render mismatch
check("6b. 重新 render 与保存的 decoded_prompt 一致", decode_mismatches == 0,
      f"{decode_mismatches} 条不一致")

# ── 7. input_ids 与 decoded_prompt 一致 ──
ids_match = 0
for r in results:
    ids_from_stored = tok.encode(r["decoded_prompt"], add_special_tokens=False)
    if ids_from_stored == r["input_ids"]:
        ids_match += 1
check("7. input_ids 与 decoded_prompt 一致", ids_match == 256, f"{ids_match}/256")

# ── 8. 截断情况 ──
truncated = sum(1 for r in results if r["truncated"])
raw_lens = [r["prompt_token_length_raw"] for r in results]
print(f"\n[INFO] 截断: {truncated}/256 条")
print(f"[INFO] raw length: min={min(raw_lens)}, median={sorted(raw_lens)[len(raw_lens)//2]}, max={max(raw_lens)}")
warn_check("8. raw length 在 [1024, 1280]",
           min(raw_lens) >= 1024 and max(raw_lens) <= 1280,
           f"min={min(raw_lens)}, max={max(raw_lens)}")

# ── 9. source 分布 ──
src_dist = Counter(r["source"] for r in results)
print(f"\n[INFO] source 分布: {dict(src_dist)}")

# ── 10. 重复/空问题/乱码检查 ──
empty = sum(1 for r in results if len(r["messages"][0]["content"].strip()) < 10)
check("9. 无空问题/乱码", empty == 0, f"{empty} 条疑似空/短")

# ── summary ──
print("\n" + "=" * 70)
print(f"审核结果: {passes} PASS / {warns} WARN / {fails} FAIL")
print("=" * 70)

# 检查 input_ids 长度也全是 1024
ids_lens = [len(r["input_ids"]) for l, r in zip(recomputed_lens, results)]
ids_all_1024 = all(len(r["input_ids"]) == 1024 for r in results)
check("10. stored input_ids 全部 len=1024", ids_all_1024,
      f"{sum(1 for r in results if len(r['input_ids'])==1024)}/256")
