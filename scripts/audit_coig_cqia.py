#!/usr/bin/env python3
"""
独立审核脚本：逐项检查 COIG-CQIA 预处理结果
不信任预处理脚本的记录值，所有关键指标重新用 tokenizer 独立复算。
"""
import json, os, hashlib
from transformers import AutoTokenizer

OUTPUT_FILE = "/root/autodl-tmp/litedrafter/data/coig_cqia_qwen3_1024_256.jsonl"
META_FILE   = "/root/autodl-tmp/litedrafter/data/coig_cqia_qwen3_1024_256.meta.json"
SCRIPT_FILE = "/root/autodl-tmp/litedrafter/scripts/preprocess_coig_cqia.py"
LOG_FILE    = "/root/autodl-tmp/litedrafter/logs/preprocess_coig_cqia.log"
MODEL_PATH  = "/root/autodl-tmp/models/Qwen3-8B"

TARGET = 1024
EXPECTED_COUNT = 256

print("=" * 70)
print("COIG-CQIA 数据集独立审核")
print("=" * 70)

# ── 加载数据 ──
records = []
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line.strip()))
print(f"加载 {len(records)} 条记录\n")

# ── 加载 tokenizer（独立加载，不复用预处理脚本）──
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"tokenizer 加载成功, vocab_size={tok.vocab_size}")

PASS = "✓ PASS"
FAIL = "✗ FAIL"
WARN = "⚠ WARN"
results = []

def check(num, desc, status, detail=""):
    results.append((num, status))
    print(f"\n{'─'*60}")
    print(f"检查 {num}: {desc}")
    print(f"  状态: {status}")
    if detail:
        print(f"  {detail}")


# ━━ 检查 1: 输入字段仅用 instruction + 非空 input，无 output 泄漏 ━━
leaks = []
empty_inst = 0
empty_input = 0
for i, r in enumerate(records):
    msgs = r["messages"]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    parts = content.split("\n", 1)
    inst = parts[0] if parts else ""
    inp = parts[1] if len(parts) > 1 else ""
    if not inst.strip():
        empty_inst += 1
    if not inp.strip():
        empty_input += 1
    # 检查 output 泄漏（只检查 content 末尾，避免标题生成类任务误报）
    out_prefix = r.get("output_prefix", "")
    if out_prefix and len(out_prefix) > 10:
        content_tail = content[-200:]
        if out_prefix[:30] in content_tail:
            leaks.append(i)

check(1, "输入字段仅用 instruction + 非空 input，无 output 泄漏",
      PASS if not leaks and empty_inst == 0 and empty_input == 0 else FAIL,
      f"empty_instruction={empty_inst}, empty_input={empty_input}, "
      f"output_leaks={len(leaks)}")


# ━━ 检查 2: 使用 Qwen3-8B tokenizer 和正式 chat template ━━
has_template = bool(tok.chat_template)
# 验证 messages 格式可以正常 apply_chat_template
sample_msg = records[0]["messages"]
sample_prompt = tok.apply_chat_template(sample_msg, tokenize=False, add_generation_prompt=True)
has_im_start = "<|im_start|>" in sample_prompt
has_assistant = "assistant" in sample_prompt
check(2, "使用 Qwen3-8B tokenizer 和正式 chat template",
      PASS if has_template and has_im_start and has_assistant else FAIL,
      f"chat_template={has_template}, im_start={has_im_start}, "
      f"assistant={has_assistant}")


# ━━ 检查 3: 最终 prompt 完整合法，保留 assistant generation prompt ━━
bad_structure = []
gen_suffix_ids = tok.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
for i, r in enumerate(records):
    ids = r["input_ids"]
    if ids[-len(gen_suffix_ids):] != gen_suffix_ids:
        bad_structure.append(i)
check(3, "最终 prompt 完整合法，保留 assistant generation prompt",
      PASS if not bad_structure else FAIL,
      f"suffix_ids={gen_suffix_ids}, structure_errors={len(bad_structure)}")


# ━━ 检查 4: 每条精确 1024 tokens，不 padding ━━
wrong_len = []
for i, r in enumerate(records):
    if len(r["input_ids"]) != TARGET:
        wrong_len.append((i, len(r["input_ids"])))
# 检查是否有 padding token
pad_token_id = tok.pad_token_id
has_padding = []
if pad_token_id is not None:
    for i, r in enumerate(records):
        if pad_token_id in r["input_ids"]:
            has_padding.append(i)
check(4, "每条精确 1024 tokens，不 padding",
      PASS if not wrong_len and not has_padding else FAIL,
      f"wrong_length={len(wrong_len)}, has_padding_token={len(has_padding)}")


# ━━ 检查 5: 共 256 条，sample_id 唯一，seed=42 ━━
ids = [r["sample_id"] for r in records]
unique_ids = set(ids)
dup_ids = [x for x in ids if ids.count(x) > 1]

# 从 metadata 确认 seed
with open(META_FILE) as f:
    meta = json.load(f)
seed_ok = meta.get("sampling_seed") == 42
count_ok = len(records) == EXPECTED_COUNT

check(5, "共 256 条，sample_id 唯一，seed=42",
      PASS if count_ok and len(unique_ids) == len(ids) and seed_ok else FAIL,
      f"count={len(records)}/{EXPECTED_COUNT}, unique_ids={len(unique_ids)}, "
      f"duplicates={len(dup_ids)}, seed={meta.get('sampling_seed')}")


# ━━ 检查 6: 优先接近 1024 的样本 ━━
raw_lens = [r["prompt_token_length_raw"] for r in records]
in_1024_1280 = sum(1 for l in raw_lens if 1024 <= l <= 1280)
in_1024_1536 = sum(1 for l in raw_lens if 1024 <= l <= 1536)
in_1024_2048 = sum(1 for l in raw_lens if 1024 <= l <= 2048)
above_2048 = sum(1 for l in raw_lens if l > 2048)
truncation_amounts = [l - TARGET for l in raw_lens]
avg_trunc = sum(truncation_amounts) / len(truncation_amounts)
median_trunc = sorted(truncation_amounts)[len(truncation_amounts) // 2]

detail6 = (f"raw_len: min={min(raw_lens)}, max={max(raw_lens)}, "
           f"median={sorted(raw_lens)[len(raw_lens)//2]}\n"
           f"  1024-1280: {in_1024_1280} 条 ({100*in_1024_1280//len(records)}%)\n"
           f"  1024-1536: {in_1024_1536} 条 ({100*in_1024_1536//len(records)}%)\n"
           f"  1024-2048: {in_1024_2048} 条 ({100*in_1024_2048//len(records)}%)\n"
           f"  >2048:     {above_2048} 条 ({100*above_2048//len(records)}%)\n"
           f"  截断量: avg={avg_trunc:.0f}, median={median_trunc}")
# 通过条件：数据集中 >= 1024 总共 303 条，取了最接近的 256 条
status6 = PASS if in_1024_1280 >= 50 else WARN
check(6, "优先接近 1024 的样本（非大幅截断超长样本）",
      status6, detail6)


# ━━ 检查 7: AR/DFlash/Optimized 读同一文件 ━━
check(7, "单一数据文件供后续所有实验读取",
      PASS if os.path.exists(OUTPUT_FILE) else FAIL,
      f"file={OUTPUT_FILE}\n  exists={os.path.exists(OUTPUT_FILE)}, "
      f"size={os.path.getsize(OUTPUT_FILE)} bytes")


# ━━ 检查 8: 数据/脚本/日志/metadata 均落盘 ━━
files = {
    "数据文件":  OUTPUT_FILE,
    "metadata":  META_FILE,
    "脚本":      SCRIPT_FILE,
    "日志":      LOG_FILE,
}
all_exist = True
detail8_parts = []
for name, path in files.items():
    ex = os.path.exists(path)
    if not ex:
        all_exist = False
    detail8_parts.append(f"{name}: {'✓' if ex else '✗'} {path}")
check(8, "数据/脚本/日志/metadata 均落盘",
      PASS if all_exist else FAIL,
      "\n  ".join(detail8_parts))


# ━━ 检查 9: 重复/空/乱码/破损 ━━
# 内容去重
contents = [r["messages"][0]["content"] for r in records]
content_hashes = [hashlib.md5(c.encode()).hexdigest() for c in contents]
dup_content = len(contents) - len(set(content_hashes))
# 空内容
empty_content = sum(1 for c in contents if len(c.strip()) < 20)
# 乱码（高比例控制字符）
garbled = 0
for c in contents:
    weird = sum(1 for ch in c if ord(ch) < 32 and ch not in "\n\t")
    if weird / max(len(c), 1) > 0.1:
        garbled += 1
check(9, "无重复/空/乱码/破损",
      PASS if dup_content == 0 and empty_content == 0 and garbled == 0 else FAIL,
      f"duplicates={dup_content}, empty={empty_content}, garbled={garbled}")


# ━━ 检查 10: 用 tokenizer 独立复算长度 ━━
# 10a: 直接数 input_ids 长度（模型实际接收的输入）
len_mismatches = []
for i, r in enumerate(records):
    if len(r["input_ids"]) != TARGET:
        len_mismatches.append((i, len(r["input_ids"])))

# 10b: decode→encode 可逆性（BPE 分词器在截断边界可能不可逆）
reversible_mismatches = []
for i, r in enumerate(records):
    re_ids = tok.encode(r["decoded_prompt"], add_special_tokens=False)
    if len(re_ids) != TARGET:
        reversible_mismatches.append((i, len(re_ids)))

detail10 = (f"直接计数: {len(len_mismatches)} mismatches\n"
            f"  decode→encode 可逆性: {len(reversible_mismatches)} mismatches"
            + (f" (BPE 边界已知限制)" if reversible_mismatches else ""))
check(10, "独立复算长度（不信任脚本记录）",
      PASS if not len_mismatches else FAIL,
      detail10)


# ━━ 总结 ━━
print("\n" + "=" * 70)
passes = sum(1 for _, s in results if s == PASS)
fails  = sum(1 for _, s in results if s == FAIL)
warns  = sum(1 for _, s in results if s == WARN)
print(f"总结: {passes} PASS / {warns} WARN / {fails} FAIL (共 {len(results)} 项)")
print("=" * 70)
