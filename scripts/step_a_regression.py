#!/usr/bin/env python3
"""
Step A: 正确性回归测试
GSM8K 128条, max_new_tokens=256, greedy (temperature=0)

5 个验证点:
  1. DFlash batch=1 新旧实现: raw>=1 时一致, raw=0 时新=1旧=2
  2. 相同样本单独运行与 batch 内 raw_accept 基本一致
  3. DFlash 最终输出与 target greedy AR 一致
  4. raw=0 时只推进 1 个 bonus token
  5. padding/mask/position/KV crop 不改变样本语义
"""
import gc, json, time, torch, transformers
from dflash.model import DFlashDraftModel, dflash_generate

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DRAFTER_PATH = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
DATA_FILE = "/root/autodl-tmp/litedrafter/data/gsm8k_128.jsonl"
DTYPE = torch.bfloat16
MAX_NEW = 256
N = 128
EOS_DISABLE_ID = 999999  # 不存在的 token id, 禁用 AR 的 EOS 停止

# ── 加载数据 ──
records = [json.loads(line) for line in open(DATA_FILE)]
assert len(records) >= N, f"需要 {N} 条, 只有 {len(records)}"
print(f"加载 {N} 条 GSM8K, token_len: min={min(r['input_len'] for r in records[:N])} "
      f"max={max(r['input_len'] for r in records[:N])}")

# ── 加载模型 ──
print("加载 target...")
target = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=DTYPE, attn_implementation="sdpa"
).to("cuda").eval()

print("加载 drafter...")
drafter = DFlashDraftModel.from_pretrained(
    DRAFTER_PATH, dtype=DTYPE, attn_implementation="sdpa"
).to("cuda").eval()
print(f"block_size={drafter.block_size}")

pad_id = transformers.AutoTokenizer.from_pretrained(MODEL_PATH).pad_token_id
print(f"pad_id={pad_id}")

# ============================================================
# Phase 1: AR batch=1 ground truth (128条逐条)
# ============================================================
print(f"\n{'='*60}")
print(f"Phase 1: AR batch=1 ground truth ({N} 条)")
print(f"{'='*60}")

ar_outputs = []  # ar_outputs[i] = list of token ids
t0 = time.time()
for i in range(N):
    ids = torch.tensor([records[i]["token_ids"]], dtype=torch.long, device="cuda")
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        out = target.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=pad_id, attention_mask=mask,
            eos_token_id=EOS_DISABLE_ID,
        )
    new_tokens = out[0, ids.shape[1]:ids.shape[1]+MAX_NEW].tolist()
    ar_outputs.append(new_tokens)
    if (i+1) % 16 == 0:
        elapsed = time.time() - t0
        print(f"  AR: {i+1}/{N} done, {elapsed:.0f}s, avg {elapsed/(i+1):.1f}s/条")

print(f"  AR 全部完成: {time.time()-t0:.0f}s")

# ============================================================
# Phase 2: DFlash batch=1 (128条逐条)
# ============================================================
print(f"\n{'='*60}")
print(f"Phase 2: DFlash batch=1 ({N} 条)")
print(f"{'='*60}")

df1_outputs = []     # df1_outputs[i] = list of token ids
df1_step_acc = []    # df1_step_acc[i] = list of {raw_draft_accept, used_draft_accept, advance_tokens}

t0 = time.time()
for i in range(N):
    ids = torch.tensor([records[i]["token_ids"]], dtype=torch.long, device="cuda")
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        stats = dflash_generate(
            drafter, target=target, input_ids=ids,
            max_new_tokens=MAX_NEW, stop_token_ids=None,
            temperature=0.0, return_stats=True,
            ignore_eos=True, attention_mask=mask,
        )
    new_tokens = stats.output_ids[0, ids.shape[1]:ids.shape[1]+MAX_NEW].tolist()
    df1_outputs.append(new_tokens)
    df1_step_acc.append(stats.acceptance_lengths)
    if (i+1) % 16 == 0:
        elapsed = time.time() - t0
        print(f"  DFlash b1: {i+1}/{N} done, {elapsed:.0f}s")

print(f"  DFlash b1 全部完成: {time.time()-t0:.0f}s")

# ============================================================
# Phase 3: 验证点 1, 3, 4
# ============================================================
print(f"\n{'='*60}")
print("Phase 3: 验证 (batch=1)")
print(f"{'='*60}")

# ── 验证点 3: DFlash batch=1 vs AR 逐 token ──
exact_match = 0
prefix_match = 0
mismatch_positions = []
total_positions = 0
matched_positions = 0

for i in range(N):
    ar = ar_outputs[i]
    df = df1_outputs[i]
    n = min(len(ar), len(df))
    total_positions += n

    first_mm = None
    for j in range(n):
        if ar[j] == df[j]:
            matched_positions += 1
        else:
            if first_mm is None:
                first_mm = j
    if first_mm is None:
        exact_match += 1
    else:
        mismatch_positions.append((i, first_mm))

prefix_match_ratio = matched_positions / total_positions if total_positions else 0
print(f"\n验证点3: DFlash batch=1 vs AR greedy")
print(f"  逐 token match: {matched_positions}/{total_positions} ({prefix_match_ratio*100:.2f}%)")
print(f"  完全一致样本: {exact_match}/{N} ({exact_match/N*100:.1f}%)")
if mismatch_positions:
    mm_pos = [p for _, p in mismatch_positions]
    print(f"  首 mismatch 位置: min={min(mm_pos)}, mean={sum(mm_pos)/len(mm_pos):.0f}")
    print(f"  前5个 mismatch 样本: {mismatch_positions[:5]}")

# ── 验证点 1: advance 逻辑 ──
v1_pass = True
v1_raw0_count = 0
v1_raw_ge1_count = 0
v1_advance_check_fail = 0

for i in range(N):
    for step in df1_step_acc[i]:
        used = step["used_draft_accept"]
        advance = step["advance_tokens"]
        # 新逻辑: advance = used + 1
        if advance != used + 1:
            v1_advance_check_fail += 1
            v1_pass = False
        # 旧逻辑对比: raw >= 1 时新旧一致; raw == 0 时新 advance=1, 旧 advance=2
        if used == 0:
            v1_raw0_count += 1
            if advance != 1:
                v1_advance_check_fail += 1
                v1_pass = False
        else:
            v1_raw_ge1_count += 1

total_steps = sum(len(df1_step_acc[i]) for i in range(N))
print(f"\n验证点1: advance 逻辑 (新旧对比)")
print(f"  总步数: {total_steps}")
print(f"  raw>=1 步数: {v1_raw_ge1_count} (新旧应一致)")
print(f"  raw==0 步数: {v1_raw0_count} (新 advance=1, 旧 advance=2)")
print(f"  advance != used+1 的步数: {v1_advance_check_fail}")
print(f"  结果: {'PASS' if v1_pass else 'FAIL'}")

# ── 验证点 4: raw=0 时只推进 1 个 bonus ──
v4_pass = True
v4_examples = []
for i in range(N):
    for step in df1_step_acc[i]:
        if step["used_draft_accept"] == 0:
            if step["advance_tokens"] != 1:
                v4_pass = False
            if len(v4_examples) < 3:
                v4_examples.append({"sample": i, "advance": step["advance_tokens"]})

print(f"\n验证点4: raw=0 时 advance=1")
print(f"  raw=0 步数: {v1_raw0_count}")
print(f"  全部 advance=1: {'PASS' if v4_pass else 'FAIL'}")
if v4_examples:
    print(f"  示例: {v4_examples}")

# ============================================================
# Phase 4: DFlash batch=2/4/8 — 验证点 2, 5
# ============================================================
print(f"\n{'='*60}")
print("Phase 4: DFlash batch=2/4/8 (batch 语义验证)")
print(f"{'='*60}")

def build_batch(indices):
    """构造 left-padded batch"""
    all_ids = [records[i]["token_ids"] for i in indices]
    max_len = max(len(x) for x in all_ids)
    batch_ids = []
    batch_mask = []
    for ids in all_ids:
        pad = max_len - len(ids)
        batch_ids.append([pad_id] * pad + ids)
        batch_mask.append([0] * pad + [1] * len(ids))
    return (
        torch.tensor(batch_ids, dtype=torch.long, device="cuda"),
        torch.tensor(batch_mask, dtype=torch.long, device="cuda"),
    )

def run_dflash_batch(batch_size):
    """跑全部 N 条, 返回 outputs[sample_idx] 和 step_acc[sample_idx]"""
    outputs = [None] * N
    step_accs = [None] * N
    n_batches = (N + batch_size - 1) // batch_size
    t0 = time.time()

    for b in range(n_batches):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, N)
        indices = list(range(start_idx, end_idx))
        actual_bs = len(indices)

        ids, mask = build_batch(indices)
        with torch.inference_mode():
            stats = dflash_generate(
                drafter, target=target, input_ids=ids,
                max_new_tokens=MAX_NEW, stop_token_ids=None,
                temperature=0.0, return_stats=True,
                ignore_eos=True, attention_mask=mask,
            )

        for local_i, global_i in enumerate(indices):
            outputs[global_i] = stats.output_ids[local_i, ids.shape[1]:ids.shape[1]+MAX_NEW].tolist()

        # 每个 step 的 raw_draft_accept 是 per-sample 的
        step_accs_per_sample = [[] for _ in range(actual_bs)]
        for step in stats.acceptance_lengths:
            raw_list = step["raw_draft_accept"]
            used = step["used_draft_accept"]
            advance = step["advance_tokens"]
            for s in range(actual_bs):
                step_accs_per_sample[s].append({
                    "raw": raw_list[s] if s < len(raw_list) else 0,
                    "used": used,
                    "advance": advance,
                })
        for local_i, global_i in enumerate(indices):
            step_accs[global_i] = step_accs_per_sample[local_i]

    print(f"  DFlash b{batch_size}: {n_batches} batches, {time.time()-t0:.0f}s")
    return outputs, step_accs

# 跑 batch=2, 4, 8
df_outputs = {}
df_step_accs = {}
for bs in [2, 4, 8]:
    df_outputs[bs], df_step_accs[bs] = run_dflash_batch(bs)

# ── 验证点 5: padding 语义不变 — batch=1 vs batch=8 输出对比 ──
print(f"\n验证点5: padding/mask 语义 (batch=1 vs batch=8 输出)")
v5_exact = 0
v5_token_match = 0
v5_total = 0
v5_mm_samples = []
for i in range(N):
    out1 = df1_outputs[i]
    out8 = df_outputs[8][i]
    n = min(len(out1), len(out8))
    v5_total += n
    mm = False
    for j in range(n):
        if out1[j] == out8[j]:
            v5_token_match += 1
        else:
            mm = True
    if not mm:
        v5_exact += 1
    else:
        if len(v5_mm_samples) < 5:
            first_mm = next((j for j in range(n) if out1[j] != out8[j]), -1)
            v5_mm_samples.append((i, first_mm))

print(f"  逐 token match: {v5_token_match}/{v5_total} ({v5_token_match/v5_total*100:.2f}%)")
print(f"  完全一致样本: {v5_exact}/{N}")
if v5_mm_samples:
    print(f"  前5个 mismatch: {v5_mm_samples}")

# 同样对比 batch=1 vs batch=2, batch=1 vs batch=4
for bs in [2, 4]:
    match_count = 0
    total_count = 0
    exact = 0
    for i in range(N):
        out1 = df1_outputs[i]
        out_b = df_outputs[bs][i]
        n = min(len(out1), len(out_b))
        total_count += n
        mm = False
        for j in range(n):
            if out1[j] == out_b[j]:
                match_count += 1
            else:
                mm = True
        if not mm:
            exact += 1
    print(f"  batch=1 vs batch={bs}: token match {match_count}/{total_count} "
          f"({match_count/total_count*100:.2f}%), exact {exact}/{N}")

# ── 验证点 2: batch 内外 raw acceptance ──
print(f"\n验证点2: raw_accept batch 内外一致性")
# batch=1 的 per-sample raw
df1_raw_means = []
for i in range(N):
    raws = [s["raw_draft_accept"][0] for s in df1_step_acc[i]]
    df1_raw_means.append(sum(raws) / len(raws) if raws else 0)

for bs in [2, 4, 8]:
    batch_raw_means = []
    for i in range(N):
        raws = [s["raw"] for s in df_step_accs[bs][i]]
        batch_raw_means.append(sum(raws) / len(raws) if raws else 0)

    diffs = [abs(a - b) for a, b in zip(df1_raw_means, batch_raw_means)]
    mean_diff = sum(diffs) / len(diffs) if diffs else 0
    max_diff = max(diffs) if diffs else 0
    close_count = sum(1 for d in diffs if d < 0.5)
    print(f"  batch=1 vs batch={bs}: mean_diff={mean_diff:.3f}, max_diff={max_diff:.3f}, "
          f"|diff|<0.5: {close_count}/{N}")

# ============================================================
# Phase 5: 汇总
# ============================================================
print(f"\n{'='*60}")
print("汇总")
print(f"{'='*60}")

results = {
    "config": {"dataset": "GSM8K-128", "max_new_tokens": MAX_NEW, "temperature": 0.0},
    "v1_advance_logic": {"pass": v1_pass, "raw0_steps": v1_raw0_count,
                         "raw_ge1_steps": v1_raw_ge1_count, "total_steps": total_steps},
    "v3_ar_vs_dflash_b1": {
        "token_match_ratio": round(prefix_match_ratio, 4),
        "exact_match": exact_match,
        "total": N,
    },
    "v4_raw0_bonus": {"pass": v4_pass},
    "v5_padding_semantics_b1_vs_b8": {
        "token_match_ratio": round(v5_token_match / v5_total, 4) if v5_total else 0,
        "exact_match": v5_exact,
    },
}

# raw_accept 统计
for label, step_data in [("b1", df1_step_acc)]:
    all_raw = []
    all_advance = []
    for i in range(N):
        for s in step_data[i]:
            all_raw.extend(s["raw_draft_accept"])
            all_advance.append(s["advance_tokens"])
    results[f"raw_accept_{label}"] = {
        "mean": round(sum(all_raw) / len(all_raw), 3) if all_raw else 0,
        "advance_mean": round(sum(all_advance) / len(all_advance), 3) if all_advance else 0,
    }

with open("/root/autodl-tmp/litedrafter/outputs/step_a_regression.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\n输出: /root/autodl-tmp/litedrafter/outputs/step_a_regression.json")
