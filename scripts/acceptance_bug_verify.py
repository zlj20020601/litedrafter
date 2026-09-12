#!/usr/bin/env python3
"""
验证 max(1, per_sample_accept.min()) 导致的正确性问题。
同一个 prompt：
  1. AR greedy = ground truth
  2. DFlash batch=1/2/4/8，逐 token 对比
  3. 统计每步 raw acceptance，看 min==0 频率
"""
import gc, json, torch, transformers
from dflash.model import DFlashDraftModel, dflash_generate

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DRAFTER_PATH = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
DATA_FILE = "/root/autodl-tmp/litedrafter/data/coig_cqia_buckets.jsonl"
DTYPE = torch.bfloat16
MAX_NEW = 128

# ── 加载数据 ──
with open(DATA_FILE) as f:
    items = [json.loads(line) for line in f]
candidates = [it for it in items if it.get("bucket") == 1024]
prompt_ids = candidates[0]["token_ids"]

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── 加载模型 ──
print("Loading target...")
target = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=DTYPE, attn_implementation="sdpa"
).to("cuda").eval()

print("Loading drafter...")
drafter = DFlashDraftModel.from_pretrained(
    DRAFTER_PATH, dtype=DTYPE, attn_implementation="sdpa"
).to("cuda").eval()
print(f"block_size={drafter.block_size}")

# ── AR greedy ground truth ──
input_ids_1 = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
attn_mask_1 = torch.ones_like(input_ids_1)

print("\n=== AR greedy (ground truth) ===")
with torch.inference_mode():
    ar_out = target.generate(
        input_ids_1, max_new_tokens=MAX_NEW, do_sample=False,
        pad_token_id=tokenizer.eos_token_id, attention_mask=attn_mask_1,
    )
ar_new = ar_out[0, input_ids_1.shape[1]:].tolist()
ar_text = tokenizer.decode(ar_new, skip_special_tokens=True)
print(f"  AR: {len(ar_new)} tokens")

# ── DFlash 各 batch ──
def run_dflash(batch_size):
    """跑 DFlash，返回 (output_tokens_sample0, acceptance_stats)"""
    # 构造 batch（同一个 prompt 复制）
    ids = prompt_ids.copy()
    batch_ids = [ids for _ in range(batch_size)]
    max_len = max(len(x) for x in batch_ids)
    padded = []
    masks = []
    for x in batch_ids:
        pad = max_len - len(x)
        padded.append([tokenizer.pad_token_id] * pad + x)
        masks.append([0] * pad + [1] * len(x))
    input_ids = torch.tensor(padded, dtype=torch.long, device="cuda")
    attn_mask = torch.tensor(masks, dtype=torch.long, device="cuda")

    with torch.inference_mode():
        stats = dflash_generate(
            drafter, target=target, input_ids=input_ids,
            max_new_tokens=MAX_NEW, stop_token_ids=None,
            temperature=0.0, return_stats=True,
            ignore_eos=True, attention_mask=attn_mask,
        )

    df_new = stats.output_ids[0, input_ids.shape[1]:].tolist()

    # 解析 acceptance
    used_vals = []
    raw_min_vals = []     # 每步 raw 的 min
    raw_all = []          # 每步所有样本的 raw
    zero_min_count = 0
    bug_trigger_count = 0  # min==0 但被 max(1,..) 强制为 1 的步数

    for acc in stats.acceptance_lengths:
        used = acc["used"]
        raw = acc["raw"]
        used_vals.append(used)
        raw_min = min(raw)
        raw_min_vals.append(raw_min)
        raw_all.append(raw)

        if raw_min == 0:
            zero_min_count += 1
            # bug: max(1, 0) = 1, advance=2 instead of advance=1
            bug_trigger_count += 1

    return df_new, {
        "num_steps": len(used_vals),
        "used_mean": sum(used_vals) / len(used_vals),
        "raw_min_mean": sum(raw_min_vals) / len(raw_min_vals),
        "zero_min_count": zero_min_count,
        "bug_trigger_count": bug_trigger_count,
        "bug_trigger_rate": bug_trigger_count / len(used_vals),
    }


print("\n=== DFlash 各 batch 对比 ===\n")

for bs in [1, 2, 4, 8]:
    df_new, acc_stats = run_dflash(bs)

    # 逐 token 对比 AR
    min_len = min(len(ar_new), len(df_new))
    matches = sum(1 for i in range(min_len) if ar_new[i] == df_new[i])
    mismatches = min_len - matches
    first_mismatch = None
    for i in range(min_len):
        if ar_new[i] != df_new[i]:
            first_mismatch = i
            break

    print(f"--- batch={bs} ---")
    print(f"  AR tokens: {len(ar_new)}, DFlash tokens: {len(df_new)}")
    print(f"  Token match: {matches}/{min_len} ({matches/min_len*100:.1f}%)")
    print(f"  First mismatch at position: {first_mismatch}")
    print(f"  Acceptance steps: {acc_stats['num_steps']}")
    print(f"  Used acc mean: {acc_stats['used_mean']:.2f}")
    print(f"  Raw min acc mean: {acc_stats['raw_min_mean']:.2f}")
    print(f"  Steps with raw_min==0: {acc_stats['zero_min_count']}/{acc_stats['num_steps']} "
          f"({acc_stats['bug_trigger_rate']*100:.1f}%)")
    print(f"  >>> Bug trigger count (max(1,0)=1, advance=2): {acc_stats['bug_trigger_count']}")

    # 显示前几个 mismatch 的具体 token
    if first_mismatch is not None:
        print(f"  --- 前3个 mismatch ---")
        shown = 0
        for i in range(min_len):
            if ar_new[i] != df_new[i]:
                ar_tok = tokenizer.decode([ar_new[i]])
                df_tok = tokenizer.decode([df_new[i]])
                print(f"    pos={i}: AR='{ar_tok}' vs DFlash='{df_tok}'")
                shown += 1
                if shown >= 3:
                    break
    print()

# ── 关键结论 ──
print("=" * 60)
print("结论：")
print("  max(1, per_sample_accept.min()) 在 raw acceptance=0 时：")
print("    代码执行: acceptance_length=1, advance=2")
print("    正确应为: acceptance_length=0, advance=1")
print("    后果: 被拒绝的 draft token 被写入 output_ids")
print("  这导致 DFlash 输出与 AR greedy 不一致。")
print("=" * 60)
