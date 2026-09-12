#!/usr/bin/env python3
"""
修复后正确性验证 + 新指标验证
同一个 prompt：AR greedy vs DFlash (batch=1/2/4/8)
验证：1) token match rate 是否提升  2) 新指标是否正确输出
"""
import gc, json, torch, transformers
from dflash.model import DFlashDraftModel, dflash_generate

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DRAFTER_PATH = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
DATA_FILE = "/root/autodl-tmp/litedrafter/data/coig_cqia_buckets.jsonl"
DTYPE = torch.bfloat16
MAX_NEW = 128

with open(DATA_FILE) as f:
    items = [json.loads(line) for line in f]
candidates = [it for it in items if it.get("bucket") == 1024]
prompt_ids = candidates[0]["token_ids"]

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
print(f"  AR: {len(ar_new)} tokens")

# ── DFlash 各 batch ──
def run_dflash(batch_size):
    batch_ids = [prompt_ids for _ in range(batch_size)]
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

    # 新格式聚合
    all_raw = []
    all_used = []
    all_advance = []
    zero_min_steps = 0

    for acc in stats.acceptance_lengths:
        all_raw.extend(acc["raw_draft_accept"])
        all_used.append(acc["used_draft_accept"])
        all_advance.append(acc["advance_tokens"])
        if acc["used_draft_accept"] == 0:
            zero_min_steps += 1

    num_steps = len(all_advance)
    raw_accept_mean = sum(all_raw) / len(all_raw) if all_raw else 0
    mean_advance = sum(all_advance) / num_steps if num_steps else 0

    total_validated = sum(all_raw)
    total_utilized = batch_size * sum(all_used)
    sync_eff = (total_utilized / total_validated) if total_validated > 0 else 0
    waste = total_validated - total_utilized
    zero_min_rate = zero_min_steps / num_steps if num_steps else 0

    return df_new, {
        "num_steps": num_steps,
        "raw_accept_mean": raw_accept_mean,
        "mean_advance_tokens": mean_advance,
        "sync_efficiency": sync_eff,
        "accepted_token_waste": waste,
        "zero_min_rate": zero_min_rate,
    }


print("\n=== DFlash 修复后各 batch 对比 ===\n")

for bs in [1, 2, 4, 8]:
    df_new, m = run_dflash(bs)
    min_len = min(len(ar_new), len(df_new))
    matches = sum(1 for i in range(min_len) if ar_new[i] == df_new[i])
    first_mismatch = None
    for i in range(min_len):
        if ar_new[i] != df_new[i]:
            first_mismatch = i
            break

    print(f"--- batch={bs} ---")
    print(f"  Token match: {matches}/{min_len} ({matches/min_len*100:.1f}%)")
    print(f"  First mismatch at: {first_mismatch}")
    print(f"  raw_accept_mean:     {m['raw_accept_mean']:.2f}")
    print(f"  mean_advance_tokens: {m['mean_advance_tokens']:.2f}")
    print(f"  sync_efficiency:     {m['sync_efficiency']:.4f}")
    print(f"  accepted_token_waste:{m['accepted_token_waste']}")
    print(f"  zero_min_rate:       {m['zero_min_rate']:.4f}")

    if first_mismatch is not None and first_mismatch < 5:
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

print("=" * 60)
