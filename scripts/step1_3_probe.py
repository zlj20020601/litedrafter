#!/usr/bin/env python3
"""
Step 1.3: draft/verify 显存插桩

通过 monkey-patch dflash_generate,在各阶段记录显存。
不修改上游源码,只读取显存状态。

阶段记录点:
  M2_drafter_loaded      — drafter 加载后基线
  prefill                — target prefill 后峰值
  draft_forward          — 每个 decode step 的 draft forward 后峰值
  verify_forward         — 每个 decode step 的 verify forward 后峰值
  accept_update          — accept/update 后峰值
  global_peak            — 全程峰值

每步记录: current_allocated, current_reserved, max_allocated_so_far, max_reserved_so_far, nvidia-smi
"""

import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import torch

# ── 配置 ──────────────────────────────────────────────
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DRAFTER_PATH = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
DATA_FILE = "/root/autodl-tmp/litedrafter/data/coig_cqia_buckets.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/litedrafter/outputs"

DTYPE = torch.bfloat16
INPUT_LEN = 1024
OUTPUT_LEN = 256
BATCH_SIZE = 1  # 可通过 --batch N 覆盖

TODAY = datetime.now().strftime("%Y%m%d")
NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ── 显存工具 ──────────────────────────────────────────
def nvidia_smi_process_mem():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        pid = os.getpid()
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) == 2 and int(parts[0].strip()) == pid:
                return round(float(parts[1].strip()), 1)
        return 0.0
    except Exception:
        return -1.0


class MemoryProbe:
    """在各阶段采集显存快照"""

    def __init__(self):
        self.probes = []  # list of {stage, step, allocated, reserved, max_alloc, max_resv, nvidia}
        self._step_counter = {"prefill": 0, "draft": 0, "verify": 0, "accept": 0}

    def record(self, stage, label=""):
        torch.cuda.synchronize()
        entry = {
            "stage": stage,
            "step": self._step_counter.get(stage, 0),
            "label": label,
            "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576, 1),
            "nvidia_smi_mb": nvidia_smi_process_mem(),
            "timestamp": round(time.perf_counter(), 3),
        }
        self.probes.append(entry)
        self._step_counter[stage] = self._step_counter.get(stage, 0) + 1
        return entry

    def summary_by_stage(self):
        """每个 stage 取最大 max_allocated"""
        stages = {}
        for p in self.probes:
            s = p["stage"]
            if s not in stages or p["max_allocated_mb"] > stages[s]["max_allocated_mb"]:
                stages[s] = p
        return stages


# ── patched dflash_generate ──────────────────────────
def patched_dflash_generate(model, target, input_ids, max_new_tokens,
                            stop_token_ids=None, temperature=0.0,
                            block_size=None, mask_token_id=None,
                            return_stats=False, ignore_eos=False,
                            attention_mask=None,
                            probe=None):
    """
    完整复制 dflash_generate 逻辑,在关键点插入 probe.record()
    保持原始执行路径不变。
    """
    from transformers import DynamicCache
    from dflash.model import sample, extract_context_feature
    from types import SimpleNamespace

    batch_size = input_ids.shape[0]
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full(
        (batch_size, max_length + block_size), mask_token_id,
        dtype=torch.long, device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # ── prefill ──
    prefill_pos = position_ids[:, :num_input_tokens]
    output = target(
        input_ids,
        position_ids=prefill_pos,
        attention_mask=attention_mask,
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )
    if probe:
        probe.record("prefill", "target prefill")

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)

    target_hidden = None
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    # ── decode loop ──
    acceptance_lengths = []
    start = num_input_tokens
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start:start + block_size].clone()
        block_position_ids = position_ids[:, start:start + block_size]

        # ── draft forward ──
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length():start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, 1 - block_size:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if probe:
                probe.record("draft", f"step {start}")

        # ── verify forward ──
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )
        if probe:
            probe.record("verify", f"step {start}")

        posterior = sample(output.logits, temperature)
        per_sample_accept = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)
        raw_draft_accept = per_sample_accept.clone().tolist()
        used_draft_accept = int(per_sample_accept.min().item())
        advance = used_draft_accept + 1

        output_ids[:, start:start + used_draft_accept + 1] = block_output_ids[:, :used_draft_accept + 1]
        output_ids[:, start + used_draft_accept + 1] = posterior[:, used_draft_accept]
        start += advance
        past_key_values_target.crop(start)

        if probe:
            probe.record("accept", f"step {start}")

        acceptance_lengths.append({
            "raw_draft_accept": raw_draft_accept,
            "used_draft_accept": used_draft_accept,
            "advance_tokens": advance,
        })

        if block_size > 1 and start < max_length:
            target_hidden = extract_context_feature(
                output.hidden_states, model.target_layer_ids
            )[:, :used_draft_accept + 1, :]

        if not ignore_eos and stop_token_ids is not None:
            if any(stop_token_id in output_ids[:, num_input_tokens:]
                   for stop_token_id in stop_token_ids):
                break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    num_output_tokens = (output_ids.shape[1] - num_input_tokens) * batch_size

    stats = SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        acceptance_lengths=acceptance_lengths,
    )
    return stats if return_stats else output_ids


# ── 主流程 ────────────────────────────────────────────
def main():
    import transformers

    # 解析 batch 参数
    batch_size = BATCH_SIZE
    for i, arg in enumerate(sys.argv):
        if arg == "--batch" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])

    print("=" * 60)
    print(f"  Step 1.3: draft/verify 显存插桩 (batch={batch_size})")
    print(f"  {NOW}")
    print("=" * 60)

    # 加载数据
    with open(DATA_FILE) as f:
        items = [json.loads(line) for line in f]
    candidates = [it for it in items if it.get("bucket") == INPUT_LEN]

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    m0 = {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
        "nvidia_smi_mb": nvidia_smi_process_mem(),
    }
    print(f"  [M0] alloc={m0['allocated_mb']:.0f} resv={m0['reserved_mb']:.0f} nvsmi={m0['nvidia_smi_mb']:.0f}")

    target = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, attn_implementation="sdpa"
    ).to("cuda").eval()
    m1 = {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
        "nvidia_smi_mb": nvidia_smi_process_mem(),
    }
    print(f"  [M1] alloc={m1['allocated_mb']:.0f} resv={m1['reserved_mb']:.0f} nvsmi={m1['nvidia_smi_mb']:.0f}")

    from dflash.model import DFlashDraftModel
    drafter = DFlashDraftModel.from_pretrained(
        DRAFTER_PATH, dtype=DTYPE, attn_implementation="sdpa"
    ).to("cuda").eval()
    m2 = {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
        "nvidia_smi_mb": nvidia_smi_process_mem(),
    }
    print(f"  [M2] alloc={m2['allocated_mb']:.0f} resv={m2['reserved_mb']:.0f} nvsmi={m2['nvidia_smi_mb']:.0f}")
    print(f"  block_size: {drafter.block_size}")

    # 准备输入 (batch 构造,循环复用样本)
    batch_items = [candidates[i % len(candidates)] for i in range(batch_size)]
    all_ids = [it["token_ids"] for it in batch_items]
    max_len = max(len(ids) for ids in all_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batch_ids = []
    batch_mask = []
    for ids in all_ids:
        pad = max_len - len(ids)
        batch_ids.append([pad_id] * pad + ids)
        batch_mask.append([0] * pad + [1] * len(ids))
    input_ids = torch.tensor(batch_ids, dtype=torch.long, device="cuda")
    attn_mask = torch.tensor(batch_mask, dtype=torch.long, device="cuda")
    eos_id = tokenizer.eos_token_id
    print(f"  batch={batch_size}, input tokens: {input_ids.shape[1]}, output_len: {OUTPUT_LEN}")

    # 初始化 probe
    probe = MemoryProbe()
    torch.cuda.reset_peak_memory_stats()

    # 运行 patched dflash_generate
    print(f"\n  运行 patched dflash_generate (batch={batch_size}, input_len={INPUT_LEN})...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        stats = patched_dflash_generate(
            drafter, target=target, input_ids=input_ids,
            max_new_tokens=OUTPUT_LEN, stop_token_ids=[eos_id],
            temperature=0.0, return_stats=True,
            ignore_eos=True, attention_mask=attn_mask,
            probe=probe,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  完成: {stats.num_output_tokens} tokens, {elapsed:.2f}s")

    # ── 分析 ──
    stage_summary = probe.summary_by_stage()

    print("\n" + "=" * 70)
    print("  显存插桩结果 (batch=1, input_len=1024)")
    print("=" * 70)

    print(f"\n  {'阶段':>12} | {'curr_alloc':>10} {'curr_resv':>10} | "
          f"{'max_alloc':>10} {'max_resv':>10} {'nvsmi':>8} | {'Δ_vs_M2':>8}")
    print("  " + "-" * 78)

    base_alloc = m2["allocated_mb"]
    for stage in ["prefill", "draft", "verify", "accept"]:
        if stage in stage_summary:
            p = stage_summary[stage]
            delta = p["max_allocated_mb"] - base_alloc
            print(f"  {stage:>12} | {p['allocated_mb']:>10.0f} {p['reserved_mb']:>10.0f} | "
                  f"{p['max_allocated_mb']:>10.0f} {p['max_reserved_mb']:>10.0f} "
                  f"{p['nvidia_smi_mb']:>8.0f} | {delta:>+8.0f}")

    global_max_alloc = max(p["max_allocated_mb"] for p in probe.probes)
    global_max_resv = max(p["max_reserved_mb"] for p in probe.probes)
    global_max_nvsmi = max(p["nvidia_smi_mb"] for p in probe.probes)
    print(f"  {'global peak':>12} | {'':>10} {'':>10} | "
          f"{global_max_alloc:>10.0f} {global_max_resv:>10.0f} "
          f"{global_max_nvsmi:>8.0f} | {global_max_alloc - base_alloc:>+8.0f}")

    # 一致性检查
    max_of_stages = max(
        stage_summary[s]["max_allocated_mb"]
        for s in ["prefill", "draft", "verify", "accept"]
        if s in stage_summary
    )
    print(f"\n  一致性检查:")
    print(f"    max(prefill, draft, verify, accept) = {max_of_stages:.0f} MB")
    print(f"    global peak                        = {global_max_alloc:.0f} MB")
    print(f"    一致: {'✓' if abs(max_of_stages - global_max_alloc) < 50 else '✗ (差值>50MB)'}")

    # 每阶段采样数
    print(f"\n  采样统计:")
    for stage in ["prefill", "draft", "verify", "accept"]:
        count = sum(1 for p in probe.probes if p["stage"] == stage)
        print(f"    {stage}: {count} 次")

    # acceptance
    all_raw = [v for a in stats.acceptance_lengths for v in a["raw_draft_accept"]]
    all_advance = [a["advance_tokens"] for a in stats.acceptance_lengths]
    raw_accept_mean = sum(all_raw) / len(all_raw) if all_raw else 0
    mean_advance = sum(all_advance) / len(all_advance) if all_advance else 0
    zero_min_steps = sum(1 for a in stats.acceptance_lengths if a["used_draft_accept"] == 0)
    print(f"\n  acceptance: raw_mean={raw_accept_mean:.1f} advance_mean={mean_advance:.1f} "
          f"zero_min={zero_min_steps}/{len(all_advance)} steps")

    # ── 写结果 ──
    result = {
        "timestamp": NOW,
        "config": {
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "batch": batch_size,
            "dtype": str(DTYPE),
        },
        "stages": {"m0": m0, "m1": m1, "m2": m2},
        "stage_summary": {s: p for s, p in stage_summary.items()},
        "global_peak": {
            "max_allocated_mb": global_max_alloc,
            "max_reserved_mb": global_max_resv,
            "nvidia_smi_mb": global_max_nvsmi,
        },
        "consistency_check": {
            "max_of_stages": max_of_stages,
            "global_peak": global_max_alloc,
            "consistent": abs(max_of_stages - global_max_alloc) < 50,
        },
        "acceptance": {
            "raw_accept_mean": round(raw_accept_mean, 2),
            "mean_advance_tokens": round(mean_advance, 2),
            "zero_min_rate": round(zero_min_steps / len(all_advance), 4) if all_advance else 0,
            "num_steps": len(all_advance),
        },
        "all_probes": probe.probes,
    }

    out_path = os.path.join(OUTPUT_DIR, f"step13_stage_probe_b{batch_size}_{TODAY}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  输出: {out_path}")


if __name__ == "__main__":
    main()
