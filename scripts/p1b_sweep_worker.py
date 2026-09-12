#!/usr/bin/env python3
"""
p1b_sweep_worker.py - 单配置 batch sweep worker

Usage: python p1b_sweep_worker.py <mode> <variant> <batch_size> [run_id]

Args:
  mode:      "ar" | "dflash"
  variant:   "raw" | "ctx1024" | "ctx2048" | "ctx4096"
  batch_size: 1/2/4/8/16/32
  run_id:    1-based run number (default 1)

Output:
  outputs/p1b_{mode}_{variant}_b{batch}_r{run}.json

Keys: success, peak_allocated_mb, peak_reserved_mb, total_latency_s,
      num_output_tokens, throughput_tok_s, tpot_ms, headroom_mb,
      oom, num_batches, acceptance_stats (DFlash only)
"""

import gc
import json
import os
import sys
import time
import torch

# ---- Config ----
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DRAFTER_PATH = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
DATA_DIR = "/root/autodl-tmp/litedrafter/data"
OUTPUT_DIR = "/root/autodl-tmp/litedrafter/outputs"
DTYPE = torch.bfloat16
MAX_NEW = 256
TEMPERATURE = 0.0
ATTN = "sdpa"
GPU_MEM_MB = 24564  # RTX 4090


def mem_now():
    torch.cuda.synchronize()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
    }


def peak_full():
    torch.cuda.synchronize()
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576, 1),
    }


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def load_and_sort(data_file):
    """Load dataset, sort by token length ascending."""
    with open(data_file) as f:
        items = [json.loads(line) for line in f]
    items.sort(key=lambda x: x["token_len"])
    return items


def collate_batch(items, tokenizer, device="cuda"):
    """Left-pad a group of items to max length in the group."""
    encoded = [tokenizer.encode(it["prompt"], add_special_tokens=True) for it in items]
    max_len = max(len(e) for e in encoded)
    batch_ids = []
    batch_mask = []
    for e in encoded:
        pad = max_len - len(e)
        batch_ids.append([tokenizer.pad_token_id] * pad + e)
        batch_mask.append([0] * pad + [1] * len(e))
    return (
        torch.tensor(batch_ids, dtype=torch.long, device=device),
        torch.tensor(batch_mask, dtype=torch.long, device=device),
    )


def run_ar(tokenizer, items, batch_size):
    from transformers import AutoModelForCausalLM

    target = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, attn_implementation=ATTN,
    ).to("cuda").eval()

    # Group into batches
    batches = []
    for i in range(0, len(items), batch_size):
        group = items[i:i + batch_size]
        input_ids, attn_mask = collate_batch(group, tokenizer)
        batches.append((input_ids, attn_mask))

    print(f"  [{len(batches)} batches, batch_size <= {batch_size}]")

    total_latency = 0.0
    total_out = 0
    peak_pa, peak_pr = 0.0, 0.0
    start_all = time.perf_counter()

    for bi, (inp, mask) in enumerate(batches):
        reset_peak()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = target.generate(
                inp, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, attention_mask=mask,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        pk = peak_full()
        peak_pa = max(peak_pa, pk["peak_allocated_mb"])
        peak_pr = max(peak_pr, pk["peak_reserved_mb"])
        n_out = (out.shape[1] - inp.shape[1]) * inp.shape[0]
        total_latency += elapsed
        total_out += n_out
        del out, inp, mask
        gc.collect()

        if (bi + 1) % max(1, len(batches) // 5) == 0:
            print(f"    batch {bi+1}/{len(batches)}: peak_resv={pk['peak_reserved_mb']:.0f}")

    total_wall = time.perf_counter() - start_all
    tpot = total_latency / max(total_out, 1) * 1000
    throughput = total_out / max(total_wall, 0.001)
    headroom = GPU_MEM_MB - peak_pr

    del target
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "success": True, "oom": False,
        "peak_allocated_mb": peak_pa, "peak_reserved_mb": peak_pr,
        "total_latency_s": round(total_latency, 2),
        "total_wall_s": round(total_wall, 2),
        "num_output_tokens": total_out,
        "throughput_tok_s": round(throughput, 1),
        "tpot_ms": round(tpot, 2),
        "headroom_mb": round(headroom, 1),
        "num_batches": len(batches),
    }


def run_dflash(tokenizer, items, batch_size):
    from transformers import AutoModelForCausalLM
    from dflash.model import DFlashDraftModel, dflash_generate

    target = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, attn_implementation=ATTN,
    ).to("cuda").eval()

    drafter = DFlashDraftModel.from_pretrained(
        DRAFTER_PATH, dtype=DTYPE, attn_implementation=ATTN,
    ).to("cuda").eval()

    # Test batch=1 warmup
    group0 = items[:1]
    inp0, mask0 = collate_batch(group0, tokenizer)
    with torch.inference_mode():
        _ = dflash_generate(drafter, target=target, input_ids=inp0,
                            max_new_tokens=8, stop_token_ids=None,
                            temperature=TEMPERATURE, return_stats=False,
                            ignore_eos=True, attention_mask=mask0)
    del inp0, mask0, _
    gc.collect()
    torch.cuda.empty_cache()

    batches = []
    for i in range(0, len(items), batch_size):
        group = items[i:i + batch_size]
        inp, mask = collate_batch(group, tokenizer)
        batches.append((inp, mask))

    print(f"  [{len(batches)} batches, batch_size <= {batch_size}]")

    total_latency = 0.0
    total_out = 0
    peak_pa, peak_pr = 0.0, 0.0
    all_acceptance = []  # list of {"used": int, "raw": [int]}
    start_all = time.perf_counter()

    for bi, (inp, mask) in enumerate(batches):
        reset_peak()
        t0 = time.perf_counter()
        with torch.inference_mode():
            stats = dflash_generate(
                drafter, target=target, input_ids=inp,
                max_new_tokens=MAX_NEW, stop_token_ids=None,
                temperature=TEMPERATURE, return_stats=True,
                ignore_eos=True, attention_mask=mask,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        pk = peak_full()
        peak_pa = max(peak_pa, pk["peak_allocated_mb"])
        peak_pr = max(peak_pr, pk["peak_reserved_mb"])
        total_latency += elapsed
        total_out += stats.num_output_tokens
        all_acceptance.extend(stats.acceptance_lengths)
        del stats, inp, mask
        gc.collect()

        if (bi + 1) % max(1, len(batches) // 5) == 0:
            print(f"    batch {bi+1}/{len(batches)}: peak_resv={pk['peak_reserved_mb']:.0f}")

    total_wall = time.perf_counter() - start_all
    tpot = total_latency / max(total_out, 1) * 1000
    throughput = total_out / max(total_wall, 0.001)
    headroom = GPU_MEM_MB - peak_pr

    # Aggregate acceptance stats
    used_vals = [a["used"] for a in all_acceptance]
    raw_vals = [v for a in all_acceptance for v in a["raw"]]

    import statistics
    def pct(data, p):
        s = sorted(data)
        return s[int(len(s) * p / 100)] if s else 0

    acceptance_stats = {
        "used_mean": round(statistics.mean(used_vals), 2) if used_vals else 0,
        "used_median": round(statistics.median(used_vals), 2) if used_vals else 0,
        "used_p10": round(pct(used_vals, 10), 2),
        "used_p90": round(pct(used_vals, 90), 2),
        "raw_mean": round(statistics.mean(raw_vals), 2) if raw_vals else 0,
        "raw_median": round(statistics.median(raw_vals), 2) if raw_vals else 0,
        "raw_p10": round(pct(raw_vals, 10), 2),
        "raw_p90": round(pct(raw_vals, 90), 2),
        "raw_histogram": {},  # filled later by orchestrator
    }

    del target, drafter
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "success": True, "oom": False,
        "peak_allocated_mb": peak_pa, "peak_reserved_mb": peak_pr,
        "total_latency_s": round(total_latency, 2),
        "total_wall_s": round(total_wall, 2),
        "num_output_tokens": total_out,
        "throughput_tok_s": round(throughput, 1),
        "tpot_ms": round(tpot, 2),
        "headroom_mb": round(headroom, 1),
        "num_batches": len(batches),
        "acceptance_stats": acceptance_stats,
        "acceptance_raw": all_acceptance,  # full list for histogram
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: p1b_sweep_worker.py <mode> <variant> <batch_size> [run_id]")
        sys.exit(1)

    mode = sys.argv[1]
    variant = sys.argv[2]
    batch_size = int(sys.argv[3])
    run_id = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    data_file = os.path.join(DATA_DIR, f"he164_{variant}.jsonl")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[{mode}] {variant} batch={batch_size} run={run_id}")
    items = load_and_sort(data_file)

    try:
        if mode == "ar":
            result = run_ar(tokenizer, items, batch_size)
        else:
            result = run_dflash(tokenizer, items, batch_size)
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        result = {"success": False, "oom": True, "batch_size": batch_size}
    except Exception as e:
        result = {"success": False, "oom": False, "error": str(e)}

    result["mode"] = mode
    result["variant"] = variant
    result["batch_size"] = batch_size
    result["run_id"] = run_id

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = f"p1b_{mode}_{variant}_b{batch_size}_r{run_id}.json"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    status = "OK" if result["success"] else ("OOM" if result.get("oom") else "ERR")
    print(f"  [{status}] → {out_name}")


if __name__ == "__main__":
    main()
