#!/usr/bin/env python3
"""
p1b_batch_scan.py - Step 1.2: 小规模 batch 压力扫描

HumanEval 前 8 条, left-pad 到批内最长, batch=1/2/4/8
AR vs DFlash, 每个 batch 跑 1 次
记录 OOM 边界 + batch 级性能
"""

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DRAFTER_PATH = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
DATA_PATH = "/root/autodl-tmp/litedrafter/data/humaneval_prompts_full.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/litedrafter/outputs"
DTYPE = torch.bfloat16
DEVICE = "cuda"
BATCH_SIZES = [1, 2, 4, 8]
MAX_NEW_TOKENS = 256
ATTN_IMPL = "sdpa"


def mem_now():
    torch.cuda.synchronize()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
    }


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def get_peak_full():
    torch.cuda.synchronize()
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576, 1),
    }


def load_prompts(n):
    """Load first n HumanEval prompts, return list of token tensors."""
    with open(DATA_PATH) as f:
        lines = [json.loads(line) for line in f]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for item in lines[:n]:
        ids = tokenizer.encode(item["prompt"], add_special_tokens=True)
        prompts.append({"id": item["id"], "token_ids": ids, "text": item["prompt"]})
    return prompts, tokenizer


def collate_batch(prompts, tokenizer):
    """Left-pad prompts to max length in batch, return tensor [batch, max_len]."""
    max_len = max(len(p["token_ids"]) for p in prompts)
    batch_ids = []
    for p in prompts:
        ids = p["token_ids"]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = [tokenizer.pad_token_id] * pad_len + ids
        batch_ids.append(ids)
    return torch.tensor(batch_ids, dtype=torch.long, device=DEVICE)


def run_ar_baseline(tokenizer, all_prompts):
    from transformers import AutoModelForCausalLM

    print("\n" + "=" * 60)
    print("  AR Baseline: batch scan")
    print("=" * 60)

    M0 = mem_now()
    print(f"  [M0] Empty: alloc={M0['allocated_mb']:.0f} resv={M0['reserved_mb']:.0f}")

    target = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, attn_implementation=ATTN_IMPL,
    ).to(DEVICE).eval()

    M1 = mem_now()
    print(f"  [M1] Target: alloc={M1['allocated_mb']:.0f} resv={M1['reserved_mb']:.0f}")

    results = []
    for batch_size in BATCH_SIZES:
        prompts = all_prompts[:batch_size]
        input_ids = collate_batch(prompts, tokenizer)
        total_tokens = input_ids.shape[0] * input_ids.shape[1]
        print(f"\n  batch={batch_size} ({batch_size}×{input_ids.shape[1]}={total_tokens} tokens)")

        try:
            reset_peak()
            t0 = time.perf_counter()
            with torch.inference_mode():
                output = target.generate(
                    input_ids, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak = get_peak_full()

            n_input = input_ids.shape[1]
            n_out_total = output.shape[1] - n_input
            tpot = elapsed / max(n_out_total, 1) * 1000

            entry = {
                "mode": "ar", "batch_size": batch_size,
                "input_len": input_ids.shape[1], "total_input_tokens": total_tokens,
                "peak_allocated_mb": peak["peak_allocated_mb"],
                "peak_reserved_mb": peak["peak_reserved_mb"],
                "wall_clock_s": round(elapsed, 3),
                "tpot_ms": round(tpot, 2),
                "num_output_tokens": n_out_total,
                "oom": False,
            }
            results.append(entry)
            print(f"    alloc={peak['peak_allocated_mb']:.0f} "
                  f"resv={peak['peak_reserved_mb']:.0f} "
                  f"elapsed={elapsed:.1f}s tpot={tpot:.1f}ms out={n_out_total}")

            del output
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            entry = {"mode": "ar", "batch_size": batch_size, "oom": True}
            results.append(entry)
            print(f"    OOM at batch={batch_size}")
            gc.collect()
            torch.cuda.empty_cache()
            break

    del target
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_dflash_scan(tokenizer, all_prompts):
    from transformers import AutoModelForCausalLM
    from dflash.model import DFlashDraftModel, dflash_generate

    print("\n" + "=" * 60)
    print("  DFlash: batch scan")
    print("=" * 60)

    M0 = mem_now()
    print(f"  [M0] Empty: alloc={M0['allocated_mb']:.0f} resv={M0['reserved_mb']:.0f}")

    target = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, attn_implementation=ATTN_IMPL,
    ).to(DEVICE).eval()

    M1 = mem_now()
    print(f"  [M1] Target: alloc={M1['allocated_mb']:.0f} resv={M1['reserved_mb']:.0f}")

    drafter = DFlashDraftModel.from_pretrained(
        DRAFTER_PATH, dtype=DTYPE, attn_implementation=ATTN_IMPL,
    ).to(DEVICE).eval()

    M2 = mem_now()
    print(f"  [M2] +Drafter: alloc={M2['allocated_mb']:.0f} "
          f"resv={M2['reserved_mb']:.0f} (+{M2['allocated_mb']-M1['allocated_mb']:.0f})")

    results = []
    for batch_size in BATCH_SIZES:
        prompts = all_prompts[:batch_size]
        input_ids = collate_batch(prompts, tokenizer)
        total_tokens = input_ids.shape[0] * input_ids.shape[1]
        print(f"\n  batch={batch_size} ({batch_size}×{input_ids.shape[1]}={total_tokens} tokens)")

        try:
            reset_peak()
            t0 = time.perf_counter()
            with torch.inference_mode():
                stats = dflash_generate(
                    drafter, target=target, input_ids=input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=0.0, return_stats=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak = get_peak_full()

            n_out = stats.num_output_tokens
            tpot = elapsed / max(n_out, 1) * 1000
            accept_lens = stats.acceptance_lengths
            avg_accept = sum(accept_lens) / max(len(accept_lens), 1)

            entry = {
                "mode": "dflash", "batch_size": batch_size,
                "input_len": input_ids.shape[1], "total_input_tokens": total_tokens,
                "peak_allocated_mb": peak["peak_allocated_mb"],
                "peak_reserved_mb": peak["peak_reserved_mb"],
                "wall_clock_s": round(elapsed, 3),
                "tpot_ms": round(tpot, 2),
                "num_output_tokens": n_out,
                "avg_accepted_length": round(avg_accept, 2),
                "acceptance_lengths": accept_lens,
                "block_size": drafter.block_size,
                "oom": False,
            }
            results.append(entry)
            print(f"    alloc={peak['peak_allocated_mb']:.0f} "
                  f"resv={peak['peak_reserved_mb']:.0f} "
                  f"elapsed={elapsed:.1f}s tpot={tpot:.1f}ms "
                  f"acc={avg_accept:.1f}/{drafter.block_size} out={n_out}")

            del stats
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            entry = {"mode": "dflash", "batch_size": batch_size, "oom": True}
            results.append(entry)
            print(f"    OOM at batch={batch_size}")
            gc.collect()
            torch.cuda.empty_cache()
            break

    del target
    del drafter
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    print("Loading data...")
    all_prompts, tokenizer = load_prompts(max(BATCH_SIZES))

    import subprocess
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True).strip()

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": "HumanEval (first 8)",
        "gpu": gpu_name,
        "dtype": str(DTYPE),
        "attn_impl": ATTN_IMPL,
        "batch_sizes": BATCH_SIZES,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    ar_results = run_ar_baseline(tokenizer, all_prompts)
    dflash_results = run_dflash_scan(tokenizer, all_prompts)

    # Summary
    print(f"\n{'=' * 60}")
    print("  BATCH SCAN SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'batch':>5s}  {'AR alloc':>10s} {'AR resv':>10s} {'DF alloc':>10s} {'DF resv':>10s} {'diff':>8s} {'AR tpot':>8s} {'DF tpot':>8s} {'acc':>6s}")
    print("-" * 90)

    for i, bs in enumerate(BATCH_SIZES):
        ar = ar_results[i] if i < len(ar_results) else None
        df = dflash_results[i] if i < len(dflash_results) else None

        ar_str = f"{ar['peak_allocated_mb']:7.0f} {ar['peak_reserved_mb']:7.0f}" if ar and not ar.get("oom") else "    OOM   "
        df_str = f"{df['peak_allocated_mb']:7.0f} {df['peak_reserved_mb']:7.0f}" if df and not df.get("oom") else "    OOM   "
        diff_str = ""
        ar_tp = ""
        df_tp = ""
        acc_str = ""
        if ar and not ar.get("oom") and df and not df.get("oom"):
            diff_str = f"+{df['peak_allocated_mb']-ar['peak_allocated_mb']:.0f} MB"
            ar_tp = f"{ar['tpot_ms']:5.1f}"
            df_tp = f"{df['tpot_ms']:5.1f}"
            acc_str = f"{df['avg_accepted_length']:.1f}/{df['block_size']}"
        print(f"  {bs:3d}   {ar_str}  {df_str}  {diff_str:>8s}  {ar_tp:>8s}  {df_tp:>8s}  {acc_str:>6s}")

    # Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(OUTPUT_DIR) / f"p1b_batch_scan_{ts}.json"
    output = {"metadata": metadata, "ar_results": ar_results, "dflash_results": dflash_results}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
