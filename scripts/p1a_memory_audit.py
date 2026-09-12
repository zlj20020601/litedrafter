#!/usr/bin/env python3
"""
p1a_memory_audit.py - Phase 1 Step 1.1: 单 batch 干净基线

修正:
1. 统一外层计时 (AR 和 DFlash 都用 time.perf_counter + cuda.synchronize)
2. 峰值同时记录 peak_allocated + peak_reserved
3. 每次 run 后显式 del + gc + empty_cache
4. 中文真实数据集 (COIG-CQIA) 替换 dummy text

输出:
  outputs/p1a_memory_audit_<timestamp>.json
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
DATA_PATH = "/root/autodl-tmp/litedrafter/data/coig_cqia_buckets.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/litedrafter/outputs"
DTYPE = torch.bfloat16
WARMUP_RUNS = 1
REPEATS = 3
DEVICE = "cuda"


# ---- Memory utilities ----

def mem_now():
    """Snapshot current allocated + reserved VRAM in MB."""
    torch.cuda.synchronize()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
    }


def reset_peak():
    """Reset peak memory counters (both allocated and reserved)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def get_peak_full():
    """Return peak allocated + peak reserved since last reset."""
    torch.cuda.synchronize()
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576, 1),
    }


def cleanup(*names):
    """Explicit del + gc + empty_cache. Names are local variable names."""
    for name in names:
        # Caller is responsible for del'ing their locals
        pass
    gc.collect()
    torch.cuda.empty_cache()


# ---- Data loading ----

def load_buckets(data_path):
    """Load Chinese real data, grouped by bucket length."""
    import json as _json
    from collections import defaultdict

    buckets = defaultdict(list)
    with open(data_path) as f:
        for line in f:
            item = _json.loads(line)
            buckets[item["bucket"]].append(item)

    result = {}
    for bucket_len in sorted(buckets.keys()):
        result[bucket_len] = buckets[bucket_len]
        print(f"  Loaded bucket {bucket_len}: {len(result[bucket_len])} samples")

    return result


def make_input_tensor(sample):
    """Convert a data sample to input_ids tensor on GPU."""
    ids = sample["token_ids"]
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


# ---- AR baseline ----

def run_ar_baseline(tokenizer, buckets):
    from transformers import AutoModelForCausalLM

    print("\n" + "=" * 60)
    print("PHASE A: AR Baseline (target only)")
    print("=" * 60)

    attn = "sdpa"
    eos_id = tokenizer.eos_token_id

    M0 = mem_now()
    print(f"[M0] Empty: {M0['allocated_mb']:.0f} MB allocated, "
          f"{M0['reserved_mb']:.0f} MB reserved")

    target = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        attn_implementation=attn,
    ).to(DEVICE).eval()

    M1 = mem_now()
    print(f"[M1] Target loaded: {M1['allocated_mb']:.0f} MB allocated "
          f"(+{M1['allocated_mb'] - M0['allocated_mb']:.0f} MB)")

    results = []
    for in_len, samples in buckets.items():
        sample = samples[0]  # Use first sample per bucket
        input_ids = make_input_tensor(sample)
        print(f"\n  [AR] input_len={in_len} (token_ids from COIG-CQIA)")

        # Warmup
        for _ in range(WARMUP_RUNS):
            with torch.inference_mode():
                tmp = target.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    eos_token_id=eos_id,
                )
            del tmp
            gc.collect()
            torch.cuda.empty_cache()

        # Formal runs
        for rep in range(1, REPEATS + 1):
            reset_peak()
            t0 = time.perf_counter()
            with torch.inference_mode():
                output = target.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    eos_token_id=eos_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak = get_peak_full()
            n_out = output.shape[1] - input_ids.shape[1]
            tpot = elapsed / max(n_out, 1) * 1000

            entry = {
                "mode": "ar",
                "input_len": in_len,
                "repeat": rep,
                "peak_allocated_mb": peak["peak_allocated_mb"],
                "peak_reserved_mb": peak["peak_reserved_mb"],
                "wall_clock_s": round(elapsed, 3),
                "tpot_ms": round(tpot, 2),
                "num_output_tokens": n_out,
            }
            results.append(entry)
            print(f"    rep {rep}: peak_alloc={peak['peak_allocated_mb']:.0f} MB, "
                  f"peak_resv={peak['peak_reserved_mb']:.0f} MB, "
                  f"tpot={tpot:.1f} ms, out_tokens={n_out}")
            del output
            gc.collect()
            torch.cuda.empty_cache()

    del target
    gc.collect()
    torch.cuda.empty_cache()

    return {"stages": {"m0_empty": M0, "m1_target_loaded": M1}, "runs": results}


# ---- DFlash ----

def run_dflash(tokenizer, buckets):
    from transformers import AutoModelForCausalLM
    from dflash.model import DFlashDraftModel, dflash_generate

    print("\n" + "=" * 60)
    print("PHASE B: DFlash (target + drafter)")
    print("=" * 60)

    attn = "sdpa"
    eos_id = tokenizer.eos_token_id

    M0 = mem_now()
    print(f"[M0] Empty: {M0['allocated_mb']:.0f} MB allocated, "
          f"{M0['reserved_mb']:.0f} MB reserved")

    target = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        attn_implementation=attn,
    ).to(DEVICE).eval()

    M1 = mem_now()
    print(f"[M1] Target loaded: {M1['allocated_mb']:.0f} MB allocated "
          f"(+{M1['allocated_mb'] - M0['allocated_mb']:.0f} MB)")

    drafter = DFlashDraftModel.from_pretrained(
        DRAFTER_PATH,
        dtype=DTYPE,
        attn_implementation=attn,
    ).to(DEVICE).eval()

    M2 = mem_now()
    drafter_overhead = M2['allocated_mb'] - M1['allocated_mb']
    print(f"[M2] Drafter loaded: {M2['allocated_mb']:.0f} MB allocated "
          f"(+{drafter_overhead:.0f} MB)")

    block_size = drafter.block_size
    print(f"[INFO] DFlash block_size={block_size}")

    results = []
    for in_len, samples in buckets.items():
        sample = samples[0]  # Use first sample per bucket
        input_ids = make_input_tensor(sample)
        print(f"\n  [DFlash] input_len={in_len} (token_ids from COIG-CQIA)")

        # Warmup
        for _ in range(WARMUP_RUNS):
            with torch.inference_mode():
                tmp = dflash_generate(
                    drafter,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=256,
                    stop_token_ids=[eos_id],
                    temperature=0.0,
                    return_stats=True,
                )
            del tmp
            gc.collect()
            torch.cuda.empty_cache()

        # Formal runs
        for rep in range(1, REPEATS + 1):
            reset_peak()
            t0 = time.perf_counter()
            with torch.inference_mode():
                stats = dflash_generate(
                    drafter,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=256,
                    stop_token_ids=[eos_id],
                    temperature=0.0,
                    return_stats=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak = get_peak_full()
            n_out = stats.num_output_tokens
            tpot = elapsed / max(n_out, 1) * 1000
            avg_accept = sum(stats.acceptance_lengths) / max(len(stats.acceptance_lengths), 1)

            entry = {
                "mode": "dflash",
                "input_len": in_len,
                "repeat": rep,
                "peak_allocated_mb": peak["peak_allocated_mb"],
                "peak_reserved_mb": peak["peak_reserved_mb"],
                "wall_clock_s": round(elapsed, 3),
                "tpot_ms": round(tpot, 2),
                "num_output_tokens": n_out,
                "block_size": block_size,
                "avg_accepted_length": round(avg_accept, 2),
                "acceptance_lengths": stats.acceptance_lengths,
            }
            results.append(entry)
            print(f"    rep {rep}: peak_alloc={peak['peak_allocated_mb']:.0f} MB, "
                  f"peak_resv={peak['peak_reserved_mb']:.0f} MB, "
                  f"tpot={tpot:.1f} ms, avg_accept={avg_accept:.1f}/{block_size}, "
                  f"out_tokens={n_out}")
            del stats
            gc.collect()
            torch.cuda.empty_cache()

    del target
    del drafter
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "stages": {"m0_empty": M0, "m1_target_loaded": M1, "m2_drafter_loaded": M2},
        "runs": results,
    }


# ---- Main ----

def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.isdir(DRAFTER_PATH):
        print(f"[ERROR] Drafter not found: {DRAFTER_PATH}")
        sys.exit(1)
    if not os.path.isfile(DATA_PATH):
        print(f"[ERROR] Data not found: {DATA_PATH}")
        print("Run scripts/prepare_data.py first.")
        sys.exit(1)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    import subprocess
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "Qwen3-8B",
        "drafter": "Qwen3-8B-DFlash-b16",
        "gpu": gpu_name,
        "dtype": str(DTYPE),
        "attn_impl": "sdpa",
        "data_source": "COIG-CQIA",
        "warmup_runs": WARMUP_RUNS,
        "repeats": REPEATS,
        "output_length": 256,
    }

    print("Loading data...")
    buckets = load_buckets(DATA_PATH)

    ar_results = run_ar_baseline(tokenizer, buckets)
    dflash_results = run_dflash(tokenizer, buckets)

    output = {
        "metadata": metadata,
        "ar_baseline": ar_results,
        "dflash": dflash_results,
    }

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(OUTPUT_DIR) / f"p1a_memory_audit_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"[SAVED] {out_path}")
    print(f"{'=' * 60}")

    # Quick summary
    print("\n--- SUMMARY ---")
    m1 = ar_results["stages"]["m1_target_loaded"]["allocated_mb"]
    m2 = dflash_results["stages"]["m2_drafter_loaded"]["allocated_mb"]
    print(f"Target weights (M1):      {m1:.0f} MB")
    print(f"Drafter overhead (M2-M1): {m2 - m1:.0f} MB")

    for in_len in buckets.keys():
        ar_runs = [r for r in ar_results["runs"] if r["input_len"] == in_len]
        df_runs = [r for r in dflash_results["runs"] if r["input_len"] == in_len]
        if not ar_runs or not df_runs:
            continue
        ar_pa = sum(r["peak_allocated_mb"] for r in ar_runs) / len(ar_runs)
        ar_pr = sum(r["peak_reserved_mb"] for r in ar_runs) / len(ar_pr_runs) if (ar_pr_runs := [r for r in ar_runs]) else 0
        df_pa = sum(r["peak_allocated_mb"] for r in df_runs) / len(df_runs)
        df_pr = sum(r["peak_reserved_mb"] for r in df_runs) / len(df_runs)
        ar_tp = sum(r["tpot_ms"] for r in ar_runs) / len(ar_runs)
        df_tp = sum(r["tpot_ms"] for r in df_runs) / len(df_runs)
        avg_acc = sum(r["avg_accepted_length"] for r in df_runs) / len(df_runs)
        print(f"  input={in_len:4d}  "
              f"AR: alloc={ar_pa:.0f} resv={ar_pr:.0f} tpot={ar_tp:.1f}  "
              f"DFlash: alloc={df_pa:.0f} resv={df_pr:.0f} tpot={df_tp:.1f} "
              f"acc={avg_acc:.1f}/{df_runs[0]['block_size']}  "
              f"speedup={ar_tp/df_tp:.1f}x  "
              f"diff={df_pa-ar_pa:+.0f} MB")


if __name__ == "__main__":
    main()
