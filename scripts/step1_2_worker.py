#!/usr/bin/env python3
"""
Step 1.2 worker: batch 压力扫描

Usage:
  python step1_2_worker.py --mode ar --batch 1
  python step1_2_worker.py --mode dflash --batch 8

每个调用 = 独立进程,避免 OOM 污染。
数据: COIG-CQIA bucket=1024 (全部 1024 token,无需 padding)
batch > 样本数时循环复用样本 (greedy, 不影响显存测试)
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
MAX_NEW = 256
TEMPERATURE = 0.0
ATTN = "sdpa"
GPU_MEM_MB = 24564

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
            if len(parts) == 2:
                if int(parts[0].strip()) == pid:
                    return round(float(parts[1].strip()), 1)
        return 0.0
    except Exception:
        return -1.0


def mem_now(label=""):
    torch.cuda.synchronize()
    snap = {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
        "nvidia_smi_mb": nvidia_smi_process_mem(),
    }
    if label:
        print(f"  [{label}] alloc={snap['allocated_mb']:.0f} "
              f"resv={snap['reserved_mb']:.0f} nvsmi={snap['nvidia_smi_mb']:.0f}")
    return snap


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_now(label=""):
    torch.cuda.synchronize()
    snap = {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576, 1),
        "nvidia_smi_mb": nvidia_smi_process_mem(),
    }
    if label:
        print(f"  [{label}] peak_alloc={snap['peak_allocated_mb']:.0f} "
              f"peak_resv={snap['peak_reserved_mb']:.0f} "
              f"nvsmi={snap['nvidia_smi_mb']:.0f}")
    return snap


# ── 数据 ──────────────────────────────────────────────
def load_data(bucket=1024):
    """加载 COIG bucket=1024 的全部样本,用于 batch 构造"""
    with open(DATA_FILE) as f:
        items = [json.loads(line) for line in f]
    candidates = [it for it in items if it.get("bucket") == bucket]
    if not candidates:
        raise ValueError(f"bucket={bucket} 无样本")
    # 提取 token_ids (已预编码)
    return candidates


def build_batch(items, batch_size, tokenizer):
    """用 token_ids 直接构造 batch (全部 1024 token,无需 padding)"""
    n = len(items)
    batch_items = [items[i % n] for i in range(batch_size)]
    # COIG 数据已含 token_ids
    all_ids = [it["token_ids"] for it in batch_items]
    max_len = max(len(ids) for ids in all_ids)
    batch_ids = []
    batch_mask = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for ids in all_ids:
        pad = max_len - len(ids)
        batch_ids.append([pad_id] * pad + ids)
        batch_mask.append([0] * pad + [1] * len(ids))
    return (
        torch.tensor(batch_ids, dtype=torch.long, device="cuda"),
        torch.tensor(batch_mask, dtype=torch.long, device="cuda"),
    )


# ── 主流程 ────────────────────────────────────────────
def main():
    # 解析参数
    mode = "ar"
    batch_size = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg == "--batch" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])

    if mode not in ("ar", "dflash"):
        print("Usage: step1_2_worker.py --mode ar|dflash --batch N")
        sys.exit(1)

    import transformers

    print("=" * 60)
    print(f"  Step 1.2: mode={mode} batch={batch_size}")
    print(f"  {NOW}")
    print("=" * 60)

    items = load_data(1024)
    print(f"  数据: COIG-CQIA bucket=1024, {len(items)} 条可用")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    result = {
        "mode": mode,
        "batch_size": batch_size,
        "timestamp": NOW,
        "config": {
            "input_len": 1024,
            "output_len": MAX_NEW,
            "dtype": str(DTYPE),
            "attn_impl": ATTN,
            "data_source": "COIG-CQIA bucket=1024",
        },
    }

    # ── M0 / M1 / M2 ──
    m0 = mem_now("M0 空进程")

    target = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, attn_implementation=ATTN
    ).to("cuda").eval()
    m1 = mem_now("M1 target加载后")

    drafter = None
    m2 = None
    if mode == "dflash":
        from dflash.model import DFlashDraftModel, dflash_generate
        drafter = DFlashDraftModel.from_pretrained(
            DRAFTER_PATH, dtype=DTYPE, attn_implementation=ATTN
        ).to("cuda").eval()
        m2 = mem_now("M2 drafter加载后")
        print(f"  block_size: {drafter.block_size}")
    else:
        from dflash.model import dflash_generate

    eos_id = tokenizer.eos_token_id
    result["stages"] = {"m0": m0, "m1": m1}
    if m2:
        result["stages"]["m2"] = m2

    # ── 构造 batch ──
    input_ids, attn_mask = build_batch(items, batch_size, tokenizer)
    actual_batch = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    print(f"\n  batch={actual_batch}, seq_len={seq_len}")

    # ── warmup (batch=1 子集) ──
    print(f"  warmup...")
    warmup_ids = input_ids[:1]
    warmup_mask = attn_mask[:1]
    with torch.inference_mode():
        if mode == "ar":
            _ = target.generate(
                warmup_ids, max_new_tokens=8, do_sample=False,
                pad_token_id=eos_id, attention_mask=warmup_mask,
            )
        else:
            _ = dflash_generate(
                drafter, target=target, input_ids=warmup_ids,
                max_new_tokens=8, stop_token_ids=None,
                temperature=TEMPERATURE, return_stats=False,
                ignore_eos=True, attention_mask=warmup_mask,
            )
    del _, warmup_ids, warmup_mask
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  warmup done")

    # ── 正式运行 ──
    reset_peak()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        with torch.inference_mode():
            if mode == "ar":
                out = target.generate(
                    input_ids, max_new_tokens=MAX_NEW,
                    do_sample=False, pad_token_id=eos_id,
                    attention_mask=attn_mask,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                peak = peak_now("AR peak")

                num_out = (out.shape[1] - seq_len) * actual_batch
                tpot = elapsed / max(num_out, 1) * 1000
                throughput = num_out / max(elapsed, 0.001)

                run_result = {
                    "success": True, "oom": False,
                    "peak_allocated_mb": peak["peak_allocated_mb"],
                    "peak_reserved_mb": peak["peak_reserved_mb"],
                    "nvidia_smi_mb": peak["nvidia_smi_mb"],
                    "wall_clock_s": round(elapsed, 3),
                    "num_output_tokens": num_out,
                    "tpot_ms": round(tpot, 2),
                    "throughput_tok_s": round(throughput, 1),
                    "headroom_mb": round(GPU_MEM_MB - peak["peak_reserved_mb"], 1),
                }
                print(f"  AR: {num_out} tokens, {elapsed:.1f}s, "
                      f"tpot={tpot:.1f}ms, throughput={throughput:.1f} tok/s")
                del out

            else:
                stats = dflash_generate(
                    drafter, target=target, input_ids=input_ids,
                    max_new_tokens=MAX_NEW, stop_token_ids=None,
                    temperature=TEMPERATURE, return_stats=True,
                    ignore_eos=True, attention_mask=attn_mask,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                peak = peak_now("DFlash peak")

                num_out = stats.num_output_tokens
                tpot = elapsed / max(num_out, 1) * 1000
                throughput = num_out / max(elapsed, 0.001)

                # acceptance 指标聚合
                import statistics
                all_raw = []
                all_used = []
                all_advance = []
                zero_min_steps = 0

                for acc in stats.acceptance_lengths:
                    raw_list = acc["raw_draft_accept"]
                    used = acc["used_draft_accept"]
                    advance = acc["advance_tokens"]
                    all_raw.extend(raw_list)
                    all_used.append(used)
                    all_advance.append(advance)
                    if used == 0:
                        zero_min_steps += 1

                num_steps = len(all_advance)
                raw_accept_mean = statistics.mean(all_raw) if all_raw else 0
                mean_advance_tokens = statistics.mean(all_advance) if all_advance else 0

                total_validated = sum(all_raw)
                total_utilized = actual_batch * sum(all_used)
                sync_efficiency = (total_utilized / total_validated) if total_validated > 0 else 0.0
                accepted_token_waste = total_validated - total_utilized
                zero_min_rate = zero_min_steps / num_steps if num_steps else 0

                run_result = {
                    "success": True, "oom": False,
                    "peak_allocated_mb": peak["peak_allocated_mb"],
                    "peak_reserved_mb": peak["peak_reserved_mb"],
                    "nvidia_smi_mb": peak["nvidia_smi_mb"],
                    "wall_clock_s": round(elapsed, 3),
                    "num_output_tokens": num_out,
                    "tpot_ms": round(tpot, 2),
                    "throughput_tok_s": round(throughput, 1),
                    "headroom_mb": round(GPU_MEM_MB - peak["peak_reserved_mb"], 1),
                    "raw_accept_mean": round(raw_accept_mean, 2),
                    "mean_advance_tokens": round(mean_advance_tokens, 2),
                    "sync_efficiency": round(sync_efficiency, 4),
                    "accepted_token_waste": accepted_token_waste,
                    "zero_min_rate": round(zero_min_rate, 4),
                    "num_acceptance_steps": num_steps,
                }
                print(f"  DFlash: {num_out} tokens, {elapsed:.1f}s, "
                      f"tpot={tpot:.1f}ms, throughput={throughput:.1f} tok/s")
                print(f"         raw_acc={raw_accept_mean:.1f} advance={mean_advance_tokens:.1f} "
                      f"sync_eff={sync_efficiency:.2f} zero_min_rate={zero_min_rate:.2f}")
                del stats

    except torch.cuda.OutOfMemoryError as e:
        gc.collect()
        torch.cuda.empty_cache()
        peak = peak_now("OOM")
        run_result = {
            "success": False, "oom": True,
            "peak_allocated_mb": peak["peak_allocated_mb"],
            "peak_reserved_mb": peak["peak_reserved_mb"],
            "nvidia_smi_mb": peak["nvidia_smi_mb"],
            "error": "OutOfMemoryError",
            "headroom_mb": 0,
        }
        print(f"  OOM! peak_resv={peak['peak_reserved_mb']:.0f}")

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        run_result = {
            "success": False, "oom": False,
            "error": str(e)[:500],
        }
        print(f"  ERROR: {e}")

    del input_ids, attn_mask
    gc.collect()
    torch.cuda.empty_cache()

    result["run"] = run_result
    result["stages_final"] = mem_now("运行后")

    out_path = os.path.join(
        OUTPUT_DIR, f"step12_{mode}_b{batch_size}_{TODAY}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  输出: {out_path}")
    status = "OK" if run_result.get("success") else ("OOM" if run_result.get("oom") else "ERR")
    print(f"  状态: {status}")


if __name__ == "__main__":
    main()
