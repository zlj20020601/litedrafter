#!/usr/bin/env python3
"""
Step 1.1 worker: 单 batch 干净基线

Usage:
  python step1_1_worker.py --mode ar
  python step1_1_worker.py --mode dflash

AR 和 DFlash 分进程执行,避免显存污染。
每个进程内跑 input_len=512/1024/2048 × 3 repeats。

输出: outputs/step11_{mode}_batch1_{date}.json
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
PROJECT_DIR = "/root/autodl-tmp/litedrafter"
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

DTYPE = torch.bfloat16
INPUT_LENS = [512, 1024, 2048]
OUTPUT_LEN = 256
REPEATS = 3
WARMUP_RUNS = 1

TODAY = datetime.now().strftime("%Y%m%d")
NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ── 显存工具 ──────────────────────────────────────────
def nvidia_smi_process_mem():
    """读取当前进程在 GPU 上的显存占用 (MiB)"""
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
                p = int(parts[0].strip())
                mem = float(parts[1].strip())
                if p == pid:
                    return round(mem, 1)
        return 0.0
    except Exception:
        return -1.0


def mem_now(label=""):
    """三种口径同时记录"""
    torch.cuda.synchronize()
    snap = {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
        "nvidia_smi_mb": nvidia_smi_process_mem(),
    }
    if label:
        print(f"    [{label}] alloc={snap['allocated_mb']:.0f} "
              f"resv={snap['reserved_mb']:.0f} "
              f"nvsmi={snap['nvidia_smi_mb']:.0f}")
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
        print(f"    [{label}] peak_alloc={snap['peak_allocated_mb']:.0f} "
              f"peak_resv={snap['peak_reserved_mb']:.0f} "
              f"nvsmi={snap['nvidia_smi_mb']:.0f}")
    return snap


# ── 数据准备 ──────────────────────────────────────────
def load_data():
    """按 bucket 加载,每个 bucket 取第 1 条作为固定样本"""
    with open(DATA_FILE) as f:
        items = [json.loads(line) for line in f]
    samples = {}
    for b in INPUT_LENS:
        candidates = [it for it in items if it.get("bucket") == b]
        if not candidates:
            raise ValueError(f"bucket={b} 无样本")
        samples[b] = candidates[0]["text"]
    return samples


# ── 主流程 ────────────────────────────────────────────
def main():
    mode = None
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
    if mode not in ("ar", "dflash"):
        print("Usage: python step1_1_worker.py --mode ar|dflash")
        sys.exit(1)

    import transformers

    print("=" * 60)
    print(f"  Step 1.1 worker: mode={mode}")
    print(f"  {NOW}")
    print("=" * 60)

    samples = load_data()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

    result = {
        "mode": mode,
        "timestamp": NOW,
        "config": {
            "input_lens": INPUT_LENS,
            "output_len": OUTPUT_LEN,
            "repeats": REPEATS,
            "warmup_runs": WARMUP_RUNS,
            "dtype": str(DTYPE),
            "attn_impl": "sdpa",
            "data_source": "COIG-CQIA",
        },
    }

    # ── M0: 空进程 ──
    m0 = mem_now("M0 空CUDA进程")

    # ── 加载 target ──
    print("\n  加载 Qwen3-8B target...")
    target = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, attn_implementation="sdpa"
    ).to("cuda").eval()
    m1 = mem_now("M1 target加载后")

    # ── 加载 drafter (仅 dflash) ──
    drafter = None
    m2 = None
    if mode == "dflash":
        from dflash.model import DFlashDraftModel, dflash_generate
        print("\n  加载 DFlash drafter...")
        drafter = DFlashDraftModel.from_pretrained(
            DRAFTER_PATH, dtype=DTYPE, attn_implementation="sdpa"
        ).to("cuda").eval()
        m2 = mem_now("M2 drafter加载后")
        print(f"  block_size: {drafter.block_size}")
    else:
        from dflash.model import dflash_generate  # 确保 import 链可用

    eos_id = tokenizer.eos_token_id

    result["stages"] = {"m0": m0, "m1": m1}
    if m2:
        result["stages"]["m2"] = m2

    # ── 逐 input_len 跑 ──
    runs = []

    for input_len in INPUT_LENS:
        text = samples[input_len]
        input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
        actual_len = input_ids.shape[1]
        print(f"\n  --- input_len={input_len} (actual={actual_len} tokens) ---")

        # warmup
        for w in range(WARMUP_RUNS):
            print(f"    warmup {w+1}/{WARMUP_RUNS}...", end=" ", flush=True)
            with torch.inference_mode():
                if mode == "ar":
                    _ = target.generate(
                        input_ids, max_new_tokens=OUTPUT_LEN,
                        do_sample=False, pad_token_id=eos_id,
                    )
                else:
                    _ = dflash_generate(
                        drafter, target=target, input_ids=input_ids,
                        max_new_tokens=OUTPUT_LEN,
                        stop_token_ids=[eos_id], temperature=0.0,
                        return_stats=True,
                    )
            # 清理 warmup 输出
            del _
            gc.collect()
            torch.cuda.empty_cache()
            print("done")

        # 正式 repeats
        for rep in range(1, REPEATS + 1):
            reset_peak()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.inference_mode():
                if mode == "ar":
                    out = target.generate(
                        input_ids, max_new_tokens=OUTPUT_LEN,
                        do_sample=False, pad_token_id=eos_id,
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    new_tokens = out[0][actual_len:]
                    num_out = new_tokens.shape[0]
                    peak = peak_now(f"rep{rep}")
                    run = {
                        "mode": "ar",
                        "input_len": input_len,
                        "actual_input_tokens": actual_len,
                        "repeat": rep,
                        "peak_allocated_mb": peak["peak_allocated_mb"],
                        "peak_reserved_mb": peak["peak_reserved_mb"],
                        "nvidia_smi_mb": peak["nvidia_smi_mb"],
                        "wall_clock_s": round(elapsed, 3),
                        "tpot_ms": round(elapsed / num_out * 1000, 2),
                        "num_output_tokens": num_out,
                    }
                    del out
                else:
                    stats = dflash_generate(
                        drafter, target=target, input_ids=input_ids,
                        max_new_tokens=OUTPUT_LEN,
                        stop_token_ids=[eos_id], temperature=0.0,
                        return_stats=True,
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    df_tokens = stats.output_ids[0][actual_len:]
                    num_out = df_tokens.shape[0]

                    # 解析 acceptance (batch=1: used == raw)
                    all_raw = []
                    all_advance = []
                    for acc in stats.acceptance_lengths:
                        if isinstance(acc, dict):
                            all_raw.extend(acc.get("raw_draft_accept", [acc.get("used_draft_accept", 0)]))
                            all_advance.append(acc.get("advance_tokens", acc.get("used_draft_accept", 0) + 1))
                        else:
                            all_advance.append(acc)

                    raw_accept_mean = sum(all_raw) / len(all_raw) if all_raw else 0
                    mean_advance_tokens = sum(all_advance) / len(all_advance) if all_advance else 0
                    peak = peak_now(f"rep{rep}")

                    run = {
                        "mode": "dflash",
                        "input_len": input_len,
                        "actual_input_tokens": actual_len,
                        "repeat": rep,
                        "peak_allocated_mb": peak["peak_allocated_mb"],
                        "peak_reserved_mb": peak["peak_reserved_mb"],
                        "nvidia_smi_mb": peak["nvidia_smi_mb"],
                        "wall_clock_s": round(elapsed, 3),
                        "tpot_ms": round(elapsed / num_out * 1000, 2),
                        "num_output_tokens": num_out,
                        "block_size": drafter.block_size,
                        "raw_accept_mean": round(raw_accept_mean, 2),
                        "mean_advance_tokens": round(mean_advance_tokens, 2),
                        "acceptance_steps": len(all_advance),
                    }
                    del stats

            gc.collect()
            torch.cuda.empty_cache()

            tpot_str = f"tpot={run['tpot_ms']:.1f}ms"
            if mode == "dflash":
                tpot_str += f" acc={avg_acc:.1f}"
            print(f"    rep{rep}: peak_alloc={peak['peak_allocated_mb']:.0f} "
                  f"peak_resv={peak['peak_reserved_mb']:.0f} "
                  f"nvsmi={peak['nvidia_smi_mb']:.0f} "
                  f"{tpot_str} out={num_out}")

            runs.append(run)

    result["runs"] = runs

    # ── 写文件 ──
    out_path = os.path.join(OUTPUTS_DIR, f"step11_{mode}_batch1_{TODAY}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  输出: {out_path}")


if __name__ == "__main__":
    main()
