#!/usr/bin/env python3
"""
p1b_sweep_orchestrator.py - Phase 1.2 sweep orchestrator

调度流程:
  1. 小→大 batch 扫描 42 个配置 (run_id=1)
  2. 找到每个 mode×variant 的 max_batch
  3. 对 max_batch 和 max_batch//2 补跑 run_id=2,3
  4. 调用 gather_results.py 生成 csv/md/图
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

OUTPUT_DIR = "/root/autodl-tmp/litedrafter/outputs"
SCRIPTS_DIR = "/root/autodl-tmp/litedrafter/scripts"
LOGS_DIR = "/root/autodl-tmp/litedrafter/logs"
WORKER = os.path.join(SCRIPTS_DIR, "p1b_sweep_worker.py")

VARIANTS_AND_BATCHES = [
    ("raw", [1, 2, 4, 8, 16, 32]),
    ("ctx1024", [1, 2, 4, 8, 16, 32]),
    ("ctx2048", [1, 2, 4, 8, 16]),
    ("ctx4096", [1, 2, 4, 8]),
]
MODES = ["ar", "dflash"]

START_TIME = datetime.now()


def run_worker(mode, variant, batch_size, run_id):
    """Run single worker, return parsed JSON result."""
    out_name = f"p1b_{mode}_{variant}_b{batch_size}_r{run_id}.json"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)

    cmd = [
        sys.executable, WORKER, mode, variant, str(batch_size), str(run_id)
    ]
    env = os.environ.copy()
    env["CONDA_DEFAULT_ENV"] = "env_vllm023_pr40898"

    print(f"  [{mode}] {variant} batch={batch_size} run={run_id} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"CRASHED ({elapsed:.0f}s)")
        print(f"    stderr: {proc.stderr[-200:]}")
        return {"success": False, "oom": False, "error": proc.stderr[-500:]}

    with open(out_path) as f:
        result = json.load(f)

    if result.get("oom"):
        print(f"OOM ({elapsed:.0f}s)")
    elif result.get("success"):
        pk = result.get("peak_reserved_mb", 0)
        tp = result.get("throughput_tok_s", 0)
        print(f"OK peak_resv={pk:.0f}MB tput={tp:.0f}tok/s ({elapsed:.0f}s)")
    else:
        print(f"ERROR ({elapsed:.0f}s): {result.get('error', 'unknown')}")

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  Phase 1.2 Sweep  |  {START_TIME.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ---- Phase 1: Sweep (run_id=1) ----
    print("\n--- Phase 1: sweep (run_id=1) ---")
    oom_boundaries = {}

    for mode in MODES:
        for variant, batch_sizes in VARIANTS_AND_BATCHES:
            key = f"{mode}_{variant}"
            print(f"\n  Mode={mode}, Variant={variant}")
            for bs in batch_sizes:
                result = run_worker(mode, variant, bs, 1)
                if result.get("oom"):
                    oom_boundaries[key] = {"max_batch": _prev_batch(batch_sizes, bs),
                                           "oom_at": bs}
                    break
                # If last batch succeeds, max_batch = last
                if bs == batch_sizes[-1]:
                    oom_boundaries[key] = {"max_batch": bs, "oom_at": None}

    # ---- Phase 2: Repeat max_batch and max_batch//2 ----
    print("\n--- Phase 2: repeat max_batch and max_batch//2 ---")
    for key, info in oom_boundaries.items():
        mode, variant = key.split("_", 1)
        max_b = info["max_batch"]
        if max_b is None or max_b < 2:
            continue
        extra_batches = sorted(set([max_b, max_b // 2]))
        for bs in extra_batches:
            print(f"\n  [{mode}] {variant} batch={bs} (repeat)")
            for run_id in [2, 3]:
                run_worker(mode, variant, bs, run_id)

    # ---- Phase 3: Gather ----
    print("\n--- Phase 3: gather results ---")
    gather_script = os.path.join(SCRIPTS_DIR, "gather_results.py")
    if os.path.exists(gather_script):
        subprocess.run([sys.executable, gather_script], check=True)

    elapsed = (datetime.now() - START_TIME).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"  Sweep complete in {elapsed/60:.1f} min")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


def _prev_batch(batch_list, current):
    """Return the batch before current, or None."""
    idx = batch_list.index(current) if current in batch_list else -1
    return batch_list[idx - 1] if idx > 0 else None


if __name__ == "__main__":
    main()
