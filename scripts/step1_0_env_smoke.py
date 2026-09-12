#!/usr/bin/env python3
"""
Step 1.0: 环境冻结与 Smoke Test

产出：
  env/env_freeze_YYYYMMDD.txt        pip freeze
  env/system_info_YYYYMMDD.txt       OS / CUDA / driver / GPU / 关键包版本
  logs/phase1_step10_smoke_YYYYMMDD.log
  outputs/step10_smoke_result.json   结构化结果

验收标准：
  - torch / transformers / vLLM / dflash import 无报错
  - bitsandbytes / flash_attn 缺失记为环境限制，不阻塞
  - target 和 drafter 路径存在，config 可读取
  - AR 和 DFlash 都能输出至少 5 个 token
  - 显存记录包含 allocated、reserved、nvidia-smi 三种口径
  - Smoke prompt 来自真实中文数据（COIG-CQIA）
"""

import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

# ── 路径 ──────────────────────────────────────────────
PROJECT_DIR = "/root/autodl-tmp/litedrafter"
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DRAFTER_PATH = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
DATA_FILE = os.path.join(PROJECT_DIR, "data", "coig_cqia_buckets.jsonl")

ENV_DIR = os.path.join(PROJECT_DIR, "env")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

TODAY = datetime.now().strftime("%Y%m%d")
NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

DTYPE_STR = "torch.bfloat16"

import torch


# ── 显存工具 ──────────────────────────────────────────
def mem_snapshot(label=""):
    """三种口径同时记录"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1048576
    reserved = torch.cuda.memory_reserved() / 1048576
    nvidia = nvidia_smi_process_mem()
    snap = {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "nvidia_smi_mb": nvidia,
    }
    if label:
        print(f"  [{label}] alloc={snap['allocated_mb']:.0f} "
              f"resv={snap['reserved_mb']:.0f} "
              f"nvidia-smi={snap['nvidia_smi_mb']:.0f} MB")
    return snap


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
    except Exception as e:
        print(f"  [WARN] nvidia-smi 读取失败: {e}")
        return -1.0


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_snapshot(label=""):
    """读取 reset 以来的峰值"""
    torch.cuda.synchronize()
    snap = {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576, 1),
        "nvidia_smi_mb": nvidia_smi_process_mem(),
    }
    if label:
        print(f"  [{label}] peak_alloc={snap['peak_allocated_mb']:.0f} "
              f"peak_resv={snap['peak_reserved_mb']:.0f} "
              f"nvidia-smi={snap['nvidia_smi_mb']:.0f} MB")
    return snap


# ── 环境冻结 ──────────────────────────────────────────
def freeze_env():
    os.makedirs(ENV_DIR, exist_ok=True)

    # system_info
    import transformers
    try:
        import vllm as _vllm
        vllm_ver = _vllm.__version__
    except Exception:
        vllm_ver = "NOT FOUND"

    try:
        import bitsandbytes as _bnb
        bnb_ver = _bnb.__version__
    except Exception:
        bnb_ver = "NOT FOUND (环境限制)"

    try:
        import flash_attn as _fa
        fa_ver = _fa.__version__
    except Exception:
        fa_ver = "NOT FOUND (环境限制)"

    try:
        from dflash.model import DFlashDraftModel, dflash_generate
        dflash_status = "OK"
    except Exception as e:
        dflash_status = f"IMPORT FAILED: {e}"

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1048576

    sys_lines = [
        f"=== System Info ({NOW}) ===\n",
        f"OS:            {platform.platform()}\n",
        f"Python:        {sys.version.split()[0]}\n",
        f"PyTorch:       {torch.__version__}\n",
        f"CUDA (torch):  {torch.version.cuda}\n",
        f"GPU:           {gpu_name}\n",
        f"GPU Memory:    {gpu_total:.0f} MiB total\n",
        f"Transformers:  {transformers.__version__}\n",
        f"vLLM:          {vllm_ver}\n",
        f"bitsandbytes:  {bnb_ver}\n",
        f"flash_attn:    {fa_ver}\n",
        f"dflash:        {dflash_status}\n",
        f"Model path:    {MODEL_PATH} ({'EXISTS' if os.path.isdir(MODEL_PATH) else 'MISSING'})\n",
        f"Drafter path:  {DRAFTER_PATH} ({'EXISTS' if os.path.isdir(DRAFTER_PATH) else 'MISSING'})\n",
    ]
    sys_file = os.path.join(ENV_DIR, f"system_info_{TODAY}.txt")
    with open(sys_file, "w") as f:
        f.writelines(sys_lines)

    # pip freeze
    pip_file = os.path.join(ENV_DIR, f"env_freeze_{TODAY}.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, timeout=60
    )
    with open(pip_file, "w") as f:
        f.write(f"=== pip freeze ({NOW}) ===\n\n")
        f.write(result.stdout)

    print(f"  环境冻结完成: {sys_file}")
    print(f"  pip freeze:   {pip_file}")

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_torch": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_total_mb": round(gpu_total, 0),
        "transformers": transformers.__version__,
        "vllm": vllm_ver,
        "bitsandbytes": bnb_ver,
        "flash_attn": fa_ver,
        "dflash": dflash_status,
    }


# ── 主流程 ────────────────────────────────────────────
def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Step 1.0: 环境冻结与 Smoke Test")
    print(f"  {NOW}")
    print("=" * 60)

    result = {"timestamp": NOW, "project": "LiteDrafter Phase 1"}

    # ── 1. 环境冻结 ──
    print("\n--- 1. 环境冻结 ---")
    env_info = freeze_env()
    result["env"] = env_info

    # 检查 dflash 可导入
    from dflash.model import DFlashDraftModel, dflash_generate
    print(f"  dflash import: OK")

    # ── 2. 真实中文数据 ──
    print("\n--- 2. 准备真实中文 Smoke Prompt ---")
    with open(DATA_FILE) as f:
        items = [json.loads(line) for line in f]
    # 取 bucket=512 的第一条
    candidates = [it for it in items if it.get("bucket") == 512]
    if not candidates:
        candidates = items
    smoke_item = candidates[0]
    smoke_text = smoke_item["text"][:200]  # 截取前 200 字符做 smoke prompt
    print(f"  数据源: COIG-CQIA bucket={smoke_item['bucket']}")
    print(f"  原始 token_len: {smoke_item['token_len']}")
    print(f"  Smoke prompt (前80字): {smoke_text[:80]}...")

    # ── 3. 加载 target + tokenizer ──
    print("\n--- 3. 加载 Qwen3-8B ---")
    import transformers

    DTYPE = torch.bfloat16

    m0 = mem_snapshot("M0 空CUDA进程")

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"  Tokenizer loaded")

    target = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    m1 = mem_snapshot("M1 target加载后")

    # ── 4. AR generate ──
    print("\n--- 4. AR Generate ---")
    input_ids = tokenizer.encode(smoke_text, return_tensors="pt").to("cuda")
    print(f"  Input tokens: {input_ids.shape[1]}")

    reset_peak()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        ar_output = target.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    ar_elapsed = time.perf_counter() - t0

    ar_peak = peak_snapshot("AR peak")
    ar_new_tokens = ar_output[0][input_ids.shape[1]:]
    ar_decoded = tokenizer.decode(ar_new_tokens, skip_special_tokens=True)
    ar_num_tokens = ar_new_tokens.shape[0]
    print(f"  Output ({ar_num_tokens} tokens): {ar_decoded[:60]}...")
    print(f"  Wall clock: {ar_elapsed:.3f}s")

    ar_result = {
        "input_tokens": input_ids.shape[1],
        "output_tokens": ar_num_tokens,
        "output_text": ar_decoded[:200],
        "wall_clock_s": round(ar_elapsed, 3),
        "peak": ar_peak,
    }

    # ── 5. 加载 DFlash drafter ──
    print("\n--- 5. 加载 DFlash Drafter ---")
    drafter = DFlashDraftModel.from_pretrained(
        DRAFTER_PATH,
        dtype=DTYPE,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    m2 = mem_snapshot("M2 drafter加载后")
    print(f"  block_size: {drafter.block_size}")

    # ── 6. dflash_generate ──
    print("\n--- 6. dflash_generate ---")
    eos_id = tokenizer.eos_token_id

    reset_peak()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        stats = dflash_generate(
            drafter,
            target=target,
            input_ids=input_ids,
            max_new_tokens=16,
            stop_token_ids=[eos_id],
            temperature=0.0,
            return_stats=True,
        )
    torch.cuda.synchronize()
    df_elapsed = time.perf_counter() - t0

    df_peak = peak_snapshot("DFlash peak")
    df_output_ids = stats.output_ids[0][input_ids.shape[1]:]
    df_decoded = tokenizer.decode(df_output_ids, skip_special_tokens=True)
    df_num_tokens = df_output_ids.shape[0]
    print(f"  Output ({df_num_tokens} tokens): {df_decoded[:60]}...")
    print(f"  Acceptance lengths: {stats.acceptance_lengths}")
    print(f"  Wall clock: {df_elapsed:.3f}s")

    dflash_result = {
        "input_tokens": input_ids.shape[1],
        "output_tokens": df_num_tokens,
        "output_text": df_decoded[:200],
        "wall_clock_s": round(df_elapsed, 3),
        "acceptance_lengths": stats.acceptance_lengths,
        "peak": df_peak,
    }

    # ── 7. 汇总 ──
    print("\n" + "=" * 60)
    print("  SMOKE TEST SUMMARY")
    print("=" * 60)

    print(f"  M0 (empty):          alloc={m0['allocated_mb']:.0f}  "
          f"resv={m0['reserved_mb']:.0f}  nvidia-smi={m0['nvidia_smi_mb']:.0f}")
    print(f"  M1 (target):         alloc={m1['allocated_mb']:.0f}  "
          f"resv={m1['reserved_mb']:.0f}  nvidia-smi={m1['nvidia_smi_mb']:.0f}  "
          f"(+{m1['allocated_mb'] - m0['allocated_mb']:.0f})")
    print(f"  M2 (drafter):        alloc={m2['allocated_mb']:.0f}  "
          f"resv={m2['reserved_mb']:.0f}  nvidia-smi={m2['nvidia_smi_mb']:.0f}  "
          f"(+{m2['allocated_mb'] - m1['allocated_mb']:.0f})")
    print(f"  AR peak:             alloc={ar_peak['peak_allocated_mb']:.0f}  "
          f"resv={ar_peak['peak_reserved_mb']:.0f}  nvidia-smi={ar_peak['nvidia_smi_mb']:.0f}")
    print(f"  DFlash peak:         alloc={df_peak['peak_allocated_mb']:.0f}  "
          f"resv={df_peak['peak_reserved_mb']:.0f}  nvidia-smi={df_peak['nvidia_smi_mb']:.0f}")

    passed = (
        ar_num_tokens >= 5
        and df_num_tokens >= 5
        and m1["allocated_mb"] > 10000
        and m2["allocated_mb"] > m1["allocated_mb"]
    )
    print(f"\n  Result: {'PASSED' if passed else 'FAILED'}")

    result["stages"] = {"m0": m0, "m1": m1, "m2": m2}
    result["ar"] = ar_result
    result["dflash"] = dflash_result
    result["smoke_prompt"] = {
        "source": "COIG-CQIA",
        "bucket": smoke_item["bucket"],
        "text_preview": smoke_text[:80],
    }
    result["passed"] = passed

    # 写 JSON
    out_path = os.path.join(OUTPUTS_DIR, f"step10_smoke_result_{TODAY}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON: {out_path}")


if __name__ == "__main__":
    main()
