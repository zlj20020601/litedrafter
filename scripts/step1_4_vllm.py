#!/usr/bin/env python3
"""
Step 1.4: vLLM 部署观察项

单卡启动 vLLM (AR 和 DFlash),从启动日志提取 KV cache tokens。
核心指标: KV cache tokens (不是 nvidia-smi 峰值,因为 vLLM 会吃满到 gpu-memory-utilization 上限)
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

TARGET_MODEL = "/root/autodl-tmp/models/Qwen3-8B"
DRAFT_MODEL = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
PYTHON = "/root/autodl-tmp/conda_envs/env_dcut/bin/python"
OUTPUT_DIR = "/root/autodl-tmp/litedrafter/outputs"
LOGS_DIR = "/root/autodl-tmp/litedrafter/logs"

TODAY = datetime.now().strftime("%Y%m%d")
NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

SERVE_COMMON = [
    "--trust-remote-code",
    "--dtype", "bfloat16",
    "--max-model-len", "4096",  # 降低以支持 DFlash 单卡启动
    "--gpu-memory-utilization", "0.90",
    "--max-num-seqs", "32",
    "--max-num-batched-tokens", "16384",
    "--no-enable-prefix-caching",
    "--tensor-parallel-size", "1",   # 单卡
    "--served-model-name", "qwen3-8b",
    "--enforce-eager",  # 绕过 CUDA graph 编译,只需观察 KV cache tokens
]


def start_vllm(mode, port=8199):
    """启动 vLLM server,返回 (process, log_file_path)"""
    log_file = os.path.join(LOGS_DIR, f"step14_{mode}_server_{TODAY}.log")

    cmd = [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
           "--model", TARGET_MODEL,
           "--port", str(port)] + SERVE_COMMON[:]

    if mode == "dflash":
        spec_config = json.dumps({
            "method": "dflash",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 15,
            "dflash_dcut": 0,
        })
        cmd += ["--speculative-config", spec_config]

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    # 确保 conda env bin 在 PATH 前面 (ninja 等工具)
    conda_bin = os.path.dirname(PYTHON)
    env["PATH"] = conda_bin + ":" + env.get("PATH", "")

    print(f"  启动 vLLM [{mode}] (TP=1, 单卡)...")
    print(f"  命令: {' '.join(cmd[:8])}...")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
        )

    return proc, log_file


def wait_for_server(log_file, timeout=180):
    """等待 server 就绪或超时,返回启动日志内容"""
    print(f"  等待 server 就绪 (timeout={timeout}s)...")

    keywords_ready = ["Application startup complete", "Uvicorn running"]
    keywords_error = ["Traceback", "Error", "FAILED", "CUDA out of memory"]
    keywords_kv = ["GPU KV cache size"]

    start = time.time()
    found_kv = False
    found_ready = False
    error_msg = None

    while time.time() - start < timeout:
        time.sleep(3)
        elapsed = int(time.time() - start)

        if not os.path.exists(log_file):
            continue

        with open(log_file, errors="replace") as f:
            content = f.read()

        # 检查 KV cache 信息
        if not found_kv and "GPU KV cache size" in content:
            found_kv = True
            print(f"  [{elapsed}s] 发现 KV cache 信息 ✓")

        # 检查就绪
        if not found_ready:
            for kw in keywords_ready:
                if kw in content:
                    found_ready = True
                    print(f"  [{elapsed}s] Server 就绪 ✓")
                    break

        # 检查错误
        for kw in keywords_error:
            if kw in content:
                error_msg = kw
                print(f"  [{elapsed}s] 发现错误: {kw}")
                break

        if found_kv and found_ready:
            # 再等 2 秒确保日志写完
            time.sleep(2)
            break

        if error_msg:
            break

        if elapsed % 15 == 0 and elapsed > 0:
            print(f"  [{elapsed}s] 等待中...")

    with open(log_file, errors="replace") as f:
        content = f.read()
    return content, found_kv, found_ready, error_msg


def extract_kv_info(log_content):
    """从日志提取 KV cache 相关信息"""
    info = {}

    # GPU KV cache size: X tokens
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log_content)
    if m:
        info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    else:
        info["kv_cache_tokens"] = None

    # Maximum concurrency for X tokens per request: Yx
    m = re.search(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", log_content)
    if m:
        info["max_concurrency"] = float(m.group(1))
    else:
        info["max_concurrency"] = None

    # Available KV cache memory
    m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", log_content)
    if m:
        info["kv_cache_memory_gib"] = float(m.group(1))
    else:
        info["kv_cache_memory_gib"] = None

    # CUDA Graph info
    cg = re.search(r"CUDA Graph.*?(\w+)", log_content)
    if cg:
        info["cuda_graph_mode"] = cg.group(1)
    else:
        m = re.search(r"cudagraph_mode.*?(\w+)", log_content, re.IGNORECASE)
        info["cuda_graph_mode"] = m.group(1) if m else "unknown"

    # Model weights memory
    m = re.search(r"Model weights:\s*([\d.]+)\s*GiB", log_content)
    if m:
        info["model_weights_gib"] = float(m.group(1))
    else:
        info["model_weights_gib"] = None

    return info


def kill_server(proc):
    """杀掉 vLLM server"""
    print(f"  关闭 vLLM server (pid={proc.pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    # 额外清理
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], timeout=5)
    time.sleep(3)
    print(f"  已关闭")


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  Step 1.4: vLLM 部署观察项")
    print(f"  {NOW}")
    print(f"  单卡 TP=1, gpu-memory-utilization=0.90")
    print("=" * 60)

    results = {"timestamp": NOW, "config": {
        "tp": 1,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 8192,
        "max_num_seqs": 32,
    }}

    for mode in ["ar", "dflash"]:
        print(f"\n{'─' * 50}")
        print(f"  [{mode.upper()}] 启动 vLLM")
        print(f"{'─' * 50}")

        proc, log_file = start_vllm(mode)
        log_content, found_kv, found_ready, error_msg = wait_for_server(log_file, timeout=180)

        kv_info = extract_kv_info(log_content)
        kv_info["server_ready"] = found_ready
        kv_info["log_file"] = log_file

        if error_msg:
            kv_info["error"] = error_msg
            # 提取 traceback 附近内容
            for line in log_content.split("\n"):
                if error_msg in line:
                    kv_info["error_detail"] = line.strip()[:200]
                    break

        results[mode] = kv_info

        # 打印结果
        print(f"\n  [{mode.upper()}] 结果:")
        for k, v in kv_info.items():
            if v is not None and k not in ("log_file",):
                print(f"    {k}: {v}")

        if proc.poll() is None:
            kill_server(proc)
        else:
            print(f"  Server 已退出 (code={proc.returncode})")
            subprocess.run(["pkill", "-f", "vllm.entrypoints"], timeout=5)
            time.sleep(3)

    # ── 对比 ──
    print("\n" + "=" * 60)
    print("  KV cache tokens 对比")
    print("=" * 60)

    ar_kv = results.get("ar", {}).get("kv_cache_tokens")
    df_kv = results.get("dflash", {}).get("kv_cache_tokens")

    if ar_kv and df_kv:
        diff = ar_kv - df_kv
        pct = diff / ar_kv * 100
        print(f"  AR:      {ar_kv:>10,} tokens")
        print(f"  DFlash:  {df_kv:>10,} tokens")
        print(f"  差异:    {diff:>+10,} tokens ({pct:.1f}%)")
        print(f"  → DFlash 的 drafter 模型压缩了 {pct:.1f}% 的 KV cache 可用空间")
    elif ar_kv and not df_kv:
        print(f"  AR:      {ar_kv:>10,} tokens")
        print(f"  DFlash:  KV cache 信息未提取到 (可能启动失败)")
    else:
        print(f"  KV cache 信息提取失败")

    # 写结果
    out_path = os.path.join(OUTPUT_DIR, f"step14_vllm_observation_{TODAY}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  输出: {out_path}")


if __name__ == "__main__":
    main()
