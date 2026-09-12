#!/usr/bin/env python3
"""
L1: vLLM batch capacity penalty 实验 — Qwen3.5-4B 版本
固定配置:
  GPU 4090, Qwen3.5-4B BF16, max_model_len=4096,
  gpu_memory_utilization=0.90, max_num_batched_tokens=16384,
  num_speculative_tokens=15, prefix_caching=false, enforce_eager=true
流程: 启动 vLLM (AR/DFlash) → 提取显存/KV 指标 → 并发扫描 → 输出 json

与 Qwen3-8B 版本的关键差异:
  1. 模型路径 → Qwen3.5-4B / Qwen3.5-4B-DFlash
  2. Python → env_vllm026
  3. VLLM_USE_V2_MODEL_RUNNER=1（DFlash 需要）
  4. LD_LIBRARY_PATH → env_vllm026/lib（libstdc++ 依赖）
  5. AR 启动 ~3.5min（多模态 warmup），timeout 增大
  6. 使用 /v1/completions API（Qwen3.5 supported_tasks=['generate']）
"""
import json, os, re, subprocess, sys, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

TARGET_MODEL = "/root/autodl-tmp/models/Qwen3.5-4B"
DRAFT_MODEL  = "/root/autodl-tmp/models/Qwen3.5-4B-DFlash"
CONDA_ENV    = "/root/autodl-tmp/conda_envs/env_vllm026"
PYTHON       = f"{CONDA_ENV}/bin/python"
DATA_FILE    = "/root/autodl-tmp/litedrafter/data/codecontests_qwen35_4b_1024_256.jsonl"
OUTPUT_DIR   = "/root/autodl-tmp/litedrafter/outputs"
LOGS_DIR     = "/root/autodl-tmp/litedrafter/logs"
TODAY        = datetime.now().strftime("%Y%m%d")
NOW          = datetime.now().isoformat(timespec="seconds")

INPUT_LEN   = 1024
OUTPUT_LEN  = 256
PER_REQ     = INPUT_LEN + OUTPUT_LEN   # 1280 tokens/req 预算
PORT        = 8299
MODEL_NAME  = "qwen35-4b"

SERVE_COMMON = [
    "--trust-remote-code", "--dtype", "bfloat16",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.90",
    "--max-num-seqs", "32",
    "--max-num-batched-tokens", "16384",
    "--no-enable-prefix-caching",
    "--tensor-parallel-size", "1",
    "--served-model-name", MODEL_NAME,
    "--enforce-eager",
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def start_server(mode, port=PORT):
    log_file = os.path.join(LOGS_DIR, f"l1_{mode}_server_{TODAY}.log")
    cmd = [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
           "--model", TARGET_MODEL, "--port", str(port)] + SERVE_COMMON[:]
    if mode == "dflash":
        spec = json.dumps({
            "method": "dflash",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 15,
        })
        cmd += ["--speculative-config", spec]

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # libstdc++ 等 lib 依赖
    env["LD_LIBRARY_PATH"] = f"{CONDA_ENV}/lib:" + env.get("LD_LIBRARY_PATH", "")
    conda_bin = os.path.dirname(PYTHON)
    env["PATH"] = conda_bin + ":" + env.get("PATH", "")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return proc, log_file


def wait_ready(log_file, timeout=420):
    """等待 vLLM server 就绪。AR 模式因多模态 warmup 需要更长时间。"""
    keywords_ready = ["Application startup complete", "Uvicorn running",
                      "Starting vLLM server on"]
    keywords_error = ["Traceback", "CUDA out of memory", "ValueError",
                      "AssertionError", "RuntimeError"]
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(3)
        if not os.path.exists(log_file):
            continue
        with open(log_file, errors="replace") as f:
            content = f.read()
        if any(k in content for k in keywords_error):
            return content, False
        if any(k in content for k in keywords_ready):
            time.sleep(3)
            with open(log_file, errors="replace") as f:
                content = f.read()
            return content, True
        elapsed = int(time.time() - start)
        if elapsed % 30 == 0:
            log(f"  ...等待 server 就绪 ({elapsed}s)")
    with open(log_file, errors="replace") as f:
        return f.read(), False


def extract_metrics(log_content):
    info = {}
    def grab(pattern, key, cast=str):
        m = re.search(pattern, log_content)
        info[key] = cast(m.group(1)) if m else None

    grab(r"Model loading took\s+([\d.]+)\s+GiB", "model_weights_gib", float)
    grab(r"Available KV cache memory:\s*([\d.]+)\s*GiB", "kv_memory_gib", float)
    grab(r"GPU KV cache size:\s*([\d,]+)\s*tokens", "kv_tokens", lambda s: int(s.replace(",", "")))
    grab(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", "max_concurrency_4096", float)
    # peak activation
    m = re.search(r"Actual usage is\s*([\d.]+)\s*GiB for.*weight.*?([\d.]+)\s*GiB for peak activation", log_content)
    if m:
        info["consumed_memory_gib"] = float(m.group(1))
        info["peak_activation_gib"] = float(m.group(2))
    else:
        grab(r"Actual usage is\s*([\d.]+)\s*GiB for consumed memory", "consumed_memory_gib", float)
        grab(r"([\d.]+)\s*GiB for peak activation", "peak_activation_gib", float)
    # non-torch memory
    m = re.search(r"([\d.]+)\s*GiB for non-torch memory", log_content)
    if m:
        info["non_torch_memory_gib"] = float(m.group(1))

    info["per_request_tokens"] = PER_REQ
    info["theoretical_batch"] = int(info["kv_tokens"] // PER_REQ) if info.get("kv_tokens") else None
    return info


def load_prompts(n=40):
    """从 CodeContests jsonl 加载 prompt 文本（循环复用）"""
    prompts = []
    with open(DATA_FILE) as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["messages"][0]["content"])
            if len(prompts) >= n:
                break
    return prompts


def send_request(url, content, output_len, timeout=300):
    """使用 /v1/completions API（Qwen3.5 supported_tasks=['generate']）"""
    import urllib.request
    body = json.dumps({
        "model": MODEL_NAME,
        "prompt": content,
        "max_tokens": output_len,
        "temperature": 0.0,
        "seed": 42,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            dt = time.time() - t0
            # completions API 返回 usage.completion_tokens
            n_tokens = data.get("usage", {}).get("completion_tokens", 0)
            return {"ok": True, "latency": dt, "n_tokens": n_tokens,
                    "tok_s": n_tokens / dt if dt > 0 else 0}
    except Exception as e:
        dt = time.time() - t0
        return {"ok": False, "latency": dt, "error": str(e)[:200]}


def concurrency_scan(port, prompts, batches=(1, 2, 4, 8, 16, 24, 32)):
    """发并发请求，记录成功率、延迟、吞吐"""
    url = f"http://127.0.0.1:{port}/v1/completions"
    results = {}
    for b in batches:
        pool = ThreadPoolExecutor(max_workers=b)
        futures = []
        for j in range(b):
            p = prompts[j % len(prompts)]
            futures.append(pool.submit(send_request, url, p, OUTPUT_LEN))
        done = [f.result() for f in futures]
        ok = sum(1 for d in done if d["ok"])
        lat = [d["latency"] for d in done if d["ok"]]
        toks = [d["n_tokens"] for d in done if d["ok"]]
        avg_lat = sum(lat) / len(lat) if lat else None
        total_tokens = sum(toks)
        # 端到端吞吐: 总输出 token / 最大延迟（最慢请求决定批次完成时间）
        max_lat = max(lat) if lat else None
        throughput = total_tokens / max_lat if max_lat and max_lat > 0 else None
        results[b] = {
            "concurrency": b, "success": ok, "failed": b - ok,
            "avg_latency_s": round(avg_lat, 2) if avg_lat else None,
            "max_latency_s": round(max_lat, 2) if max_lat else None,
            "total_output_tokens": total_tokens,
            "throughput_tok_s": round(throughput, 1) if throughput else None,
            "errors": [d["error"] for d in done if not d["ok"]][:2],
        }
        status = f"success={ok}/{b}"
        if avg_lat:
            status += f", avg_lat={avg_lat:.2f}s"
        if throughput:
            status += f", throughput={throughput:.1f} tok/s"
        log(f"  batch={b}: {status}")
        pool.shutdown()
    return results


def kill_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], timeout=5)
    time.sleep(5)


def run_mode(mode, prompts):
    log(f"=== {mode.upper()} ===")
    proc, log_file = start_server(mode)
    content, ready = wait_ready(log_file)
    if not ready:
        log(f"  {mode} server 启动失败")
        kill_server(proc)
        return {"mode": mode, "ready": False, "log_tail": content[-2000:]}

    log(f"  {mode} server 就绪")
    metrics = extract_metrics(content)
    log(f"  KV tokens={metrics.get('kv_tokens')}, KV mem={metrics.get('kv_memory_gib')} GiB, "
        f"weights={metrics.get('model_weights_gib')} GiB, "
        f"peak_act={metrics.get('peak_activation_gib')} GiB, "
        f"理论 batch={metrics.get('theoretical_batch')}")

    # warmup 1 请求
    log("  warmup 请求 ...")
    warmup = send_request(f"http://127.0.0.1:{PORT}/v1/completions",
                          prompts[0], 8, timeout=120)
    if not warmup["ok"]:
        log(f"  warmup 失败: {warmup.get('error')}")
    else:
        log(f"  warmup OK ({warmup['latency']:.1f}s)")

    scan = concurrency_scan(PORT, prompts)
    kill_server(proc)
    return {"mode": mode, "ready": True, "metrics": metrics, "concurrency_scan": scan}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    prompts = load_prompts()
    log(f"加载 {len(prompts)} 条 prompt")

    results = {"timestamp": NOW, "config": {
        "gpu": "RTX 4090 24GB",
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
        "vllm_version": "0.26.0",
        "conda_env": "env_vllm026",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 32,
        "num_speculative_tokens": 15,
        "prefix_caching": False,
        "enforce_eager": True,
        "input_length": INPUT_LEN,
        "output_length": OUTPUT_LEN,
        "data_file": DATA_FILE,
    }}

    for mode in ("ar", "dflash"):
        results[mode] = run_mode(mode, prompts)

    # 核心表
    log("")
    log("=" * 70)
    log("L1 核心表 — Qwen3.5-4B")
    log("=" * 70)
    for mode in ("ar", "dflash"):
        r = results[mode]
        if not r.get("ready"):
            log(f"  {mode}: server 未就绪")
            continue
        m = r["metrics"]
        scan = r["concurrency_scan"]
        max_ok = max((b for b, s in scan.items() if s["success"] == s["concurrency"]), default=None)
        log(f"  {mode.upper()}: weights={m.get('model_weights_gib')} GiB, "
            f"KV={m.get('kv_tokens')} tokens ({m.get('kv_memory_gib')} GiB), "
            f"peak_act={m.get('peak_activation_gib')} GiB, "
            f"理论batch={m.get('theoretical_batch')}, 最大全成功并发={max_ok}")

    # L2 静态预算 vs 实际容量
    if results.get("ar", {}).get("ready") and results.get("dflash", {}).get("ready"):
        ar_m = results["ar"]["metrics"]
        df_m = results["dflash"]["metrics"]
        log("")
        log("=" * 70)
        log("L2 静态预算 vs 实际容量")
        log("=" * 70)
        if ar_m.get("kv_memory_gib") and df_m.get("kv_memory_gib"):
            static_ratio = df_m["kv_memory_gib"] / ar_m["kv_memory_gib"]
            actual_ratio = df_m["kv_tokens"] / ar_m["kv_tokens"]
            gap = abs(static_ratio - actual_ratio) / static_ratio if static_ratio else None
            weight_delta = df_m.get("model_weights_gib", 0) - ar_m.get("model_weights_gib", 0)
            log(f"  静态预算比例 (DFlash/AR KV mem): {static_ratio:.3f}")
            log(f"  实际容量比例 (DFlash/AR KV tokens): {actual_ratio:.3f}")
            log(f"  相对差距: {gap:.1%}" if gap else "  相对差距: N/A")
            log(f"  权重增量: +{weight_delta:.2f} GiB")
            results["l2_analysis"] = {
                "static_ratio": round(static_ratio, 3),
                "actual_ratio": round(actual_ratio, 3),
                "gap_pct": round(gap * 100, 1) if gap else None,
                "weight_delta_gib": round(weight_delta, 2),
            }

    out_file = os.path.join(OUTPUT_DIR, f"l1_capacity_qwen35_{TODAY}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"结果已保存: {out_file}")


if __name__ == "__main__":
    main()
