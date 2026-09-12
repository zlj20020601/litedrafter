#!/usr/bin/env python3
"""
L1: vLLM batch capacity penalty 实验
固定配置:
  GPU 4090, Qwen3-8B BF16, max_model_len=4096,
  gpu_memory_utilization=0.90, max_num_batched_tokens=16384,
  num_speculative_tokens=15, prefix_caching=false, enforce_eager=true
流程: 启动 vLLM (AR/DFlash) → 提取显存/KV 指标 → 并发扫描 → 输出 json
"""
import json, os, re, subprocess, sys, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

TARGET_MODEL = "/root/autodl-tmp/models/Qwen3-8B"
DRAFT_MODEL  = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
PYTHON       = "/root/autodl-tmp/conda_envs/env_dcut/bin/python"
DATA_FILE    = "/root/autodl-tmp/litedrafter/data/codecontests_qwen3_1024_256.jsonl"
OUTPUT_DIR   = "/root/autodl-tmp/litedrafter/outputs"
LOGS_DIR     = "/root/autodl-tmp/litedrafter/logs"
TODAY        = datetime.now().strftime("%Y%m%d")
NOW          = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

INPUT_LEN   = 1024
OUTPUT_LEN  = 256
PER_REQ     = INPUT_LEN + OUTPUT_LEN   # 1280 tokens/req 预算
PORT        = 8299

SERVE_COMMON = [
    "--trust-remote-code", "--dtype", "bfloat16",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.90",
    "--max-num-seqs", "32",
    "--max-num-batched-tokens", "16384",
    "--no-enable-prefix-caching",
    "--tensor-parallel-size", "1",
    "--served-model-name", "qwen3-8b",
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
            "method": "dflash", "model": DRAFT_MODEL,
            "num_speculative_tokens": 15, "dflash_dcut": 0,
        })
        cmd += ["--speculative-config", spec]
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    conda_bin = os.path.dirname(PYTHON)
    env["PATH"] = conda_bin + ":" + env.get("PATH", "")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return proc, log_file

def wait_ready(log_file, timeout=240):
    keywords_ready = ["Application startup complete", "Uvicorn running"]
    keywords_error = ["Traceback", "CUDA out of memory", "ValueError", "AssertionError"]
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
            time.sleep(2)
            with open(log_file, errors="replace") as f:
                content = f.read()
            return content, True
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
    m = re.search(r"consumed memory \(weights \+ non-torch\),\s*([\d.]+)\s*GiB for peak activation", log_content)
    if m:
        info["peak_activation_gib"] = float(m.group(1))
    m = re.search(r"Actual usage is\s*([\d.]+)\s*GiB for consumed memory", log_content)
    if m:
        info["consumed_memory_gib"] = float(m.group(1))
    # 每请求预算
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
    import urllib.request
    body = json.dumps({
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": output_len,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            dt = time.time() - t0
            return {"ok": True, "latency": dt, "n_tokens": len(data.get("choices", [{}])[0].get("message", {}).get("content", "").split())}
    except Exception as e:
        dt = time.time() - t0
        return {"ok": False, "latency": dt, "error": str(e)[:200]}

def concurrency_scan(port, prompts, batches=(1, 2, 4, 8, 16, 24, 32)):
    """发并发请求，记录成功率与平均延迟"""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
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
        avg_lat = sum(lat) / len(lat) if lat else None
        results[b] = {
            "concurrency": b, "success": ok, "failed": b - ok,
            "avg_latency_s": round(avg_lat, 2) if avg_lat else None,
            "errors": [d["error"] for d in done if not d["ok"]][:2],
        }
        log(f"  batch={b}: success={ok}/{b}, avg_lat={avg_lat:.2f}s" if avg_lat else f"  batch={b}: success={ok}/{b}")
        pool.shutdown()
    return results

def kill_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], timeout=5)
    time.sleep(3)

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
        f"weights={metrics.get('model_weights_gib')} GiB, 理论 batch={metrics.get('theoretical_batch')}")
    scan = concurrency_scan(PORT, prompts)
    kill_server(proc)
    return {"mode": mode, "ready": True, "metrics": metrics, "concurrency_scan": scan}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    prompts = load_prompts()
    log(f"加载 {len(prompts)} 条 prompt")

    results = {"timestamp": NOW, "config": {
        "gpu": "RTX 4090 24GB", "model": "Qwen3-8B BF16",
        "max_model_len": 4096, "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": 16384, "num_speculative_tokens": 15,
        "prefix_caching": False, "enforce_eager": True,
        "input_length": INPUT_LEN, "output_length": OUTPUT_LEN,
    }}
    for mode in ("ar", "dflash"):
        results[mode] = run_mode(mode, prompts)

    # 核心表
    log("")
    log("=" * 70)
    log("第一层核心表")
    log("=" * 70)
    for mode in ("ar", "dflash"):
        r = results[mode]
        if not r.get("ready"):
            log(f"  {mode}: server 未就绪")
            continue
        m = r["metrics"]
        scan = r["concurrency_scan"]
        max_ok = max((b for b, s in scan.items() if s["success"] == s["concurrency"]), default=None)
        log(f"  {mode.upper()}: KV={m['kv_tokens']} tokens, 每请求={PER_REQ}, "
            f"理论batch={m['theoretical_batch']}, 实际全部成功最大并发={max_ok}")
        for b, s in scan.items():
            if s["failed"] > 0:
                log(f"    batch={b}: 失败 {s['failed']}/{s['concurrency']} {s['errors'][:1]}")

    out_file = os.path.join(OUTPUT_DIR, f"l1_capacity_{TODAY}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"结果已保存: {out_file}")

if __name__ == "__main__":
    main()
