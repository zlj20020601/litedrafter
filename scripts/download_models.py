#!/usr/bin/env python3
"""Download Qwen3-8B and DFlash drafter from ModelScope."""
import os, sys, time

os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/cache/modelscope'

from modelscope import snapshot_download

MODELS = [
    ("qwen/Qwen3-8B", "Qwen3-8B"),
    ("z-lab/Qwen3-8B-DFlash-b16", "Qwen3-8B-DFlash-b16"),
]

BASE = "/root/autodl-tmp/models"

for model_id, local_name in MODELS:
    dest = os.path.join(BASE, local_name)
    if os.path.exists(dest) and os.listdir(dest):
        print(f"[SKIP] {local_name} already exists", flush=True)
        continue
    print(f"\n{'='*60}", flush=True)
    print(f"[DOWNLOAD] {model_id} -> {dest}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    try:
        path = snapshot_download(
            model_id=model_id,
            cache_dir=BASE,
            local_dir=dest,
        )
        elapsed = time.time() - t0
        size_gb = sum(
            os.path.getsize(os.path.join(r,f))
            for r,_,fs in os.walk(dest)
            for f in fs
        ) / 1e9
        print(f"[DONE] {local_name}: {size_gb:.1f} GB in {elapsed:.0f}s", flush=True)
    except Exception as e:
        print(f"[ERROR] {model_id}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

print("\n[ALL DONE]", flush=True)
