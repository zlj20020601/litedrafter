#!/usr/bin/env python3
"""
prepare_data.py - 准备中文真实数据集 (COIG-CQIA)

直接下载 JSONL 文件并解析，绕过 MsDataset API 兼容性问题。
按 Qwen tokenizer 后的 token 数分桶: 512 / 1024 / 2048
规则: 不够长度跳过(不padding), 超长截断, 每桶 8 条
"""

import gc
import json
import os
from pathlib import Path

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
OUTPUT_PATH = "/root/autodl-tmp/litedrafter/data/coig_cqia_buckets.jsonl"
TARGET_BUCKETS = [512, 1024, 2048]
SAMPLES_PER_BUCKET = 8


def load_jsonl_from_cache():
    """Read all cached COIG-CQIA JSONL files directly."""
    import glob
    cache_dir = "/root/.cache/modelscope/hub/datasets/downloads"
    all_samples = []

    # Read all non-meta files (skip .json and .lock)
    for fpath in sorted(glob.glob(f"{cache_dir}/*")):
        if fpath.endswith('.json') or fpath.endswith('.lock'):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        instruction = item.get("instruction", "")
                        output_field = item.get("output", "")
                        text = f"{instruction}\n{output_field}".strip()
                        if len(text) > 100:
                            all_samples.append(text)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    print(f"  Total raw samples from cache: {len(all_samples)}")
    return all_samples


def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading COIG-CQIA from modelscope cache...")
    all_samples = load_jsonl_from_cache()

    if not all_samples:
        print("[ERROR] No samples found in cache")
        return

    print(f"Total samples: {len(all_samples)}")

    # Tokenize and bucket
    buckets = {b: [] for b in TARGET_BUCKETS}
    processed = 0

    for text in all_samples:
        if all(len(buckets[b]) >= SAMPLES_PER_BUCKET for b in TARGET_BUCKETS):
            break

        processed += 1
        ids = tokenizer.encode(text, truncation=True, max_length=4096)

        for target_len in TARGET_BUCKETS:
            if len(buckets[target_len]) >= SAMPLES_PER_BUCKET:
                continue

            if len(ids) >= target_len:
                truncated_ids = ids[:target_len]
                truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                buckets[target_len].append({
                    "bucket": target_len,
                    "text": truncated_text,
                    "token_ids": truncated_ids,
                    "token_len": target_len,
                })
                print(f"  bucket={target_len}: sample {len(buckets[target_len])} "
                      f"(from {len(ids)} tokens)")
                break

        # Progress
        if processed % 200 == 0:
            counts = {b: len(buckets[b]) for b in TARGET_BUCKETS}
            print(f"  Processed {processed}/{len(all_samples)}, buckets: {counts}")

    # Save
    Path(os.path.dirname(OUTPUT_PATH)).mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for bucket_len in TARGET_BUCKETS:
            samples = buckets[bucket_len]
            print(f"\nBucket {bucket_len}: {len(samples)} samples")
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    total = sum(len(v) for v in buckets.values())
    print(f"\n[SAVED] {OUTPUT_PATH}")
    print(f"Total: {total} samples")
    print(f"Processed {processed} raw samples to fill buckets")

    # Preview
    for bucket_len in TARGET_BUCKETS:
        samples = buckets[bucket_len]
        if samples:
            preview = samples[0]["text"][:80].replace('\n', ' ')
            print(f"\nBucket {bucket_len} sample 0:")
            print(f"  {preview}...")


if __name__ == "__main__":
    main()
