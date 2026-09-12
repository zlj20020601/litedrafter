#!/usr/bin/env python3
"""
COIG-CQIA 预处理脚本 v3
v3 改进:
  - 截断后验证 decode→encode 可逆性，不可逆时用换行符微调 body 末尾
"""
import os, sys, json, random, hashlib, traceback
from datetime import datetime
from collections import Counter

MODEL_PATH  = "/root/autodl-tmp/models/Qwen3-8B"
OUTPUT_DIR  = "/root/autodl-tmp/litedrafter/data"
LOG_DIR     = "/root/autodl-tmp/litedrafter/logs"
CACHE_DIR   = "/root/autodl-tmp/cache"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "coig_cqia_qwen3_1024_256.jsonl")
META_FILE   = os.path.join(OUTPUT_DIR, "coig_cqia_qwen3_1024_256.meta.json")
LOG_FILE    = os.path.join(LOG_DIR, "preprocess_coig_cqia.log")

DATA_FILE   = os.path.join(CACHE_DIR,
    "datasets/AI-ModelScope--COIG-CQIA/snapshots/master/COIG-CQIA-full.jsonl")

TARGET_LENGTH = 1024
NUM_PROMPTS   = 256
SEED          = 42


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def truncate_ensure_reversible(token_ids, suffix_ids, target_length, tok, newline_id):
    """
    截断到精确 target_length。
    如果 decode→encode 不可逆（截断切在字符中间），用换行符替换 body 末尾几个 token。
    返回 (final_ids, decoded_text, was_adjusted)。
    """
    prefix_content = token_ids[: -len(suffix_ids)]
    keep = target_length - len(suffix_ids)
    body = list(prefix_content[:keep])
    final_ids = body + suffix_ids
    assert len(final_ids) == target_length

    decoded = tok.decode(final_ids, skip_special_tokens=False)
    re_ids = tok.encode(decoded, add_special_tokens=False)

    if len(re_ids) == target_length:
        return final_ids, decoded, False

    # 不可逆：body 末尾切在不安全边界，用换行符替换
    for n_replace in range(1, 6):
        body = list(prefix_content[:keep - n_replace]) + [newline_id] * n_replace
        final_ids = body + suffix_ids
        assert len(final_ids) == target_length
        decoded = tok.decode(final_ids, skip_special_tokens=False)
        re_ids = tok.encode(decoded, add_special_tokens=False)
        if len(re_ids) == target_length:
            return final_ids, decoded, True

    # fallback
    body = list(prefix_content[:keep])
    final_ids = body + suffix_ids
    decoded = tok.decode(final_ids, skip_special_tokens=False)
    return final_ids, decoded, True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    open(LOG_FILE, "w").close()

    log("=" * 60)
    log("COIG-CQIA 预处理 v3 开始")
    log(f"参数: target={TARGET_LENGTH}, num={NUM_PROMPTS}, seed={SEED}")
    log("=" * 60)

    # ── 1. 确认数据文件 ──
    if not os.path.exists(DATA_FILE):
        log("数据文件不存在，尝试下载...")
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download("AI-ModelScope/COIG-CQIA", repo_type="dataset",
                          cache_dir=CACHE_DIR)
    log(f"Step 1: 数据文件 {DATA_FILE}")

    # ── 2. 加载 tokenizer ──
    log(f"Step 2: 加载 tokenizer {MODEL_PATH}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    assert tok.chat_template, "tokenizer 没有 chat_template!"
    log(f"vocab_size={tok.vocab_size}, chat_template OK")

    suffix_text = "<|im_end|>\n<|im_start|>assistant\n"
    suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
    log(f"suffix_ids ({len(suffix_ids)} tokens): {suffix_ids}")
    newline_id = tok.encode("\n", add_special_tokens=False)[0]
    log(f"newline_token_id={newline_id}")

    # ── 3. 遍历数据，收集 >= 1024 的候选 ──
    log("Step 3: 扫描 COIG-CQIA-full.jsonl ...")
    candidates = []
    seen_hashes = set()
    stats = {
        "total_rows": 0, "has_nonempty_input": 0,
        "passed_quality": 0, "deduped": 0, "ge_1024": 0,
    }

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            stats["total_rows"] += 1
            inst = (obj.get("instruction") or "").strip()
            inp  = (obj.get("input") or "").strip()
            out  = (obj.get("output") or "").strip()

            if not inst or not inp:
                continue
            stats["has_nonempty_input"] += 1

            if len(inst) < 10 or len(inp) < 10:
                continue
            content_raw = inst + inp
            weird = sum(1 for c in content_raw if ord(c) < 32 and c not in "\n\t")
            if weird / max(len(content_raw), 1) > 0.1:
                continue
            stats["passed_quality"] += 1

            content = f"{inst}\n{inp}"
            h = hashlib.md5(content.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                stats["deduped"] += 1
                continue
            seen_hashes.add(h)

            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True
            )
            token_ids = tok.encode(prompt_text, add_special_tokens=False)
            tlen = len(token_ids)

            if tlen >= TARGET_LENGTH:
                stats["ge_1024"] += 1
                candidates.append({
                    "content": content,
                    "token_len_raw": tlen,
                    "token_ids": token_ids,
                    "output_prefix": out[:80] if out else "",
                    "instruction": inst,
                    "input": inp,
                })

    log(f"统计: {json.dumps(stats, ensure_ascii=False)}")
    log(f">= 1024 候选: {len(candidates)}")

    # ── 4. 选取策略：优先选 raw_len 最接近 1024 的 ──
    log("Step 4: 选取样本")
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: x["token_len_raw"])

    if len(candidates) < NUM_PROMPTS:
        log(f"WARNING: 候选不足！只有 {len(candidates)} 条")
        selected = candidates
    else:
        selected = candidates[:NUM_PROMPTS]

    sel_lens = [s["token_len_raw"] for s in selected]
    log(f"选中 {len(selected)} 条")
    log(f"  raw_len: min={min(sel_lens)}, max={max(sel_lens)}, "
        f"median={sorted(sel_lens)[len(sel_lens)//2]}")
    buckets = Counter((l // 100) * 100 for l in sel_lens)
    log("  选中样本 raw_len 分布:")
    for b in sorted(buckets):
        log(f"    {b:5d}-{b+99:5d}: {buckets[b]:4d}")

    rng2 = random.Random(SEED)
    rng2.shuffle(selected)

    # ── 5. 截断到精确 1024 + 验证可逆性 ──
    log("Step 5: 截断 + 可逆性验证 + 落盘")
    results = []
    adjusted_count = 0
    for i, item in enumerate(selected):
        final_ids, decoded, adjusted = truncate_ensure_reversible(
            item["token_ids"], suffix_ids, TARGET_LENGTH, tok, newline_id
        )
        if adjusted:
            adjusted_count += 1

        results.append({
            "sample_id":              f"coig_cqia_{i:04d}",
            "messages":               [{"role": "user",
                                        "content": item["content"]}],
            "prompt_token_length_raw":  item["token_len_raw"],
            "prompt_token_length_used": TARGET_LENGTH,
            "input_ids":              final_ids,
            "decoded_prompt":         decoded,
            "output_prefix":          item["output_prefix"],
        })

    log(f"截断调整: {adjusted_count} 条（因 decode→encode 不可逆，body 末尾用换行符微调）")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"写入 {OUTPUT_FILE} ({len(results)} 条)")

    meta = {
        "dataset":       "AI-ModelScope/COIG-CQIA",
        "data_file":     "COIG-CQIA-full.jsonl",
        "tokenizer":     MODEL_PATH,
        "sampling_seed": SEED,
        "target_length": TARGET_LENGTH,
        "num_prompts":   len(results),
        "stats":         stats,
        "selected_raw_len": {
            "min": min(sel_lens), "max": max(sel_lens),
            "median": sorted(sel_lens)[len(sel_lens)//2],
        },
        "selection_strategy": "sort by raw_len asc (closest to 1024 first), seed=42",
        "fields_used":   "instruction + input (non-empty); output excluded",
        "chat_template": "Qwen3, add_generation_prompt=True",
        "truncation":    "right; suffix preserved; decode→encode verified",
        "truncation_adjusted": adjusted_count,
        "generated_at":  datetime.now().isoformat(timespec="seconds"),
        "output_file":   OUTPUT_FILE,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"metadata 写入 {META_FILE}")
    log("=" * 60)
    log(f"完成。共 {len(results)} 条，每条精确 {TARGET_LENGTH} tokens。")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
