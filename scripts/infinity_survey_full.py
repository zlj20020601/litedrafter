#!/usr/bin/env python3
"""
Infinity-Instruct-0625 全量预统计 Pipeline

严格顺序：
  1. 全量加载（7 shard parquet，约 66 万条）
  2. 提取 user-side request（conversations[0].value）
  3. 中文过滤（langdetect == zh 或 CJK 字符占比 >= 30%）
  4. 去 exact duplicate（md5）
  5. 去 near duplicate（MinHash LSH, Jaccard threshold=0.9）
  6. Qwen3-8B tokenizer
  7. 统一 chat template (enable_thinking=False, add_generation_prompt=True)
  8. 统计 FINAL rendered prompt length

输出：分桶统计表 + 逐步漏斗计数
"""
import os, sys, json, re, hashlib, time, traceback
from collections import OrderedDict, Counter
from datetime import datetime

# ── 配置 ──
MODEL_PATH  = "/root/autodl-tmp/models/Qwen3-8B"
CACHE_DIR   = "/root/autodl-tmp/cache"
DATA_GLOB   = os.path.join(CACHE_DIR,
    "datasets/BAAI--Infinity-Instruct/snapshots/master/0625/*.parquet")
LOG_DIR     = "/root/autodl-tmp/litedrafter/logs"
LOG_FILE    = os.path.join(LOG_DIR, "infinity_survey_full.log")
RESULT_FILE = os.path.join(LOG_DIR, "infinity_survey_result.json")

# MinHash 配置
NEAR_DUP_THRESHOLD = 0.9    # Jaccard 相似度阈值
NUM_PERM            = 128   # MinHash 排列数

# 分桶
BUCKETS = OrderedDict([
    ("<768",      (0, 767)),
    ("768-1023",  (768, 1023)),
    ("1024-1152", (1024, 1152)),
    ("1153-1280", (1153, 1280)),
    ("1281-1536", (1281, 1536)),
    ("1537-2048", (1537, 2048)),
    (">2048",     (2049, 999999)),
])

CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7f]')


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def extract_user_request(obj):
    """
    从一条记录中提取 user-side request。
    Infinity-Instruct 格式: conversations 列表，每项有 from/value。
    取第一个 human turn 的 value。
    """
    convs = obj.get("conversations") or []
    if not isinstance(convs, list):
        return None
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("from") or "").lower()
        val  = (turn.get("value") or "").strip()
        if role in ("human", "user") and val:
            return val
    return None


def is_chinese(text):
    """
    中文过滤：
    优先用 langdetect 字段（如果数据集提供），
    否则用 CJK 字符占比 >= 30% 判定。
    """
    if not text or len(text) < 10:
        return False
    cjk_count = len(CJK_PATTERN.findall(text))
    ratio = cjk_count / len(text)
    return ratio >= 0.30


def shingle(text, k=3):
    """character k-gram shingle for MinHash"""
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    open(LOG_FILE, "w").close()

    log("=" * 70)
    log("Infinity-Instruct-0625 全量预统计 Pipeline")
    log("=" * 70)

    import glob as globmod
    parquet_files = sorted(globmod.glob(DATA_GLOB))
    log(f"Step 0: 找到 {len(parquet_files)} 个 parquet shard")
    for f in parquet_files:
        log(f"  {os.path.basename(f)} ({os.path.getsize(f)//1024//1024}MB)")

    # ── Step 1: 全量加载 ──
    log("")
    log("Step 1: 全量加载 parquet ...")
    import pyarrow.parquet as pq

    all_records = []
    total_rows = 0
    for fpath in parquet_files:
        pf = pq.ParquetFile(fpath)
        nrows = pf.metadata.num_rows
        total_rows += nrows
        for batch in pf.iter_batches(batch_size=20000):
            d = batch.to_pydict()
            n = len(next(iter(d.values())))
            for i in range(n):
                all_records.append({k: v[i] for k, v in d.items()})
        log(f"  {os.path.basename(fpath)}: {nrows} rows loaded")
    log(f"Step 1 完成: 总计 {len(all_records)} 条 (metadata: {total_rows})")

    funnel = {"step1_total": len(all_records)}

    # ── Step 2: 提取 user-side request ──
    log("")
    log("Step 2: 提取 user-side request ...")
    items = []  # (record_id, content, langdetect, source)
    no_user = 0
    for idx, obj in enumerate(all_records):
        content = extract_user_request(obj)
        if content and len(content) >= 10:
            lang = obj.get("langdetect") or ""
            src  = obj.get("source") or ""
            items.append((idx, content, lang, src))
        else:
            no_user += 1
    log(f"Step 2 完成: {len(items)} 条有 user request (跳过 {no_user} 条)")
    funnel["step2_has_user"] = len(items)

    # 释放原始记录内存
    del all_records

    # ── Step 3: 中文过滤 ──
    log("")
    log("Step 3: 中文过滤 ...")
    zh_items = []
    lang_field_zh = 0
    cjk_ratio_zh = 0
    for idx, content, lang, src in items:
        # 优先用 langdetect 字段
        if lang and lang.lower() in ("zh", "zh-cn", "chinese", "zhongwen"):
            zh_items.append((idx, content, src))
            lang_field_zh += 1
        elif is_chinese(content):
            zh_items.append((idx, content, src))
            cjk_ratio_zh += 1
    log(f"Step 3 完成: {len(zh_items)} 条中文")
    log(f"  (langdetect=zh: {lang_field_zh}, CJK ratio>=30%: {cjk_ratio_zh})")
    funnel["step3_chinese"] = len(zh_items)

    del items

    # ── Step 4: 去 exact duplicate ──
    log("")
    log("Step 4: 去 exact duplicate (md5) ...")
    seen_md5 = set()
    exact_dedup = []
    exact_dup_count = 0
    for idx, content, src in zh_items:
        h = hashlib.md5(content.encode("utf-8")).hexdigest()
        if h in seen_md5:
            exact_dup_count += 1
            continue
        seen_md5.add(h)
        exact_dedup.append((idx, content, src))
    log(f"Step 4 完成: {len(exact_dedup)} 条 (去除 {exact_dup_count} exact dup)")
    funnel["step4_exact_dedup"] = len(exact_dedup)

    del zh_items

    # ── Step 5: 去 near duplicate (MinHash LSH) ──
    log("")
    log("Step 5: 去 near duplicate (MinHash LSH, threshold=0.9) ...")
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=NEAR_DUP_THRESHOLD, num_perm=NUM_PERM)
    near_dedup = []
    near_dup_count = 0
    processed = 0
    t0 = time.time()

    for idx, content, src in exact_dedup:
        mh = MinHash(num_perm=NUM_PERM)
        shingles = shingle(content, k=3)
        for s in shingles:
            mh.update(s.encode("utf-8"))

        # 查询是否已有近似项
        result = lsh.query(mh)
        if result:
            near_dup_count += 1
        else:
            lsh.insert(f"id_{idx}", mh)
            near_dedup.append((idx, content, src))

        processed += 1
        if processed % 50000 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            log(f"  near-dedup 进度: {processed}/{len(exact_dedup)} "
                f"({rate:.0f}/s, near_dup={near_dup_count})")

    elapsed_nd = time.time() - t0
    log(f"Step 5 完成: {len(near_dedup)} 条 (去除 {near_dup_count} near dup, "
        f"耗时 {elapsed_nd:.1f}s)")
    funnel["step5_near_dedup"] = len(near_dedup)

    del exact_dedup

    # ── Step 6+7+8: tokenize → chat template → 统计长度 ──
    log("")
    log("Step 6+7+8: Qwen3-8B tokenizer → chat template → 统计长度 ...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    log(f"tokenizer loaded, vocab_size={tok.vocab_size}")

    bucket_counts = OrderedDict((k, 0) for k in BUCKETS)
    all_lengths = []
    processed = 0
    t0 = time.time()

    for idx, content, src in near_dedup:
        try:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            ids = tok.encode(prompt, add_special_tokens=False)
            tlen = len(ids)
            all_lengths.append(tlen)

            for bname, (lo, hi) in BUCKETS.items():
                if lo <= tlen <= hi:
                    bucket_counts[bname] += 1
                    break
        except Exception:
            pass

        processed += 1
        if processed % 50000 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            log(f"  tokenize 进度: {processed}/{len(near_dedup)} ({rate:.0f}/s)")

    elapsed_tok = time.time() - t0
    total_valid = sum(bucket_counts.values())
    log(f"Step 8 完成: {total_valid} 条有效 (耗时 {elapsed_tok:.1f}s)")
    funnel["step8_final_valid"] = total_valid

    # ── 输出分桶统计 ──
    log("")
    log("=" * 70)
    log("FINAL 分桶统计 (Infinity-Instruct-0625, 全量, 中文, 去重)")
    log("=" * 70)
    log("")
    log(f"{'分桶':<16} {'条数':>10} {'占比':>8}")
    log("-" * 38)
    ge_768 = 0
    ge_1024 = 0
    for bname, (lo, hi) in BUCKETS.items():
        cnt = bucket_counts[bname]
        pct = 100 * cnt / max(total_valid, 1)
        bar = "#" * min(int(pct / 2), 40)
        log(f"{bname:<16} {cnt:>10} {pct:>7.1f}% {bar}")
        if lo >= 768:
            ge_768 += cnt
        if lo >= 1024:
            ge_1024 += cnt
    log("-" * 38)
    log(f">= 768:         {ge_768:>10} ({100*ge_768/max(total_valid,1):.1f}%)")
    log(f">= 1024:        {ge_1024:>10} ({100*ge_1024/max(total_valid,1):.1f}%)")

    a_cnt = bucket_counts["1024-1152"]
    ab1 = a_cnt + bucket_counts["1153-1280"]
    log(f"")
    log(f"关键指标:")
    log(f"  A桶(1024-1152):  {a_cnt}")
    log(f"  A+B1(1024-1280): {ab1}")
    log(f"  >= 1024 总计:    {ge_1024}")

    # ── 漏斗 ──
    log(f"")
    log("=" * 70)
    log("处理漏斗")
    log("=" * 70)
    for k, v in funnel.items():
        log(f"  {k:<24} {v:>10}")

    # 长度分布详情
    if all_lengths:
        all_lengths.sort()
        log(f"")
        log(f"长度统计 (n={len(all_lengths)}):")
        log(f"  min={min(all_lengths)}, max={max(all_lengths)}")
        log(f"  P25={all_lengths[len(all_lengths)//4]}")
        log(f"  P50={all_lengths[len(all_lengths)//2]}")
        log(f"  P75={all_lengths[3*len(all_lengths)//4]}")
        log(f"  P90={all_lengths[int(len(all_lengths)*0.9)]}")
        log(f"  P95={all_lengths[int(len(all_lengths)*0.95)]}")
        log(f"  P99={all_lengths[int(len(all_lengths)*0.99)]}")

    # 保存结果 JSON
    result = {
        "funnel": funnel,
        "buckets": dict(bucket_counts),
        "ge_768": ge_768,
        "ge_1024": ge_1024,
        "a_bucket": a_cnt,
        "ab1_bucket": ab1,
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "near_dup_removed": near_dup_count,
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"\n结果已保存: {RESULT_FILE}")
    log("完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
