#!/usr/bin/env python3
"""
数据集预统计 v2
Infinity-Instruct (BAAI) vs Firefly-train-1.1M (AI-ModelScope)

改进：
  - Infinity-Instruct: 正确 ID BAAI/Infinity-Instruct，Parquet 格式，只下载 0625 子集第 1 个文件
  - Firefly: 正确 ID AI-ModelScope/firefly-train-1.1M，JSONL 格式
  - 支持 Parquet (pyarrow) 和 JSONL 两种格式
  - 自动识别 conversations / instruction+input / messages 三种字段格式
  - 采样 10 万条做分桶统计
"""
import os, sys, json, hashlib, random, traceback
from collections import OrderedDict

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
CACHE_DIR  = "/root/autodl-tmp/cache"
LOG_DIR    = "/root/autodl-tmp/litedrafter/logs"
LOG_FILE   = os.path.join(LOG_DIR, "dataset_survey_v2.log")

MAX_SAMPLE = 100000

BUCKETS = OrderedDict([
    ("<768",      (0, 767)),
    ("768-1023",  (768, 1023)),
    ("1024-1152", (1024, 1152)),
    ("1153-1280", (1153, 1280)),
    ("1281-1536", (1281, 1536)),
    ("1537-2048", (1537, 2048)),
    (">2048",     (2049, 999999)),
])


def log(msg):
    line = f"[{msg}]"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def extract_user_content(obj):
    """从一条数据中提取 user message 内容。自动识别字段格式。"""
    # 格式1: instruction + input (BELLE/COIG 格式)
    inst = (obj.get("instruction") or "").strip()
    inp  = (obj.get("input") or "").strip()
    if inst and inp:
        return f"{inst}\n{inp}"

    # 格式1b: 只有 input 字段，没有 instruction（Firefly 格式: kind+input+target）
    if inp and not inst and len(inp) >= 10:
        return inp

    # 格式2: conversations (ShareGPT/Infinity-Instruct 格式)
    convs = obj.get("conversations") or obj.get("conversation") or []
    if isinstance(convs, list) and len(convs) >= 1:
        for turn in convs:
            if isinstance(turn, dict):
                role = (turn.get("from") or turn.get("role") or "").lower()
                val  = (turn.get("value") or turn.get("content") or "").strip()
                if role in ("human", "user") and val:
                    return val

    # 格式3: messages
    msgs = obj.get("messages") or []
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                role = (m.get("role") or "").lower()
                content = (m.get("content") or "").strip()
                if role == "user" and content:
                    return content

    # 格式4: query + response
    q = (obj.get("query") or "").strip()
    if q:
        return q

    return None


def load_jsonl_sample(filepath, max_records):
    """逐行读取 JSONL，采样到 max_records。"""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                continue
            if len(records) >= max_records:
                break
    return records


def load_parquet_sample(filepath, max_records):
    """用 pyarrow 读取 parquet，采样到 max_records。"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(filepath)
    total_rows = pf.metadata.num_rows
    records = []
    # 逐 batch 读取，避免一次性加载整个文件
    for batch in pf.iter_batches(batch_size=10000):
        df = batch.to_pydict()
        for i in range(len(next(iter(df.values())))):
            row = {k: v[i] for k, v in df.items()}
            records.append(row)
            if len(records) >= max_records:
                return records, total_rows
    return records, total_rows


def process_dataset(ds_name, ds_config, tok):
    """处理一个数据集。"""
    log(f"{'='*60}")
    log(f"处理: {ds_name}")
    log(f"{'='*60}")

    ds_id     = ds_config["id"]
    filetype  = ds_config["type"]
    file_glob = ds_config["file_glob"]
    allow     = ds_config.get("allow_patterns")

    # 下载数据
    from modelscope.hub.snapshot_download import snapshot_download
    import glob as globmod

    log(f"下载 {ds_id} ...")
    kwargs = {"repo_type": "dataset", "cache_dir": CACHE_DIR}
    if allow:
        kwargs["allow_patterns"] = allow
    local_dir = snapshot_download(ds_id, **kwargs)
    log(f"下载完成: {local_dir}")

    # 找数据文件
    files = sorted(globmod.glob(os.path.join(local_dir, "**", file_glob), recursive=True))
    if not files:
        log(f"ERROR: 找不到匹配 {file_glob} 的文件")
        return None
    data_file = files[0]
    log(f"数据文件: {data_file} ({os.path.getsize(data_file)//1024//1024}MB)")

    # 查看第一条记录的字段
    if filetype == "jsonl":
        with open(data_file) as f:
            first = json.loads(f.readline())
        log(f"首条字段: {list(first.keys())}")
        sample_val = {k: str(v)[:80] for k, v in first.items()}
        log(f"首条预览: {json.dumps(sample_val, ensure_ascii=False)[:300]}")
        records = load_jsonl_sample(data_file, MAX_SAMPLE)
        total_in_file = "unknown"
    elif filetype == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(data_file)
        total_in_file = pf.metadata.num_rows
        schema = pf.schema_arrow
        log(f"Parquet 列: {schema.names}")
        records, total_in_file = load_parquet_sample(data_file, MAX_SAMPLE)
    else:
        log(f"ERROR: 不支持的文件类型 {filetype}")
        return None

    log(f"加载记录: {len(records)} (文件总量: {total_in_file})")

    # 采样（如果超过 MAX_SAMPLE，已经是前 MAX_SAMPLE 条——改为随机采样）
    rng = random.Random(42)
    rng.shuffle(records)
    sample = records[:min(MAX_SAMPLE, len(records))]

    # 分桶统计
    has_content = 0
    seen_hashes = set()
    bucket_counts = OrderedDict((k, 0) for k in BUCKETS)

    for obj in sample:
        content = extract_user_content(obj)
        if not content or len(content) < 10:
            continue
        has_content += 1

        h = hashlib.md5(content.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        try:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True
            )
            ids = tok.encode(prompt, add_special_tokens=False)
            tlen = len(ids)
        except Exception:
            continue

        for bname, (lo, hi) in BUCKETS.items():
            if lo <= tlen <= hi:
                bucket_counts[bname] += 1
                break

    total_valid = sum(bucket_counts.values())
    log(f"")
    log(f"--- {ds_name} 结果 (采样 {len(sample)} 条) ---")
    log(f"有内容(去重后): {total_valid}")
    log(f"")

    header = f"{'分桶':<16} {'采样数':>8} {'占比':>8}"
    log(header)
    log("-" * 36)
    ge_768 = 0
    ge_1024 = 0
    for bname, (lo, hi) in BUCKETS.items():
        cnt = bucket_counts[bname]
        pct = 100 * cnt / max(total_valid, 1)
        bar = "#" * min(int(pct / 2), 30)
        log(f"{bname:<16} {cnt:>8} {pct:>7.1f}% {bar}")
        if lo >= 768:
            ge_768 += cnt
        if lo >= 1024:
            ge_1024 += cnt

    log("-" * 36)
    log(f">= 768 合计:  {ge_768:>8} ({100*ge_768/max(total_valid,1):.1f}%)")
    log(f">= 1024 合计: {ge_1024:>8} ({100*ge_1024/max(total_valid,1):.1f}%)")
    log(f"")
    a_cnt = bucket_counts["1024-1152"]
    ab1 = a_cnt + bucket_counts["1153-1280"]
    log(f"关键指标 (采样 {len(sample)} 中):")
    log(f"  A桶(1024-1152):  {a_cnt}")
    log(f"  A+B1(1024-1280): {ab1}")
    log(f"  >= 1024 总计:    {ge_1024}")

    return {
        "name": ds_name,
        "id": ds_id,
        "sample_size": len(sample),
        "total_valid": total_valid,
        "bucket_counts": dict(bucket_counts),
        "ge_768": ge_768,
        "ge_1024": ge_1024,
        "a_bucket": a_cnt,
        "ab1_bucket": ab1,
    }


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    open(LOG_FILE, "w").close()

    log("=" * 60)
    log("数据集预统计 v2")
    log("Infinity-Instruct vs Firefly-train-1.1M")
    log("=" * 60)

    from transformers import AutoTokenizer
    log(f"加载 tokenizer: {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    log(f"vocab_size={tok.vocab_size}")

    datasets_config = [
        ("Infinity-Instruct", {
            "id": "BAAI/Infinity-Instruct",
            "type": "parquet",
            "file_glob": "*.parquet",
            "allow_patterns": ["0625/train-00000-of-00007.parquet"],
        }),
        ("Firefly-train-1.1M", {
            "id": "AI-ModelScope/firefly-train-1.1M",
            "type": "jsonl",
            "file_glob": "*.jsonl",
        }),
    ]

    results = []
    for name, config in datasets_config:
        try:
            r = process_dataset(name, config, tok)
            if r:
                results.append(r)
        except Exception as e:
            log(f"FATAL on {name}: {e}")
            log(traceback.format_exc())

    # 对比汇总
    if len(results) >= 1:
        log("")
        log("=" * 60)
        log("最终对比")
        log("=" * 60)
        if len(results) == 2:
            r0, r1 = results[0], results[1]
            log(f"")
            log(f"{'分桶':<16} {r0['name']:>22} {r1['name']:>22}")
            log("-" * 64)
            for bname in BUCKETS:
                c0 = r0["bucket_counts"].get(bname, 0)
                c1 = r1["bucket_counts"].get(bname, 0)
                log(f"{bname:<16} {c0:>22} {c1:>22}")
            log("-" * 64)
            log(f"{'>= 1024':<16} {r0['ge_1024']:>22} {r1['ge_1024']:>22}")
            log(f"{'A桶(1024-1152)':<16} {r0['a_bucket']:>22} {r1['a_bucket']:>22}")
            log(f"{'A+B1(1024-1280)':<16} {r0['ab1_bucket']:>22} {r1['ab1_bucket']:>22}")
            log(f"{'采样量':<16} {r0['sample_size']:>22} {r1['sample_size']:>22}")

    log("")
    log("完成。")


if __name__ == "__main__":
    main()
