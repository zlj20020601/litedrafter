#!/usr/bin/env python3
"""
数据集预统计：Infinity-Instruct vs Firefly-train-1.1M
目标：统计哪个数据集能提供足够多"天然接近 1024 token"的中文请求。

分桶：
  <768        | 太短，不适用于 1024 目标
  768-1023    | 略短
  1024-1152   | A 桶（最理想，几乎不截断）
  1153-1280   | B1 桶（小幅截断）
  1281-1536   | B2 桶（中度截断）
  1537-2048   | C 桶（较大截断）
  >2048       | D 桶（大幅截断）

规则：
  - 只使用 instruction + 非空 input（或等价的 user message）
  - 套 Qwen3-8B chat template (add_generation_prompt=True)
  - tokenize 后统计长度
  - 大数据集采样统计
"""
import os, sys, json, hashlib, traceback
from collections import OrderedDict

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
CACHE_DIR  = "/root/autodl-tmp/cache"
LOG_DIR    = "/root/autodl-tmp/litedrafter/logs"
LOG_FILE   = os.path.join(LOG_DIR, "dataset_survey.log")

MAX_SAMPLE = 100000  # 每个数据集最多采样 10 万条

# 分桶定义（有序）
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
    """
    从一条数据中提取 user message 内容（instruction + 非空 input）。
    自动识别不同数据集的字段格式。
    返回 content 字符串，如果没有非空 input 则返回 None。
    """
    # 格式1: instruction + input + output (COIG/BELLE/Firefly 格式)
    inst = (obj.get("instruction") or "").strip()
    inp  = (obj.get("input") or "").strip()
    if inst and inp:
        return f"{inst}\n{inp}"

    # 格式2: conversations 字段 (ShareGPT/Infinity-Instruct 格式)
    convs = obj.get("conversations") or obj.get("conversation") or []
    if isinstance(convs, list) and len(convs) >= 1:
        # 找第一个 human/user turn
        for turn in convs:
            if isinstance(turn, dict):
                role = (turn.get("from") or turn.get("role") or "").lower()
                val  = (turn.get("value") or turn.get("content") or "").strip()
                if role in ("human", "user") and val:
                    # 对 conversations 格式，直接用整个 user turn 作为 content
                    return val

    # 格式3: messages 字段
    msgs = obj.get("messages") or []
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                role = (m.get("role") or "").lower()
                content = (m.get("content") or "").strip()
                if role == "user" and content:
                    return content

    # 格式4: 只有 input 字段（某些数据集）
    if inp and not inst:
        return inp

    return None


def load_dataset_lines(ds_id, cache_dir):
    """用 modelscope snapshot_download 下载数据集，返回 (list_of_jsonl_lines, source_file)。"""
    from modelscope.hub.snapshot_download import snapshot_download
    import glob

    local_dir = snapshot_download(ds_id, repo_type="dataset", cache_dir=cache_dir)

    # 找最大的 jsonl 文件（通常是主数据文件）
    jsonl_files = sorted(
        glob.glob(os.path.join(local_dir, "**", "*.jsonl"), recursive=True),
        key=lambda f: os.path.getsize(f),
        reverse=True
    )
    if not jsonl_files:
        json_files = sorted(
            glob.glob(os.path.join(local_dir, "**", "*.json"), recursive=True),
            key=lambda f: os.path.getsize(f),
            reverse=True
        )
        jsonl_files = json_files

    return local_dir, jsonl_files


def process_dataset(ds_name, ds_ids, tok):
    """处理一个数据集：下载、解析、tokenize、分桶统计。"""
    log(f"{'='*60}")
    log(f"处理数据集: {ds_name}")
    log(f"候选 ID: {ds_ids}")
    log(f"{'='*60}")

    # 尝试加载数据集
    local_dir = None
    data_files = None
    used_id = None

    for ds_id in ds_ids:
        try:
            log(f"尝试加载 {ds_id} ...")
            local_dir, data_files = load_dataset_lines(ds_id, CACHE_DIR)
            used_id = ds_id
            log(f"成功: {ds_id}")
            log(f"  local_dir: {local_dir}")
            log(f"  数据文件 ({len(data_files)} 个):")
            for df in data_files[:5]:
                log(f"    {os.path.relpath(df, local_dir)} ({os.path.getsize(df)//1024}KB)")
            break
        except Exception as e:
            log(f"失败: {str(e)[:150]}")

    if not data_files:
        log(f"ERROR: {ds_name} 所有候选 ID 都失败")
        return None

    # 读取所有记录
    all_records = []
    for fpath in data_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        all_records.append(obj)
                except Exception:
                    continue
        # 只读第一个（最大的）数据文件，避免重复
        break

    total_rows = len(all_records)
    log(f"总记录数: {total_rows}")

    # 采样
    import random
    rng = random.Random(42)
    if total_rows > MAX_SAMPLE:
        rng.shuffle(all_records)
        sample = all_records[:MAX_SAMPLE]
        log(f"采样: {MAX_SAMPLE}/{total_rows} (seed=42)")
    else:
        sample = all_records
        log(f"全量使用: {total_rows}")

    # 提取 user content + tokenize + 分桶
    has_content = 0
    has_nonempty_input = 0
    seen_hashes = set()
    bucket_counts = OrderedDict((k, 0) for k in BUCKETS)
    lengths_in_range = []  # 记录 >= 768 的长度（用于详细统计）
    sample_rate = len(sample) / max(total_rows, 1)

    for obj in sample:
        content = extract_user_content(obj)
        if not content or len(content) < 10:
            continue
        has_content += 1

        # 去重
        h = hashlib.md5(content.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # chat template + tokenize
        try:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True
            )
            ids = tok.encode(prompt, add_special_tokens=False)
            tlen = len(ids)
        except Exception:
            continue

        # 分桶
        for bname, (lo, hi) in BUCKETS.items():
            if lo <= tlen <= hi:
                bucket_counts[bname] += 1
                if tlen >= 768:
                    lengths_in_range.append(tlen)
                break

    # 外推到全集
    log(f"")
    log(f"--- {ds_name} 统计结果 ---")
    log(f"采样量: {len(sample)} / 全集: {total_rows} (采样率: {sample_rate:.1%})")
    log(f"去重后有内容: {has_content}")
    log(f"")

    header = f"{'分桶':<16} {'采样数':>8} {'外推全集':>10} {'占比':>8}"
    log(header)
    log("-" * 48)
    total_ge_768_sample = 0
    total_ge_1024_sample = 0
    for bname, (lo, hi) in BUCKETS.items():
        cnt = bucket_counts[bname]
        est_full = int(cnt / sample_rate) if sample_rate > 0 else cnt
        pct = 100 * cnt / max(sum(bucket_counts.values()), 1)
        log(f"{bname:<16} {cnt:>8} {est_full:>10} {pct:>7.1f}%")
        if lo >= 768:
            total_ge_768_sample += cnt
        if lo >= 1024:
            total_ge_1024_sample += cnt

    log("-" * 48)
    est_768 = int(total_ge_768_sample / sample_rate) if sample_rate > 0 else total_ge_768_sample
    est_1024 = int(total_ge_1024_sample / sample_rate) if sample_rate > 0 else total_ge_1024_sample
    log(f">= 768 合计:     {total_ge_768_sample:>8} {est_768:>10}")
    log(f">= 1024 合计:    {total_ge_1024_sample:>8} {est_1024:>10}")

    # A桶详细
    a_cnt = bucket_counts["1024-1152"]
    a_est = int(a_cnt / sample_rate) if sample_rate > 0 else a_cnt
    log(f"")
    log(f"关键指标:")
    log(f"  A桶(1024-1152) 采样数: {a_cnt}, 外推全集: {a_est}")
    log(f"  A+B1(1024-1280) 采样数: {a_cnt + bucket_counts['1153-1280']}")

    return {
        "name": ds_name,
        "id": used_id,
        "total_rows": total_rows,
        "sample_size": len(sample),
        "sample_rate": sample_rate,
        "bucket_counts": dict(bucket_counts),
        "total_ge_1024_sample": total_ge_1024_sample,
        "total_ge_1024_est": est_1024,
    }


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    open(LOG_FILE, "w").close()

    log("=" * 60)
    log("数据集预统计开始")
    log("Infinity-Instruct vs Firefly-train-1.1M")
    log("=" * 60)

    # 加载 tokenizer
    from transformers import AutoTokenizer
    log(f"加载 tokenizer: {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    log(f"vocab_size={tok.vocab_size}")

    # 候选 ID 列表（按优先级）
    datasets = [
        ("Infinity-Instruct", [
            "BAAI/Infinity-Instruct-0625",
            "BAAI/Infinity-Instruct-3M-0625",
            "m-a-p/Infinity-Instruct-0625",
            "AI-ModelScope/Infinity-Instruct-0625",
        ]),
        ("Firefly-train-1.1M", [
            "YeungNLP/firefly-train-1.1M",
            "AI-ModelScope/firefly-train-1.1M",
            "YeungNLP/Firefly-train-1.1M",
        ]),
    ]

    results = []
    for name, ids in datasets:
        try:
            r = process_dataset(name, ids, tok)
            if r:
                results.append(r)
        except Exception as e:
            log(f"FATAL on {name}: {e}")
            log(traceback.format_exc())

    # 汇总对比
    log("")
    log("=" * 60)
    log("最终对比汇总")
    log("=" * 60)
    if len(results) == 2:
        r0, r1 = results[0], results[1]
        log(f"")
        log(f"{'分桶':<16} {r0['name']:>20} vs {r1['name']:>20}")
        log("-" * 62)
        for bname in BUCKETS:
            c0 = r0["bucket_counts"].get(bname, 0)
            c1 = r1["bucket_counts"].get(bname, 0)
            e0 = int(c0 / r0["sample_rate"]) if r0["sample_rate"] > 0 else c0
            e1 = int(c1 / r1["sample_rate"]) if r1["sample_rate"] > 0 else c1
            log(f"{bname:<16} {c0:>8} (全集~{e0:>8}) | {c1:>8} (全集~{e1:>8})")
        log("-" * 62)
        log(f"{'>= 1024 合计':<16} "
            f"{r0['total_ge_1024_sample']:>8} (全集~{r0['total_ge_1024_est']:>8}) | "
            f"{r1['total_ge_1024_sample']:>8} (全集~{r1['total_ge_1024_est']:>8})")

    log("")
    log("完成。")


if __name__ == "__main__":
    main()
