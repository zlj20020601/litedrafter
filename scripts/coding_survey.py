#!/usr/bin/env python3
"""
CodeContests (processed) vs APPS 长度分布预统计

CodeContests: mlfoundations-dev/code_contests_processed (15 shard parquet)
  - 提取 problem 字段作为 user request
  - 按 source 分组（数字编码）

APPS: OmniData/APPS (Hendrycks 格式, 已解压 /tmp/apps_extract/APPS/)
  - 提取 question.txt 作为 user request
  - 按 metadata.json 的 difficulty 分组

统一: Qwen3-8B tokenizer, chat template
  enable_thinking=False, add_generation_prompt=True
分桶: <512, 512-767, 768-1023, 1024-1152, 1153-1280, 1281-1536, 1537-2048, >2048
硬标准: N(1024<=L<=1280) >= 256
"""
import os, sys, json, glob, time, hashlib, traceback
from collections import OrderedDict, Counter
from datetime import datetime

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
CACHE_DIR  = "/root/autodl-tmp/cache"
LOG_DIR    = "/root/autodl-tmp/litedrafter/logs"
LOG_FILE   = os.path.join(LOG_DIR, "coding_survey.log")
RESULT_FILE = os.path.join(LOG_DIR, "coding_survey_result.json")

BUCKETS = OrderedDict([
    ("<512",      (0, 511)),
    ("512-767",   (512, 767)),
    ("768-1023",  (768, 1023)),
    ("1024-1152", (1024, 1152)),
    ("1153-1280", (1153, 1280)),
    ("1281-1536", (1281, 1536)),
    ("1537-2048", (1537, 2048)),
    (">2048",     (2049, 999999)),
])

# CodeContests source 数字编码（deepmind/code_contests）
CC_SOURCE_MAP = {
    "0": "unknown", "1": "codeforces", "2": "atcoder",
    "3": "kattis", "4": "codechef", "5": "hackerrank", "6": "hackerearth",
}

def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def bucket_of(tlen):
    for bname, (lo, hi) in BUCKETS.items():
        if lo <= tlen <= hi:
            return bname
    return ">2048"


def render_and_len(tok, content):
    """套 chat template 并返回 token 长度"""
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(tok.encode(prompt, add_special_tokens=False))


# ── CodeContests ──
def survey_code_contests(tok):
    log("=" * 60)
    log("CodeContests: mlfoundations-dev/code_contests_processed")
    log("=" * 60)

    # 下载全部 15 shard（已有 1 个）
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(
        "mlfoundations-dev/code_contests_processed",
        repo_type="dataset", cache_dir=CACHE_DIR,
        allow_patterns=["data/*.parquet"],
    )
    log("download complete")

    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(
        CACHE_DIR,
        "datasets/mlfoundations-dev--code_contests_processed/**/data/train-*.parquet",
    ), recursive=True))
    log(f"shards: {len(files)}")

    # 收集 (problem_text, source) 去重
    problems = {}  # problem_id -> (text, source)
    total_rows = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        nrows = pf.metadata.num_rows
        total_rows += nrows
        for batch in pf.iter_batches(batch_size=5000):
            d = batch.to_pydict()
            n = len(next(iter(d.values())))
            for i in range(n):
                pid = d["problem_id"][i]
                if isinstance(pid, list):
                    pid = tuple(str(x) for x in pid) if pid else None
                problem = d["problem"][i]
                if isinstance(problem, list):
                    problem = " ".join(str(x) for x in problem if x)
                problem = str(problem).strip()
                source = str(d["source"][i] or "")
                if problem and len(problem) >= 10:
                    problems[pid] = (problem, source)
        log(f"  {os.path.basename(fp)}: {nrows} rows, cumulative unique problems: {len(problems)}")

    log(f"total rows: {total_rows}, unique problems: {len(problems)}")
    if not problems:
        log("ERROR: 没有提取到 problem")
        return None

    # 分桶统计（整体 + 按 source）
    overall = Counter()
    by_source = {}
    lengths = []
    for pid, (text, source) in problems.items():
        tlen = render_and_len(tok, text)
        lengths.append(tlen)
        b = bucket_of(tlen)
        overall[b] += 1
        if source not in by_source:
            by_source[source] = Counter()
        by_source[source][b] += 1

    log("")
    log(f"{'分桶':<16} {'条数':>8} {'占比':>8}")
    log("-" * 36)
    ge_768 = 0; ge_1024 = 0; a_plus_b1 = 0
    for bname in BUCKETS:
        cnt = overall[bname]
        pct = 100 * cnt / max(len(problems), 1)
        log(f"{bname:<16} {cnt:>8} {pct:>7.1f}%")
        if bname in ("768-1023", "1024-1152", "1153-1280", "1281-1536", "1537-2048", ">2048"):
            ge_768 += cnt
        if bname in ("1024-1152", "1153-1280", "1281-1536", "1537-2048", ">2048"):
            ge_1024 += cnt
        if bname in ("1024-1152", "1153-1280"):
            a_plus_b1 += cnt

    log("")
    log(f">= 768: {ge_768}, >= 1024: {ge_1024}, A+B1(1024-1280): {a_plus_b1}")
    log("")

    # 按 source 分组
    log("按 source 分组:")
    for src in sorted(by_source.keys()):
        c = by_source[src]
        total_src = sum(c.values())
        in_ab = c["1024-1152"] + c["1153-1280"]
        src_name = CC_SOURCE_MAP.get(src, f"src={src}")
        log(f"  {src_name:<12} (n={total_src:>6}): A+B1={in_ab:>4}, >=1024={sum(c[b] for b in BUCKETS if b not in ('<512','512-767','768-1023')):>4}")

    if lengths:
        lengths.sort()
        log("")
        log(f"长度: min={lengths[0]}, max={lengths[-1]}, "
            f"P50={lengths[len(lengths)//2]}, P75={lengths[3*len(lengths)//4]}, "
            f"P90={lengths[int(len(lengths)*0.9)]}")

    return {
        "name": "CodeContests",
        "total": len(problems),
        "overall": dict(overall),
        "ge_768": ge_768, "ge_1024": ge_1024, "a_plus_b1": a_plus_b1,
        "by_source": {k: dict(v) for k, v in by_source.items()},
    }


# ── APPS ──
def survey_apps(tok):
    log("")
    log("=" * 60)
    log("APPS: OmniData/APPS (Hendrycks 格式)")
    log("=" * 60)

    root = "/tmp/apps_extract/APPS/APPS"
    if not os.path.isdir(root):
        log(f"ERROR: {root} 不存在")
        return None

    problems = {}  # problem_id -> (text, difficulty)
    for split in ("train", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            log(f"  skip {split}: not found")
            continue
        dirs = sorted(os.listdir(split_dir))
        log(f"  {split}: {len(dirs)} problems")
        for pid_dir in dirs:
            pdir = os.path.join(split_dir, pid_dir)
            qfile = os.path.join(pdir, "question.txt")
            mfile = os.path.join(pdir, "metadata.json")
            if not os.path.isfile(qfile):
                continue
            with open(qfile, encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
            if len(text) < 10:
                continue
            diff = "unknown"
            if os.path.isfile(mfile):
                try:
                    with open(mfile, encoding="utf-8") as f:
                        md = json.load(f)
                    diff = md.get("difficulty", "unknown")
                except Exception:
                    pass
            # problem_id = split + pid_dir
            problems[f"{split}/{pid_dir}"] = (text, diff)

    log(f"unique problems: {len(problems)}")
    if not problems:
        return None

    overall = Counter()
    by_diff = {}
    lengths = []
    for pid, (text, diff) in problems.items():
        tlen = render_and_len(tok, text)
        lengths.append(tlen)
        b = bucket_of(tlen)
        overall[b] += 1
        if diff not in by_diff:
            by_diff[diff] = Counter()
        by_diff[diff][b] += 1

    log("")
    log(f"{'分桶':<16} {'条数':>8} {'占比':>8}")
    log("-" * 36)
    ge_768 = 0; ge_1024 = 0; a_plus_b1 = 0
    for bname in BUCKETS:
        cnt = overall[bname]
        pct = 100 * cnt / max(len(problems), 1)
        log(f"{bname:<16} {cnt:>8} {pct:>7.1f}%")
        if bname in ("768-1023", "1024-1152", "1153-1280", "1281-1536", "1537-2048", ">2048"):
            ge_768 += cnt
        if bname in ("1024-1152", "1153-1280", "1281-1536", "1537-2048", ">2048"):
            ge_1024 += cnt
        if bname in ("1024-1152", "1153-1280"):
            a_plus_b1 += cnt

    log("")
    log(f">= 768: {ge_768}, >= 1024: {ge_1024}, A+B1(1024-1280): {a_plus_b1}")
    log("")

    log("按 difficulty 分组:")
    for diff in sorted(by_diff.keys()):
        c = by_diff[diff]
        total_d = sum(c.values())
        in_ab = c["1024-1152"] + c["1153-1280"]
        ge_1024_d = sum(c[b] for b in BUCKETS if b not in ("<512", "512-767", "768-1023"))
        log(f"  {diff:<14} (n={total_d:>6}): A+B1={in_ab:>4}, >=1024={ge_1024_d:>4}")

    if lengths:
        lengths.sort()
        log("")
        log(f"长度: min={lengths[0]}, max={lengths[-1]}, "
            f"P50={lengths[len(lengths)//2]}, P75={lengths[3*len(lengths)//4]}, "
            f"P90={lengths[int(len(lengths)*0.9)]}")

    return {
        "name": "APPS",
        "total": len(problems),
        "overall": dict(overall),
        "ge_768": ge_768, "ge_1024": ge_1024, "a_plus_b1": a_plus_b1,
        "by_diff": {k: dict(v) for k, v in by_diff.items()},
    }


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    open(LOG_FILE, "w").close()

    log("=" * 60)
    log("CodeContests vs APPS 长度分布预统计")
    log("=" * 60)

    from transformers import AutoTokenizer
    log(f"加载 tokenizer: {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    log(f"vocab_size={tok.vocab_size}")

    results = {}
    try:
        results["CodeContests"] = survey_code_contests(tok)
    except Exception as e:
        log(f"CodeContests FATAL: {e}")
        log(traceback.format_exc())

    try:
        results["APPS"] = survey_apps(tok)
    except Exception as e:
        log(f"APPS FATAL: {e}")
        log(traceback.format_exc())

    # 汇总
    log("")
    log("=" * 60)
    log("汇总对比")
    log("=" * 60)
    if "CodeContests" in results and results["CodeContests"]:
        r = results["CodeContests"]
        log(f"CodeContests: n={r['total']}, A+B1={r['a_plus_b1']}, >=1024={r['ge_1024']}")
        verdict = "满足 N(1024-1280)>=256" if r["a_plus_b1"] >= 256 else "不满足"
        log(f"  硬标准 {verdict}")
    if "APPS" in results and results["APPS"]:
        r = results["APPS"]
        log(f"APPS: n={r['total']}, A+B1={r['a_plus_b1']}, >=1024={r['ge_1024']}")
        verdict = "满足 N(1024-1280)>=256" if r["a_plus_b1"] >= 256 else "不满足"
        log(f"  硬标准 {verdict}")

    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"\n结果已保存: {RESULT_FILE}")
    log("完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
