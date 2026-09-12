#!/usr/bin/env python3
"""
CodeContests benchmark 数据集生成 v2
规格:
  - Dataset: CodeContests (mlfoundations-dev/code_contests_processed)
  - Task: English competitive-programming generation
  - Selection: 249 × [1024,1152] + 7 × [1153,1280], seed=42 → 256
  - Final prompt: exactly 1024 tokens
  - Assistant/reference solution: strictly excluded (只用 problem 字段)
  - Chat template: Qwen3-8B, enable_thinking=False, add_generation_prompt=True

关键预处理:
  - 不在已 render 的序列上硬截断 token
  - 在 content 层面减少 user content → 重新套 chat template
  - 保证 generation prompt 完整 → 迭代微调至 len(input_ids)==1024
"""
import os, sys, json, glob, random, traceback
from collections import OrderedDict, Counter
from datetime import datetime

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
CACHE_DIR  = "/root/autodl-tmp/cache"
LOG_DIR    = "/root/autodl-tmp/litedrafter/logs"
OUTPUT_DIR = "/root/autodl-tmp/litedrafter/data"
LOG_FILE   = os.path.join(LOG_DIR, "codecontests_gen.log")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "codecontests_qwen3_1024_256.jsonl")
META_FILE   = os.path.join(OUTPUT_DIR, "codecontests_qwen3_1024_256.meta.json")

TARGET = 1024
SEED   = 42
A_SIZE = 249      # [1024, 1152]
B1_SIZE = 7       # [1153, 1280]
BUCKETS = OrderedDict([
    ("A",  (1024, 1152)),
    ("B1", (1153, 1280)),
])


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def render(tok, content):
    """套 chat template 并返回 (full_text, len)"""
    full = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok.encode(full, add_special_tokens=False)
    return full, len(ids)


def fit_to_target(tok, content, target=TARGET):
    """
    在 content 层面减少 user content，使 render 后精确 target tokens。
    策略：token 级粗调 → keep 邻域扫描 → 字符级微调。
    返回 (final_content, final_full_text, truncated)
    """
    full, L = render(tok, content)
    if L == target:
        return content, full, False
    if L < target:
        return content, full, False

    content_ids = tok.encode(content, add_special_tokens=False)
    keep = max(1, len(content_ids) - (L - target))

    best = content
    best_full = full
    best_L = L

    # 邻域扫描：在 keep ± 40 内找精确命中
    lo_k = max(1, keep - 40)
    hi_k = min(len(content_ids), keep + 40)
    for k in range(lo_k, hi_k + 1):
        c = tok.decode(content_ids[:k], skip_special_tokens=True).strip()
        if not c:
            continue
        cf, cl = render(tok, c)
        if cl == target:
            return c, cf, True
        if abs(cl - target) < abs(best_L - target):
            best, best_full, best_L = c, cf, cl

    # 字符级微调：在 best 末尾加/减字符
    for d in range(1, 60):
        cands = [best[:-d], best + " ", best + "\n", best + "."]
        for c in cands:
            if not c:
                continue
            cf, cl = render(tok, c)
            if cl == target:
                return c, cf, True
            if abs(cl - target) < abs(best_L - target):
                best, best_full, best_L = c, cf, cl

    # 字符二分兜底：找 render >= target 的最小字符前缀，再在分界附近搜索
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi) // 2
        c = content[:mid].strip()
        if not c:
            lo = mid + 1
            continue
        cl = render(tok, c)[1]
        if cl >= target:
            hi = mid
        else:
            lo = mid + 1
    for k in range(max(0, lo - 4), min(len(content), lo + 5)):
        c = content[:k].strip()
        if not c:
            continue
        cf, cl = render(tok, c)
        if cl == target:
            return c, cf, True
        if abs(cl - target) < abs(best_L - target):
            best, best_full, best_L = c, cf, cl

    return best, best_full, True


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    open(LOG_FILE, "w").close()

    log("=" * 70)
    log("CodeContests benchmark 数据生成 v2")
    log(f"Selection: {A_SIZE}×[1024,1152] + {B1_SIZE}×[1153,1280], seed={SEED}")
    log("=" * 70)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    log(f"tokenizer: vocab_size={tok.vocab_size}")

    # ── 1. 加载 CodeContests ──
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(
        CACHE_DIR,
        "datasets/mlfoundations-dev--code_contests_processed/**/data/train-*.parquet",
    ), recursive=True))
    log(f"shards: {len(files)}")

    problems = {}   # problem_id -> (text, source)
    for fp in files:
        pf = pq.ParquetFile(fp)
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
                src = str(d["source"][i] or "")
                if problem and len(problem) >= 10:
                    problems[pid] = (problem, src)
    log(f"unique problems: {len(problems)}")

    # ── 2. 筛选 A/B1 桶（render 测长度）──
    log("render + 分桶 ...")
    buckets = {"A": [], "B1": []}
    raw_lens = []
    for pid, (text, src) in problems.items():
        _, L = render(tok, text)
        raw_lens.append(L)
        if 1024 <= L <= 1152:
            buckets["A"].append((pid, text, src, L))
        elif 1153 <= L <= 1280:
            buckets["B1"].append((pid, text, src, L))
    log(f"A桶[1024,1152]: {len(buckets['A'])}")
    log(f"B1桶[1153,1280]: {len(buckets['B1'])}")

    # ── 3. 采样：A 全选 249 + B1 抽 7 ──
    rng = random.Random(SEED)
    if len(buckets["A"]) < A_SIZE:
        log(f"WARNING: A 桶不足 {A_SIZE}，实际 {len(buckets['A'])}")
    selected_a = buckets["A"][:A_SIZE]

    rng.shuffle(buckets["B1"])
    selected_b1 = buckets["B1"][:B1_SIZE]
    reserve_b1 = buckets["B1"][B1_SIZE:]   # 备选池（失败时补）
    log(f"选中: A={len(selected_a)}, B1={len(selected_b1)}, 备选B1={len(reserve_b1)}")

    selected = selected_a + selected_b1
    rng.shuffle(selected)   # 打乱顺序分配 sample_id

    # ── 4. content 层面截断 → 精确 1024 ──
    log("content-level truncation → 精确 1024 ...")
    results = []
    stats = {"raw_lt_target": 0, "truncated": 0, "iter_ok": 0}
    pool = list(selected) + list(reserve_b1)   # 主选区 + 备选区
    pool_iter = iter(pool)
    i = 0
    while i < 256:
        try:
            pid, text, src, L_raw = next(pool_iter)
        except StopIteration:
            log(f"FATAL: 备选池耗尽，只生成 {len(results)} 条")
            break

        final_content, final_full, truncated = fit_to_target(tok, text)
        final_ids = tok.encode(final_full, add_special_tokens=False)

        if len(final_ids) != TARGET:
            log(f"  WARN sample {i} (pid={str(pid)[:20]}): len={len(final_ids)}, "
                f"L_raw={L_raw} → 换备选")
            continue

        if truncated:
            stats["truncated"] += 1
        if L_raw < TARGET:
            stats["raw_lt_target"] += 1
        stats["iter_ok"] += 1

        results.append({
            "sample_id": f"codecontests_{i:04d}",
            "problem_id": str(pid),
            "source": src,
            "messages": [{"role": "user", "content": final_content}],
            "prompt_token_length_raw": L_raw,
            "prompt_token_length_used": TARGET,
            "input_ids": final_ids,
            "decoded_prompt": final_full,
            "truncated": truncated,
        })
        i += 1

    log(f"成功生成 {len(results)} 条 (truncated={stats['truncated']}, "
        f"len==1024={stats['iter_ok']})")

    # ── 5. 落盘 ──
    log(f"写入 {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "dataset": "mlfoundations-dev/code_contests_processed",
        "task": "English competitive-programming generation",
        "tokenizer": MODEL_PATH,
        "chat_template": "Qwen3, enable_thinking=False, add_generation_prompt=True",
        "selection": {"A_1024_1152": A_SIZE, "B1_1153_1280": B1_SIZE, "seed": SEED},
        "target_length": TARGET,
        "num_prompts": len(results),
        "source_distribution": dict(Counter(r["source"] for r in results)),
        "truncated_count": stats["truncated"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_file": OUTPUT_FILE,
        "assistant_reference_excluded": True,
        "preprocess_note": "content-level truncation: reduce user content, re-apply chat template, "
                           "generation prompt preserved; NOT hard-truncating rendered sequence",
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"metadata 写入 {META_FILE}")
    log(f"完成。共 {len(results)} 条 × {TARGET} tokens")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
