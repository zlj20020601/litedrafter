#!/usr/bin/env python3
"""
prepare_contexts.py - 构造 HumanEval-164 context 变体

规则:
  - ctx 由 OTHER problems 的 prompt + canonical_solution 拼接（不含当前 task 自己的）
  - ctx 长度按 Qwen tokenizer token 数精确控制
  - 4 个变体: raw, ctx1024, ctx2048, ctx4096
  - 随机种子固定 (42)，可复现
"""

import json
import random
import os
from pathlib import Path

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
SRC = "/root/autodl-tmp/litedrafter/data/humaneval_prompts_full.jsonl"
OUT_DIR = "/root/autodl-tmp/litedrafter/data"
TARGETS = {"ctx1024": 1024, "ctx2048": 2048, "ctx4096": 4096}
SEED = 42


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load all problems
    with open(SRC) as f:
        all_problems = [json.loads(line) for line in f]
    print(f"Loaded {len(all_problems)} problems")

    # Build pool of context chunks: (task_id, "prompt\nsolution")
    pool = []
    for p in all_problems:
        chunk = f"{p['prompt']}\n{p.get('canonical_solution', '')}"
        pool.append({"task_id": p["id"], "chunk": chunk})

    # For each target token count, construct ctx variants
    random.seed(SEED)
    rng = random.Random(SEED)

    for ctx_name, target_tokens in TARGETS.items():
        out_path = os.path.join(OUT_DIR, f"he164_{ctx_name}.jsonl")
        print(f"\nBuilding he164_{ctx_name} (target={target_tokens} tokens)...")

        # Deterministic shuffle for sampling order per problem
        indices = list(range(len(all_problems)))
        rng.shuffle(indices)

        out_lines = []
        for i, prob in enumerate(all_problems):
            my_id = prob["id"]
            my_prompt = prob["prompt"]
            my_tokens = len(tokenizer.encode(my_prompt))

            # Build context from OTHER problems only
            ctx_parts = []
            ctx_tokens = 0
            # Pick from shuffled pool, skip self
            for idx in indices:
                if pool[idx]["task_id"] == my_id:
                    continue
                chunk = pool[idx]["chunk"]
                chunk_tokens = len(tokenizer.encode(chunk))
                if ctx_tokens + chunk_tokens + my_tokens <= target_tokens:
                    ctx_parts.append(chunk)
                    ctx_tokens += chunk_tokens
                # Stop when we can't fit more
                if ctx_tokens + my_tokens >= target_tokens * 0.95:
                    break

            # Assemble: context + "---\n" + prompt
            ctx_text = "\n\n".join(ctx_parts)
            full_text = f"{ctx_text}\n\n---\n\n{my_prompt}"
            full_tokens = tokenizer.encode(full_text)
            token_ids = full_tokens[:target_tokens]  # truncate if overshoot
            final_text = tokenizer.decode(token_ids, skip_special_tokens=True)

            out_lines.append({
                "id": my_id,
                "prompt": final_text,
                "token_len": len(token_ids),
                "ctx_variant": ctx_name,
                "ctx_parts_count": len(ctx_parts),
            })

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(all_problems)}...")

        # Verify lengths
        lens = [d["token_len"] for d in out_lines]
        print(f"  Saved {len(out_lines)} samples")
        print(f"  token_len: min={min(lens)}, p50={sorted(lens)[len(lens)//2]}, max={max(lens)}, avg={sum(lens)/len(lens):.0f}")

        with open(out_path, "w") as f:
            for d in out_lines:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # raw variant: just copy with token_len field
    raw_path = os.path.join(OUT_DIR, "he164_raw.jsonl")
    print(f"\nBuilding he164_raw...")
    with open(raw_path, "w") as f:
        for p in all_problems:
            d = {"id": p["id"], "prompt": p["prompt"],
                 "token_len": len(tokenizer.encode(p["prompt"])),
                 "ctx_variant": "raw", "ctx_parts_count": 0}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  Saved {len(all_problems)} samples")

    print("\n[DONE] All 4 dataset variants created")


if __name__ == "__main__":
    main()
