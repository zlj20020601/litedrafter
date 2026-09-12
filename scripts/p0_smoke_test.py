#!/usr/bin/env python3
"""
p0_smoke_test.py - Step 1.0: 环境验证与 Smoke Test

验证项：
1. Import 链完整
2. Qwen3-8B 加载到 GPU，显存合理
3. AR generate 5 tokens
4. DFlash drafter 加载，显存增量合理
5. dflash_generate 可调用
"""

import sys
import time

import torch

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DRAFTER_PATH = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
DTYPE = torch.bfloat16


def mem_mb():
    torch.cuda.synchronize()
    return round(torch.cuda.memory_allocated() / 1048576, 1)


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    # ---- 1. Import 链 ----
    section("1. Import 链验证")
    import transformers
    print(f"  torch:        {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  CUDA avail:   {torch.cuda.is_available()}")
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")

    try:
        import vllm
        print(f"  vllm:         {vllm.__version__}")
    except Exception as e:
        print(f"  vllm:         IMPORT FAILED: {e}")
        sys.exit(1)

    try:
        import bitsandbytes as bnb
        print(f"  bitsandbytes: {bnb.__version__}")
    except Exception as e:
        print(f"  bitsandbytes: IMPORT FAILED: {e}")
        sys.exit(1)

    try:
        from dflash.model import DFlashDraftModel, dflash_generate
        print("  dflash:       OK (DFlashDraftModel, dflash_generate)")
    except Exception as e:
        print(f"  dflash:       IMPORT FAILED: {e}")
        sys.exit(1)

    # ---- 2. 模型加载 ----
    section("2. Qwen3-8B 加载")

    m0 = mem_mb()
    print(f"  [M0] 空 CUDA 进程: {m0:.0f} MB")

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"  Tokenizer loaded")

    target = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    m1 = mem_mb()
    print(f"  [M1] Target 加载后: {m1:.0f} MB (+{m1 - m0:.0f} MB)")

    # ---- 3. AR generate ----
    section("3. AR Generate (5 tokens)")

    prompt = "你好，请用一句话解释什么是投机解码。"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    print(f"  Prompt: {prompt}")
    print(f"  Input tokens: {input_ids.shape[1]}")

    with torch.inference_mode():
        output = target.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
        )
    decoded = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Output (5 tokens): {decoded}")
    print(f"  AR generate: OK")

    # ---- 4. DFlash drafter 加载 ----
    section("4. DFlash Drafter 加载")

    drafter = DFlashDraftModel.from_pretrained(
        DRAFTER_PATH,
        dtype=DTYPE,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    m2 = mem_mb()
    print(f"  [M2] Drafter 加载后: {m2:.0f} MB (+{m2 - m1:.0f} MB)")
    print(f"  block_size: {drafter.block_size}")

    # ---- 5. dflash_generate ----
    section("5. dflash_generate (5 tokens)")

    eos_id = tokenizer.eos_token_id
    with torch.inference_mode():
        stats = dflash_generate(
            drafter,
            target=target,
            input_ids=input_ids,
            max_new_tokens=5,
            stop_token_ids=[eos_id],
            temperature=0.0,
            return_stats=True,
        )
    print(f"  Output tokens: {stats.num_output_tokens}")
    print(f"  Acceptance lengths: {stats.acceptance_lengths}")
    print(f"  dflash_generate: OK")

    # ---- Summary ----
    section("SMOKE TEST SUMMARY")
    print(f"  M0 (empty):          {m0:.0f} MB")
    print(f"  M1 (target loaded):  {m1:.0f} MB  (delta: {m1 - m0:.0f} MB)")
    print(f"  M2 (drafter loaded): {m2:.0f} MB  (delta: {m2 - m1:.0f} MB)")
    print(f"  All checks: PASSED")


if __name__ == "__main__":
    main()
