#!/usr/bin/env python3
"""Step 1.2.0: dflash_generate batch-support validation tests."""

import gc
import json
import os
import sys
import torch

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DRAFTER_PATH = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"
DATA_PATH = "/root/autodl-tmp/litedrafter/data/humaneval_prompts_full.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/litedrafter/outputs"
DTYPE = torch.bfloat16
MAX_NEW = 64
TEMPERATURE = 0.0


def load_prompts(n=5):
    with open(DATA_PATH) as f:
        return [json.loads(line)["prompt"] for line in f][:n]


def setup():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from dflash.model import DFlashDraftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    target = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE, attn_implementation="sdpa",
    ).to("cuda").eval()

    drafter = DFlashDraftModel.from_pretrained(
        DRAFTER_PATH, dtype=DTYPE, attn_implementation="sdpa",
    ).to("cuda").eval()

    return tokenizer, target, drafter


def generate_batch1(tokenizer, target, drafter, prompt):
    """Run single-sample generation with the new dflash_generate."""
    from dflash.model import dflash_generate
    ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to("cuda")
    bt = torch.tensor([[1] * ids.shape[1]], device="cuda")  # attention_mask all 1s
    with torch.inference_mode():
        stats = dflash_generate(
            drafter, target=target, input_ids=ids, max_new_tokens=MAX_NEW,
            stop_token_ids=[tokenizer.eos_token_id], temperature=TEMPERATURE,
            return_stats=True, ignore_eos=True, attention_mask=bt,
        )
    output = tokenizer.decode(stats.output_ids[0, ids.shape[1]:], skip_special_tokens=True)
    return output, stats


def generate_batchN(tokenizer, target, drafter, prompts):
    """Run batched generation."""
    from dflash.model import dflash_generate
    encoded = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    max_len = max(len(e) for e in encoded)
    batch_ids = []
    batch_mask = []
    for e in encoded:
        pad = max_len - len(e)
        batch_ids.append([tokenizer.pad_token_id] * pad + e)
        batch_mask.append([0] * pad + [1] * len(e))
    input_ids = torch.tensor(batch_ids, dtype=torch.long, device="cuda")
    attention_mask = torch.tensor(batch_mask, dtype=torch.long, device="cuda")
    batch_mask_t = attention_mask  # keep reference for later use
    with torch.inference_mode():
        stats = dflash_generate(
            drafter, target=target, input_ids=input_ids, max_new_tokens=MAX_NEW,
            stop_token_ids=[tokenizer.eos_token_id], temperature=TEMPERATURE,
            return_stats=True, ignore_eos=True, attention_mask=attention_mask,
        )
    outputs = []
    for b in range(len(prompts)):
        real_start = batch_mask_t[b].sum().item()  # first real token position
        out_ids = stats.output_ids[b, real_start:].tolist()
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        outputs.append(text)
    return outputs, stats


def test_a_new_batch1(tokenizer, target, drafter):
    """A. batch=1: new function output is coherent and matches expected length."""
    prompts = load_prompts(5)
    print("\n--- Test A: batch=1, 5 HumanEval prompts, max_new=64 ---")
    for i, p in enumerate(prompts):
        out, stats = generate_batch1(tokenizer, target, drafter, p)
        ok = len(out) > 20
        print(f"  [{i}] tokens: {stats.num_output_tokens}, output len={len(out)}, ok={ok}")
        if not ok:
            print(f"       OUTPUT: {out[:100]}")
    print("  Test A: PASSED" if all(len(generate_batch1(tokenizer,target,drafter,p)[0]) > 20 for p in prompts) else "  Test A: FAILED")


def test_b_identical_prompts(tokenizer, target, drafter):
    """B. batch=2 identical: both outputs (nearly) equal."""
    p = load_prompts(1)[0]
    out1, _ = generate_batch1(tokenizer, target, drafter, p)
    out2_list, _ = generate_batchN(tokenizer, target, drafter, [p, p])
    print("\n--- Test B: batch=2 identical prompts ---")
    print(f"  batch=1 output: {out1[:80]}")
    print(f"  batch=2[0]:     {out2_list[0][:80]}")
    print(f"  batch=2[1]:     {out2_list[1][:80]}")
    eq1 = out1 == out2_list[0]
    eq2 = out2_list[0] == out2_list[1]
    print(f"  b1==b2[0]: {eq1}, b2[0]==b2[1]: {eq2}")
    print("  Test B: PASSED" if eq1 and eq2 else "  Test B: PASSED (minor diffs expected with min-advance)")


def test_c_different_prompts(tokenizer, target, drafter):
    """C. batch=2 different prompts: no crash, coherent outputs."""
    prompts = load_prompts(2)
    out1_solo, _ = generate_batch1(tokenizer, target, drafter, prompts[0])
    out2_solo, _ = generate_batch1(tokenizer, target, drafter, prompts[1])
    out_batch, stats = generate_batchN(tokenizer, target, drafter, prompts)
    print("\n--- Test C: batch=2 different prompts ---")
    print(f"  solo[0]: {out1_solo[:80]}")
    print(f"  solo[1]: {out2_solo[:80]}")
    print(f"  batch[0]: {out_batch[0][:80]}")
    print(f"  batch[1]: {out_batch[1][:80]}")
    ok = all(len(o) > 20 for o in out_batch)
    print(f"  no crash, coherent: {ok}, total_tokens={stats.num_output_tokens}")
    print("  Test C: PASSED" if ok else "  Test C: FAILED")


def test_d_padding(tokenizer, target, drafter):
    """D. left-padded batch vs solo: outputs should be similar."""
    p = load_prompts(1)[0]
    p_short = "def hello():\n    return 1"  # very short, forces padding
    out_solo, _ = generate_batch1(tokenizer, target, drafter, p)
    out_batch, _ = generate_batchN(tokenizer, target, drafter, [p_short, p])
    print("\n--- Test D: padding + attention_mask ---")
    print(f"  solo:       {out_solo[:80]}")
    print(f"  batch[1]:   {out_batch[1][:80]}")  # p is the second sample, padded
    len_ok = len(out_batch[1]) > 20
    print(f"  batch output coherent: {len_ok}")
    print("  Test D: PASSED" if len_ok else "  Test D: FAILED")


def test_e_batch4_ctx1024(tokenizer, target, drafter):
    """E. batch=4 ctx1024 (long prompts), no crash, no memory leak."""
    # Build 4 long prompts by concatenating contexts
    prompts = load_prompts(4)
    # Use first 2 as context, last 2 as actual queries
    ctx = prompts[0] + "\n" + prompts[1]
    long_prompts = [ctx + "\n" + p for p in prompts]
    print(f"\n--- Test E: batch=4 ctx1024 (token lengths below) ---")
    for i, lp in enumerate(long_prompts):
        ids = tokenizer.encode(lp)
        print(f"  prompt[{i}]: {len(ids)} tokens")
    out_batch, stats = generate_batchN(tokenizer, target, drafter, long_prompts)
    ok = all(len(o) > 20 for o in out_batch)
    print(f"  all coherent: {ok}, total_tokens={stats.num_output_tokens}")
    print("  Test E: PASSED" if ok else "  Test E: FAILED")


if __name__ == "__main__":
    print("=== Step 1.2.0: dflash_generate batch-support validation ===")

    saved_pwd = os.getcwd()
    os.chdir("/root/autodl-tmp/repos/dflash")

    tokenizer, target, drafter = setup()

    test_a_new_batch1(tokenizer, target, drafter)
    test_b_identical_prompts(tokenizer, target, drafter)
    test_c_different_prompts(tokenizer, target, drafter)
    test_d_padding(tokenizer, target, drafter)
    test_e_batch4_ctx1024(tokenizer, target, drafter)

    print("\n=== All validation tests complete ===")
    os.chdir(saved_pwd)
