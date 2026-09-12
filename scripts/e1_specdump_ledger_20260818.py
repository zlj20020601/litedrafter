#!/usr/bin/env python3
"""E1: startup spec dump + synthetic admission ledger (AR vs DFlash, 0816 serving config).

Goals (0817 design doc):
1. Dump per-group spec: type / block_size / page_size_bytes / num_speculative_blocks /
   sliding_window / heads -> decompose the 502 blocks/req mystery.
2. Calibrate avail_bytes so get_kv_cache_capacity reproduces the observed
   AR 249,856 tokens / 61.00x  and  DFlash 32,969 tokens / 8.05x.
3. Build the REAL KVCacheManager (mirror scheduler init) and run synthetic
   admission: N requests of 1024 prompt (+256 growth), lookahead 0 (AR) / 16 (DFlash).
   Record per-group blocks/req at admit(1024) and steady(1280), and the max C admitted
   -> compare with observed ceiling (AR >96, DFlash = 9).

No formula re-implementation: all numbers come from real vLLM code paths.
"""
import os, json, gc, traceback
from math import gcd

CONDA_ENV = "/root/autodl-tmp/conda_envs/env_vllm026"
os.environ["PATH"] = f"{CONDA_ENV}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{CONDA_ENV}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

TARGET = "/root/autodl-tmp/models/Qwen3.5-4B"
DRAFT = "/root/autodl-tmp/models/Qwen3.5-4B-DFlash"
MAX_LEN, MAX_SEQS, MNBT = 4096, 128, 16384
TARGET_TOKENS = {"ar": 249856, "dflash": 32969}
OUT = "/root/autodl-tmp/litedrafter/outputs/e1_specdump_ledger_20260818.json"

from vllm.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel,
    set_custom_all_reduce, destroy_model_parallel,
)


def build_vllm_config(speculative=None):
    from vllm.engine.arg_utils import EngineArgs
    kwargs = dict(
        model=TARGET, dtype="bfloat16",
        gpu_memory_utilization=0.90, max_num_seqs=MAX_SEQS,
        max_num_batched_tokens=MNBT, enforce_eager=True,
        max_model_len=MAX_LEN,
        enable_prefix_caching=False, trust_remote_code=True,
    )
    if speculative:
        kwargs["speculative_config"] = speculative
    return EngineArgs(**kwargs).create_engine_config(usage_context="test")


def get_specs(vllm_config):
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    import torch
    device = torch.device("cuda:0")
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        runner = GPUModelRunner(vllm_config, device)
        runner.load_model(load_dummy_weights=True)
        specs = runner.get_kv_cache_spec()
    return runner, specs


def spec_row(g):
    s = g.kv_cache_spec
    row = {
        "spec_type": type(s).__name__,
        "block_size": s.block_size,
        "page_size_bytes": s.page_size_bytes,
        "n_layers": len(g.layer_names),
        "layers_head": g.layer_names[:3],
    }
    for attr in ("num_speculative_blocks", "sliding_window", "num_kv_heads", "head_size"):
        if hasattr(s, attr):
            row[attr] = getattr(s, attr)
    return row


def analyze(mode):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs, get_kv_cache_capacity
    print("=" * 72, flush=True)
    print(f"E1 {mode.upper()}", flush=True)
    print("=" * 72, flush=True)

    speculative = None
    if mode == "dflash":
        speculative = {"method": "dflash", "model": DRAFT, "num_speculative_tokens": 15}
    vc = build_vllm_config(speculative)
    runner, specs = get_specs(vc)
    print(f"kv_cache_specs: {len(specs)} layers", flush=True)

    # --- fixed avail from 0816 server logs (ground truth) ---
    # AR: "Available KV cache memory: 10.82 GiB", DF: 8.85 GiB
    avail_bytes = int((10.82 if mode == "ar" else 8.85) * 1024**3)
    kcc = get_kv_cache_configs(vc, [specs], [avail_bytes])[0]
    toks, conc = get_kv_cache_capacity(vc, kcc)
    print(f"fixed avail: tokens={toks:,} (observed {TARGET_TOKENS[mode]:,}) max_concurrency={conc:.2f}x (observed {'61.00' if mode=='ar' else '8.05'})", flush=True)

    groups = [spec_row(g) for g in kcc.kv_cache_groups]
    for i, gr in enumerate(groups):
        extra = {k: v for k, v in gr.items() if k not in ("layers_head",)}
        print(f"  group[{i:>2}]: {extra}", flush=True)
    n_blocks = kcc.num_blocks
    print(f"pool num_blocks={n_blocks}, n_groups={len(groups)}", flush=True)

    # --- real KVCacheManager, mirror scheduler init ---
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    sbs = 1
    for gr in groups:
        sbs = sbs * gr["block_size"] // gcd(sbs, gr["block_size"])
    mift = getattr(vc, "max_in_flight_tokens", None)
    mgr = KVCacheManager(
        kv_cache_config=kcc, max_model_len=MAX_LEN,
        scheduler_block_size=sbs, hash_block_size=sbs,
        max_in_flight_tokens=mift, enable_caching=False, use_eagle=False,
        log_stats=False, watermark=0.0,
    )
    lookahead = 0 if mode == "ar" else 15 + 1  # scheduler: dflash = num_spec + 1

    # --- synthetic admission ledger ---
    from vllm.v1.request import Request, RequestStatus
    from vllm.sampling_params import SamplingParams
    reqs, admit_log = [], []
    while len(reqs) < 200:
        rid = f"r{len(reqs)}"
        r = Request(rid, list(range(1024)),
                    SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True),
                    None, arrival_time=0.0)
        nb = mgr.allocate_slots(
            request=r, num_new_tokens=1024, num_lookahead_tokens=lookahead,
            full_sequence_must_fit=True, has_scheduled_reqs=bool(reqs),
        )
        if nb is None:
            admit_log.append({"C_attempt": len(reqs) + 1, "admitted": False,
                              "free_blocks": mgr.block_pool.get_num_free_blocks()})
            break
        per_g = [len(m) for m in nb.blocks] if nb else []
        # grow to 1280 (decode), track per-group held blocks
        r.status = RequestStatus.RUNNING
        r.num_computed_tokens = 1024
        nb2 = mgr.allocate_slots(request=r, num_new_tokens=256,
                                 num_lookahead_tokens=lookahead)
        held = [len(mgr.coordinator.single_type_managers[i].req_to_blocks[rid])
                for i in range(len(groups))]
        admit_log.append({"C": len(reqs) + 1, "admitted": True,
                          "blocks_at_admit": per_g, "held_at_1280": held,
                          "total_held_1280": sum(held),
                          "growth_ok": nb2 is not None})
        reqs.append(r)

    ceiling = len(reqs)
    last = admit_log[-2] if ceiling and admit_log[-1].get("admitted") is False else admit_log[-1]
    print(f"\nledger: admitted C={ceiling}; per-req held@1280 (last) = "
          f"{last.get('total_held_1280')} blocks across {len(groups)} groups", flush=True)
    if last.get("held_at_1280"):
        print(f"  per-group held@1280: {last['held_at_1280']}", flush=True)
    if admit_log[-1].get("admitted") is False:
        print(f"  reject at C={admit_log[-1]['C_attempt']}: free={admit_log[-1]['free_blocks']}", flush=True)

    # cleanup
    del mgr
    del runner
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    destroy_model_parallel()

    return {
        "mode": mode, "n_spec_layers": len(specs),
        "pool_num_blocks": n_blocks, "pool_tokens": toks,
        "max_concurrency_at_maxlen": round(conc, 3),
        "scheduler_block_size": sbs, "lookahead": lookahead,
        "groups": groups,
        "admit_ceiling_C": ceiling,
        "admit_log_tail": admit_log[-3:],
        "per_req_total_1280": last.get("total_held_1280"),
        "per_group_1280": last.get("held_at_1280"),
    }


def main():
    init_distributed_environment(world_size=1, rank=0,
                                 distributed_init_method="tcp://127.0.0.1:6451",
                                 local_rank=0, backend="nccl")
    set_custom_all_reduce(False)
    results = {}
    for mode in ("ar", "dflash"):
        try:
            results[mode] = analyze(mode)
        except Exception as e:
            traceback.print_exc()
            results[mode] = {"error": str(e)}
    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    print(f"\nsaved: {OUT}", flush=True)

    # comparison summary
    print("\n" + "=" * 72)
    print("E1 SUMMARY (observed: AR >96 running @C96 kv .86 | DF ceiling=9, kv .957)")
    print("=" * 72)
    for m in ("ar", "dflash"):
        r = results[m]
        if "error" in r:
            print(f"{m}: ERROR {r['error'][:120]}")
            continue
        print(f"{m:>6}: pool_blocks={r['pool_num_blocks']} tokens={r['pool_tokens']:,} "
              f"conc@4096={r['max_concurrency_at_maxlen']}x | ledger_ceiling={r['admit_ceiling_C']} "
              f"| per-req@1280={r['per_req_total_1280']} blocks over {len(r['groups'])} groups")
    print("validation: ledger_ceiling vs observed 9 / >96; per-req@1280 x blocks/page vs 1.05GB")


if __name__ == "__main__":
    main()
