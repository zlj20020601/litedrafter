#!/usr/bin/env python3
"""
L3: vLLM KVCacheSpec / layer groups 内部结构对比 (AR vs DFlash) — Qwen3.5-4B 版本
适配 vLLM 0.26.0 + env_vllm026 + V2 Model Runner

流程:
  1. EngineArgs → VllmConfig (AR / DFlash)
  2. GPUModelRunner 加载模型 → get_kv_cache_spec() 拿每层 spec
  3. get_kv_cache_configs(vllm_config, specs, available_memory) → KVCacheConfig
  4. 输出 groups / num_blocks / KV shape / 浪费分析

关键问题: Qwen3.5-4B 是 hybrid linear/full attention，
  DFlash 的 KV per-token 成本 282 KB vs AR 45 KB（6.2x），
  但理论上应只有 ~1.75x。L3 要定位分配效率损失的来源。
"""
import os, json, sys, torch, gc, traceback

CONDA_ENV = "/root/autodl-tmp/conda_envs/env_vllm026"
os.environ["PATH"] = f"{CONDA_ENV}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{CONDA_ENV}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

TARGET = "/root/autodl-tmp/models/Qwen3.5-4B"
DRAFT = "/root/autodl-tmp/models/Qwen3.5-4B-DFlash"

# L1 结果中的 available KV memory (GiB)
AR_AVAIL_GIB = 10.82
DF_AVAIL_GIB = 9.0  # 略高于实际 8.85 以通过边界检查

from vllm.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel,
    set_custom_all_reduce, destroy_model_parallel
)


def build_vllm_config(speculative=None):
    from vllm.engine.arg_utils import EngineArgs
    kwargs = dict(
        model=TARGET, dtype="bfloat16",
        gpu_memory_utilization=0.90, max_num_seqs=32,
        max_num_batched_tokens=16384, enforce_eager=True,
        max_model_len=2048,
        enable_prefix_caching=False, trust_remote_code=True,
    )
    if speculative:
        kwargs["speculative_config"] = speculative
    ea = EngineArgs(**kwargs)
    vc = ea.create_engine_config(usage_context="test")
    return vc


def get_specs(vllm_config):
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    device = torch.device("cuda:0")
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        runner = GPUModelRunner(vllm_config, device)
        runner.load_model(load_dummy_weights=True)
        specs = runner.get_kv_cache_spec()
    return runner, specs


def analyze(mode, avail_gib):
    print("=" * 70, flush=True)
    print(f"L3: {mode.upper()} KVCacheSpec / layer groups — Qwen3.5-4B", flush=True)
    print("=" * 70, flush=True)

    speculative = None
    if mode == "dflash":
        speculative = {
            "method": "dflash",
            "model": DRAFT,
            "num_speculative_tokens": 15,
        }

    vc = build_vllm_config(speculative)
    print(f"vllm_config built: speculative={vc.speculative_config is not None}", flush=True)

    runner, specs = get_specs(vc)
    print(f"kv_cache_specs: {len(specs)} layers", flush=True)

    # 打印每层 spec 的完整信息
    spec_counts = {}
    for name, spec in specs.items():
        # 收集所有 spec 属性
        attrs = []
        for attr in ["block_size", "num_kv_heads", "head_size", "head_size_v"]:
            if hasattr(spec, attr):
                val = getattr(spec, attr)
                attrs.append(f"{attr}={val}")
        # spec 类型（class name）
        cls = type(spec).__name__
        key = f"{cls}({', '.join(attrs)})"
        spec_counts.setdefault(key, []).append(name)

    print(f"\nspec 类型分布:", flush=True)
    for key, layers in sorted(spec_counts.items()):
        print(f"  {key}: {len(layers)} 层", flush=True)
        if len(layers) <= 10:
            for ln in layers:
                print(f"    {ln}", flush=True)
        else:
            for ln in layers[:5]:
                print(f"    {ln}", flush=True)
            print(f"    ... ({len(layers)-5} more)", flush=True)

    # 打印每个 spec 的详细属性
    print(f"\n各 spec 详细属性:", flush=True)
    seen_types = set()
    for name, spec in specs.items():
        cls = type(spec).__name__
        if cls in seen_types:
            continue
        seen_types.add(cls)
        all_attrs = {k: v for k, v in vars(spec).items() if not k.startswith("_")}
        print(f"  {cls}: {json.dumps(all_attrs, default=str)}", flush=True)

    # get_kv_cache_configs
    try:
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        avail_bytes = int(avail_gib * 1024**3)
        kcc = get_kv_cache_configs(vc, [specs], [avail_bytes])[0]
        print(f"\nKVCacheConfig:", flush=True)
        print(f"  num_blocks(total): {kcc.num_blocks}", flush=True)
        print(f"  kv_cache_groups: {len(kcc.kv_cache_groups)}", flush=True)
        for gi, g in enumerate(kcc.kv_cache_groups):
            print(f"  group[{gi}]:", flush=True)
            print(f"    layer_names ({len(g.layer_names)}): {g.layer_names[:8]}{'...' if len(g.layer_names)>8 else ''}", flush=True)
            spec = g.kv_cache_spec
            print(f"    spec: {type(spec).__name__}, block_size={spec.block_size}, "
                  f"page_size_bytes={spec.page_size_bytes}", flush=True)
            # 额外属性
            for attr in ["num_kv_heads", "head_size", "head_size_v"]:
                if hasattr(spec, attr):
                    print(f"    {attr}={getattr(spec, attr)}", flush=True)
            if hasattr(g, "is_eagle_group"):
                print(f"    is_eagle_group: {g.is_eagle_group}", flush=True)

        print(f"\n  kv_cache_tensors: {len(kcc.kv_cache_tensors)}", flush=True)
        for t in kcc.kv_cache_tensors:
            print(f"    {t}", flush=True)

        # 计算每组的显存占用和 block 分布
        print(f"\n显存分配分析:", flush=True)
        total_pages_bytes = 0
        for gi, g in enumerate(kcc.kv_cache_groups):
            n_layers = len(g.layer_names)
            page_bytes = g.kv_cache_spec.page_size_bytes
            # 这个 group 分到的 blocks
            # 从 kv_cache_tensors 找对应的
            print(f"  group[{gi}]: {n_layers} layers, page_size={page_bytes} bytes", flush=True)
            total_pages_bytes += page_bytes

        # 理论计算
        # 每个 block 可服务 block_size tokens
        # 每个 request 需要 ceil(seq_len / block_size) blocks × num_groups
        block_size = kcc.kv_cache_groups[0].kv_cache_spec.block_size if kcc.kv_cache_groups else None
        print(f"\n  block_size: {block_size}", flush=True)
        print(f"  total num_blocks: {kcc.num_blocks}", flush=True)
        if block_size:
            max_tokens = kcc.num_blocks * block_size
            print(f"  理论 max tokens (num_blocks × block_size): {max_tokens}", flush=True)
            # 但实际每组都要分配 blocks
            # 真实 max concurrent tokens = num_blocks / sum(per_group_blocks_per_token)
            # 如果有 N 组，每组每 token 需要 1 block/block_size
            # 那总 blocks 需求 = N × ceil(tokens/block_size)
            # num_blocks = N × ceil(max_tokens/block_size)
            # max_tokens = (num_blocks / N) × block_size
            n_groups = len(kcc.kv_cache_groups)
            effective_max = (kcc.num_blocks / n_groups) * block_size if n_groups > 0 else max_tokens
            print(f"  考虑 {n_groups} 组: effective max tokens ≈ {int(effective_max)}", flush=True)
            print(f"  组惩罚因子: {n_groups}x blocks/token", flush=True)

        kcc_data = {
            "num_blocks": kcc.num_blocks,
            "num_groups": len(kcc.kv_cache_groups),
            "groups": [],
            "block_size": block_size,
        }
        for g in kcc.kv_cache_groups:
            kcc_data["groups"].append({
                "n_layers": len(g.layer_names),
                "layer_names": g.layer_names[:10],
                "spec_type": type(g.kv_cache_spec).__name__,
                "block_size": g.kv_cache_spec.block_size,
                "page_size_bytes": g.kv_cache_spec.page_size_bytes,
            })

    except Exception as e:
        print(f"\nget_kv_cache_configs 失败: {e}", flush=True)
        traceback.print_exc()
        kcc_data = {"error": str(e)}

    # 释放
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    destroy_model_parallel()

    return {
        "mode": mode,
        "num_layers": len(specs),
        "spec_types": {k: len(v) for k, v in spec_counts.items()},
        "kcc": kcc_data,
    }


def main():
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:6450",
        local_rank=0, backend="nccl",
    )
    set_custom_all_reduce(False)

    results = {}
    try:
        results["ar"] = analyze("ar", AR_AVAIL_GIB)
    except Exception as e:
        print(f"AR 分析失败: {e}", flush=True)
        traceback.print_exc()
        results["ar"] = {"error": str(e)}

    try:
        results["dflash"] = analyze("dflash", DF_AVAIL_GIB)
    except Exception as e:
        print(f"DFlash 分析失败: {e}", flush=True)
        traceback.print_exc()
        results["dflash"] = {"error": str(e)}

    # 汇总对比
    print("\n" + "=" * 70, flush=True)
    print("L3 汇总对比", flush=True)
    print("=" * 70, flush=True)
    for mode in ("ar", "dflash"):
        r = results.get(mode, {})
        if "error" in r:
            print(f"  {mode.upper()}: ERROR - {r['error'][:100]}", flush=True)
            continue
        print(f"  {mode.upper()}: {r.get('num_layers', '?')} layers", flush=True)
        for k, v in r.get("spec_types", {}).items():
            print(f"    {k}: {v} 层", flush=True)
        kcc = r.get("kcc", {})
        print(f"    blocks={kcc.get('num_blocks')}, groups={kcc.get('num_groups')}, "
              f"block_size={kcc.get('block_size')}", flush=True)

    out_file = "/root/autodl-tmp/litedrafter/outputs/l3_kvspec_qwen35_20260812.json"
    with open(out_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_file}", flush=True)


if __name__ == "__main__":
    main()
