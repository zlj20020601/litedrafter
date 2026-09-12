#!/usr/bin/env python3
"""
L3: vLLM KVCacheSpec / layer groups 内部结构对比 (AR vs DFlash)
流程:
  1. EngineArgs → VllmConfig (AR / DFlash)
  2. GPUModelRunner 加载模型 → get_kv_cache_spec() 拿每层 spec
  3. get_kv_cache_configs(vllm_config, specs, available_memory) → KVCacheConfig
  4. 输出 groups / num_blocks / KV shape / 浪费分析
"""
import os, json, torch, gc

conda_bin = "/root/autodl-tmp/conda_envs/env_dcut/bin"
os.environ["PATH"] = conda_bin + ":" + os.environ.get("PATH", "")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

TARGET = "/root/autodl-tmp/models/Qwen3-8B"
DRAFT = "/root/autodl-tmp/models/Qwen3-8B-DFlash-b16"

# 初始化分布式环境（模型构建需要 parallel group）
from vllm.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel,
    set_custom_all_reduce
)
init_distributed_environment(
    world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:6450",
    local_rank=0, backend="nccl",
)
set_custom_all_reduce(not False)

# 从 L1 日志拿 available KV memory (GiB)
AR_AVAIL_GIB = 3.92
DF_AVAIL_GIB = 1.02

def build_vllm_config(speculative=None):
    from vllm.engine.arg_utils import EngineArgs
    kwargs = dict(
        model=TARGET, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.90, max_num_seqs=32,
        max_num_batched_tokens=16384, enforce_eager=True,
        enable_prefix_caching=False, trust_remote_code=True,
    )
    if speculative:
        kwargs["speculative_config"] = speculative  # dict, 不要 json.dumps
    ea = EngineArgs(**kwargs)
    vc = ea.create_engine_config(usage_context="test")
    return vc

def get_specs(vllm_config):
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import initialize_model_parallel
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
    print(f"L3: {mode.upper()} KVCacheSpec / layer groups", flush=True)
    print("=" * 70, flush=True)

    speculative = None
    if mode == "dflash":
        speculative = {"method": "dflash", "model": DRAFT,
                       "num_speculative_tokens": 15, "dflash_dcut": 0}

    vc = build_vllm_config(speculative)
    print(f"vllm_config built: speculative={vc.speculative_config is not None}", flush=True)

    runner, specs = get_specs(vc)
    print(f"kv_cache_specs: {len(specs)} layers", flush=True)

    # 打印每层 spec
    spec_counts = {}
    for name, spec in specs.items():
        key = f"block_size={spec.block_size}"
        for attr in ["num_kv_heads", "head_size", "head_size_v"]:
            if hasattr(spec, attr):
                key += f",{attr}={getattr(spec, attr)}"
        spec_counts.setdefault(key, []).append(name)
    print(f"spec 类型分布:", flush=True)
    for key, layers in spec_counts.items():
        print(f"  {key}: {len(layers)} 层 {layers[:8]}{'...' if len(layers)>8 else ''}", flush=True)

    # get_kv_cache_configs
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    avail_bytes = int(avail_gib * 1024**3)
    kcc = get_kv_cache_configs(vc, [specs], [avail_bytes])[0]
    print(f"\nKVCacheConfig:", flush=True)
    print(f"  num_blocks(total): {kcc.num_blocks}", flush=True)
    print(f"  kv_cache_groups: {len(kcc.kv_cache_groups)}", flush=True)
    for gi, g in enumerate(kcc.kv_cache_groups):
        print(f"  group[{gi}]:", flush=True)
        print(f"    layer_names: {g.layer_names[:6]}{'...' if len(g.layer_names)>6 else ''} (共{len(g.layer_names)})", flush=True)
        print(f"    kv_cache_spec: block_size={g.kv_cache_spec.block_size}, "
              f"page_size_bytes={g.kv_cache_spec.page_size_bytes}", flush=True)
        if hasattr(g, "is_eagle_group"):
            print(f"    is_eagle_group: {g.is_eagle_group}", flush=True)
    print(f"  kv_cache_tensors: {len(kcc.kv_cache_tensors)}", flush=True)
    for t in kcc.kv_cache_tensors:
        print(f"    tensor: {t}", flush=True)

    # 释放
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    # 销毁 parallel group，允许下一次 analyze 重新初始化
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    return {"mode": mode, "num_layers": len(specs),
            "spec_types": {k: len(v) for k, v in spec_counts.items()},
            "num_blocks": kcc.num_blocks,
            "groups": [{"n_layers": len(g.layer_names), "block_size": g.kv_cache_spec.block_size,
                        "page_size_bytes": g.kv_cache_spec.page_size_bytes,
                        "first_layers": g.layer_names[:5]}
                       for g in kcc.kv_cache_groups]}

results = {}
results["ar"] = analyze("ar", AR_AVAIL_GIB)
results["dflash"] = analyze("dflash", DF_AVAIL_GIB)

with open("/root/autodl-tmp/litedrafter/outputs/l3_kvspec_20260810.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n结果已保存: /root/autodl-tmp/litedrafter/outputs/l3_kvspec_20260810.json", flush=True)
