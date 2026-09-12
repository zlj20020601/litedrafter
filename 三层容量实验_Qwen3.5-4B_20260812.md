# LiteDrafter 三层容量实验 — Qwen3.5-4B（2026-08-12）

> **做了什么：** 用 CodeContests 256×1024 benchmark 数据（Qwen3.5-4B tokenizer 重新生成），在 vLLM 0.26.0 + env_vllm026 下完成三层容量实验。发现 DFlash 在 hybrid linear/full attention 模型上的容量惩罚主因不是静态显存（与 Qwen3-8B 结论不同），而是 **vLLM KVCacheSpec group 碎片化**：AR 的 4 组退化为 DFlash 的 38 组（每层一组），导致 KV 容量从 249,856 tokens 崩塌至 32,969 tokens（-86.8%）。

## 固定配置

| 项目                             | 值                                                                                            |
| ------------------------------ | -------------------------------------------------------------------------------------------- |
| GPU                            | RTX 4090 24GB 单卡                                                                             |
| Target                         | Qwen3.5-4B BF16（32层: 24 linear_attn + 8 full_attn, hybrid）                                   |
| Drafter                        | Qwen3.5-4B-DFlash（6层: 5 sliding_attn + 1 full_attn, target_layer_ids=[1,5,9,13,17,21,25,29]） |
| vLLM                           | 0.26.0（env_vllm026, VLLM_USE_V2_MODEL_RUNNER=1）                                              |
| max_model_len                  | 4096                                                                                         |
| gpu_memory_utilization         | 0.90                                                                                         |
| max_num_batched_tokens         | 16384                                                                                        |
| max_num_seqs                   | 32                                                                                           |
| num_speculative_tokens         | 15                                                                                           |
| prefix_caching / enforce_eager | false / true                                                                                 |
| workload                       | CodeContests 256 条 × 1024 tokens（Qwen3.5-4B tokenizer），output 256                            |
| 每请求预算                          | 1024 + 256 = 1280 tokens                                                                     |

## L1：batch capacity penalty 成立 ✅

| 模式     | weights GiB | KV tokens | KV GiB | peak act GiB | max_concurrency |
| ------ | ----------- | --------- | ------ | ------------ | -------- |
| AR     | 8.61        | 249,856   | 10.82  | 1.63         | **61.00** |
| DFlash | 9.96        | 32,969    | 8.85   | 2.26         | **8.05** |

口径：vLLM 启动日志 kv_cache_utils.py:2178 "Maximum concurrency for 4,096 tokens per request"。注意因果方向：源码先算 max_concurrency、再反推 KV tokens（公式见 L3.5），两列自洽。旧值 195/25.75 是 ÷每请求预算 1280 的 workload 口径——**口径不成立（假设每请求 KV 需求随长度线性缩小），不作为 baseline**（0817 裁定，见 [[实验设计_C_eff归因_20260817]]）；0816 实测 C_eff=9 与源码口径 8.05 仅差 12%，机制未证明。

- KV tokens 下降 **86.8%**，max_concurrency 61.00→8.05（7.6x）
- 权重增量仅 +1.35 GiB（drafter 1.18 GiB + 辅助层）
- 并发扫描 batch 1-32 全部成功（vLLM 排队兜底）；DFlash 低并发大幅加速（batch 1: 5.6x），batch 32 被反超（0.86x）

并发吞吐对比（output 256 tokens, tok/s）：

| batch | AR tok/s | DFlash tok/s | DFlash/AR |
| ----- | -------- | ------------ | --------- |
| 1     | 28.4     | 158.8        | 5.6x      |
| 4     | 110.1    | 370.0        | 3.4x      |
| 16    | 413.9    | 577.7        | 1.4x      |
| 32    | 731.4    | 626.1        | 0.86x     |

DFlash 在低并发下大幅加速（5.6x），但高并发下被 KV 容量限制反超。

## L2：分配效率主导（非静态显存）✅

```
静态预算比例 (DFlash/AR available KV mem) = 0.818
实际容量比例 (DFlash/AR KV tokens)        = 0.132
相对差距                                   83.9%
权重增量                                   +1.35 GiB
```

**差距 83.9% → 分配效率损失是主因，不是静态显存。**

对比 Qwen3-8B（旧实验）：

|  | Qwen3-8B | Qwen3.5-4B |
|---|---------|-----------|
| 架构 | 纯 FullAttention | Hybrid Linear/Full |
| 静态预算比例 | 0.260 | 0.818 |
| 实际容量比例 | 0.227 | 0.132 |
| 相对差距 | **12.8%** | **83.9%** |
| 结论 | 静态显存主导 | **分配效率主导** |

## L3：KVCacheSpec group 碎片化 — 根因定位 ✅

### AR group 结构（正确分组，4 组）

| Group | Spec 类型 | 层数 | block_size | page_size |
|-------|----------|------|-----------|-----------|
| 0 | MambaSpec | 8 | 2048 | 2.0 MB |
| 1 | MambaSpec | 8 | 2048 | 2.0 MB |
| 2 | MambaSpec | 8 | 2048 | 2.0 MB |
| 3 | FullAttentionSpec | 8 | 16 | 2.0 MB |

同 spec 类型的层被正确合并（每 8 层 1 组）。

### DFlash group 结构（完全碎片化，38 组！）

| Group range | Spec 类型 | 每组层数 | 说明 |
|------------|----------|---------|------|
| 0-23 | MambaSpec | 1 | 24 个 target linear_attn 层，各自为组 |
| 24-31 | FullAttentionSpec | 1 | 8 个 target full_attn 层，各自为组 |
| 32-36 | SlidingWindowSpec | 1 | 5 个 drafter sliding_attn 层 |
| 37 | FullAttentionSpec | 1 | 1 个 drafter full_attn 层 |

**全部 38 层各自为 1 组——分组算法完全退化。**

### 根因链条

1. Qwen3.5-4B target 有 2 种 KV spec：MambaSpec（24层）+ FullAttentionSpec（8层，4 KV heads, head_size=256）
2. AR 模式：vLLM 正确将同 spec 的层分组 → 4 组
3. DFlash 加入 6 层 drafter：SlidingWindowSpec（5层, 8 KV heads, head_size=128）+ FullAttentionSpec（1层, 8 KV heads, head_size=128）
4. Drafter 层的 KV shape 与 target 不同 → vLLM group 算法遇到异构 spec 后退化为 per-layer 分组
5. 38 组意味着每个请求需要从 38 个 group 各分配 block → block 需求暴涨
6. KV 容量从 249,856 tokens 崩塌至 32,969 tokens（-86.8%）

### 关键数据

```
Group 膨胀: 4 → 38 (9.5x)
KV 容量损失: 86.8%
Page size: 2.0 MB → 2.3 MB (+11.5%)
```

## 三层结论

1. DFlash 在 Qwen3.5-4B（hybrid 架构）上的 serving capacity penalty 极其严重（KV -86.8%, max_concurrency 61→8.05）
2. 惩罚主因不是静态显存（+1.35 GiB drafter 权重），而是 **vLLM KVCacheSpec group 碎片化**（4→38 组）
3. 根因：drafter 层的异构 KV spec 导致 vLLM group 算法退化，每层独立成组

**与 Qwen3-8B 结论的关键差异**：纯 FullAttention 架构下 DFlash 容量惩罚来自静态显存（可量化解决）；hybrid 架构下容量惩罚来自系统级分配效率（需要改 vLLM group 算法或对齐 drafter KV spec）。

## L3.5：vLLM Group 算法源码分析

### 源码位置

- 入口: `get_kv_cache_groups()` — `vllm/v1/core/kv_cache_utils.py:1728`
- 核心分组: `_get_kv_cache_groups_uniform_page_size()` — 同文件 `:1137`
- Spec 定义: `vllm/v1/kv_cache_interface.py` — 全部 `@dataclass(frozen=True)`

### 分组算法 3 步流程

**Step 1: 按 spec 字段值分桶**

```python
same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
for layer_name, layer_spec in kv_cache_spec.items():
    same_type_layers[layer_spec].append(layer_name)
```

frozen dataclass 自动按所有字段做 `__hash__`/`__eq__`。字段完全相同的 spec 归入同一桶。

**Step 2: 计算 group_size**

```python
min_num_layers = min([len(layers) for layers in same_type_layers.values()])
group_size = min_num_layers
max_num_layers = max([len(layers) for layers in same_type_layers.values()])
if max_num_layers < min_num_layers * 1.5:  # heuristic
    group_size = max_num_layers
```

**Step 3: 按 group_size 切分每个桶**

```python
for layers in same_type_layers.values():
    num_groups = cdiv(len(layers), group_size)
    for i in range(num_groups):
        grouped_layers.append(layers[i::num_groups])
```

### AR 模式追踪（正确，4 组）

Step 1 — 2 个桶：
- 桶 A: MambaSpec → 24 层
- 桶 B: FullAttentionSpec(num_kv_heads=4, head_size=256) → 8 层

Step 2 — group_size = min(24, 8) = 8

Step 3 — 桶 A 切 3 组(8层/组) + 桶 B 切 1 组(8层) = **4 组**

### DFlash 模式追踪（碎片化，38 组）

Step 1 — 4 个桶：
- 桶 A: MambaSpec → 24 层
- 桶 B: FullAttentionSpec(4 KV heads, head_size=256) → 8 层（target）
- 桶 C: SlidingWindowSpec(8 KV heads, head_size=128) → 5 层（drafter）
- 桶 D: FullAttentionSpec(8 KV heads, head_size=128) → **1 层**（drafter last layer）

桶 B 和桶 D 都是 FullAttentionSpec，但字段不同 → 不同桶

Step 2 — group_size = min(24, 8, 5, **1**) = **1**

Step 3 — 每层 1 组：24 + 8 + 5 + 1 = **38 组**

### 根因（1 句话）

DFlash drafter 的 1 个 FullAttentionSpec 层（8 KV heads, head_size=128）与 target 的 FullAttentionSpec（4 KV heads, head_size=256）字段不同，产生 1 个只有 1 层的新桶，使 `min_num_layers=1` → `group_size=1` → 所有 38 层各自成组。

### 代码缺陷位置

```python
# vllm/v1/core/kv_cache_utils.py:1229
min_num_layers = min([len(layers) for layers in same_type_layers.values()])
group_size = min_num_layers
```

没有对 spec decoding drafter 层的特殊处理。drafter 引入的小 spec 类型（1-6 层）会通过 min() 把 group_size 拖到 1。

### max_concurrency 公式（0817 源码复核）

启动日志 61.00x / 8.05x 出自 `get_max_concurrency_for_kv_cache_config()`（kv_cache_utils.py:937）：

```
num_layer_per_group = max(len(group.layer_names))               # AR=8, DFlash=1
每请求最坏内存       = num_layer_per_group × Σ_groups spec.max_memory_usage_bytes()
memory_per_block     = groups[0].page_size_bytes × num_layer_per_group
num_block_per_req    = cdiv(每请求最坏内存, memory_per_block)
max_concurrency      = num_blocks ÷ num_block_per_req
```

单 spec 最坏内存（context 顶满 4096）：
- FullAttentionSpec（kv_cache_interface.py:258）: cdiv(4096, block_size) × page；target FA(4头×256, block=16) = 256 blocks
- MambaSpec（:709）: page × (1 + num_speculative_blocks)，与长度无关的常数（投机解码附加 block）
- SlidingWindowSpec（:590）: 窗口内 block 数 × page

**因果方向与直觉相反**：不是"KV tokens ÷ 4096 = 并发"，而是 先算每请求最坏 block 需求 → 并发 = 池 blocks ÷ 每请求 blocks → 反推 "GPU KV cache size = 并发 × 4096"。源码注释原话：*"Sourcing this from the concurrency calculation handles hybrid layouts correctly"*。

AR 手算复现 61.00x（0817 修正版）：Σ组最坏 = FA 256 block × 64KB（真实页 = 2×4头×256×16tok×2B）+ 3 × mamba页 ≈2.15MB ≈ 22.9MB；×8层 = 183MB；memory_per_block = 2.15MB×8 = 17.2MB → **11 blocks/请求**；池 676 blocks × 17.2MB ≈ 11.6GB = 10.82GiB ✓（顺带解开 0816 踩坑5：L3 json 的 AR 676 就是池总 group-blocks）；676÷11 = 61.45 ≈ 61.00（整除细节待 spec dump）；249,856 = 61×4096 ✓

⚠️ 勘误（0817）：上一版"259 blocks/请求、池≈15,799"是单位混用错误——15,799×17.2MB≈271GB 物理不可能，且"精确复现"实为循环论证（用 61 反推池、再除回 61）。已废弃。

DFlash：38 组全 1 层 → 每请求最坏 = 38 项直接相加；由日志反解 per-request = 4039÷8.05 ≈ **502 blocks ≈ 1.2GB**。但按 AR 同法朴素估算（mamba 48页×2.39MB + FA 8×256×64KB + drafter ≈ 0.28GB ≈ 117 blocks → 4039÷117 ≈ 34.5x）与 8.05 不符，缺口 ~4.3x 未解释——各 spec 的 page_size_padded / block_size / SW 窗口 / spec_blocks 精确值需 startup spec dump（[[实验设计_C_eff归因_20260817]] E1），502 的组成是 ledger 第一目标。

**机制假设（未证明，H1-H4）**：碎片化可能通过 cdiv 取整/页对齐放大（H2）、per-group 运行时预留放大（H1）、SD 专属预留（H3）、mamba 常数 state 主导（H4）抬高每请求开销。0816 实测 C_eff=9 ≈ 源码口径 8.05 目前只是观测——"哪个组先满、经什么机制"待 per-group ledger 钉死；25.75（÷1280）不作为 baseline。

## 优化方向（路径 B：allocation optimization）

1. **修复 vLLM group 算法**：让 target 的同 spec 层即使有 drafter 存在也能合并（4 组 → 理论可恢复到 6 组：4 target + 2 drafter）
2. **Drafter KV spec 对齐**：让 drafter 层使用与 target 相同的 KV head 数和 head_size
3. **自定义 KVCacheManager**：对异构 group 做差异化 block 分配（MambaSpec 需要极少 block，不应与 FullAttention 等量分配）

## 远程产物

| 文件 | 路径 |
|---|---|
| L1 结果 | outputs/l1_capacity_qwen35_20260812.json |
| L1 脚本 | scripts/l1_capacity_qwen35.py |
| L3 结果 | outputs/l3_kvspec_qwen35_20260812.json |
| L3 脚本 | scripts/l3_kvspec_qwen35.py, scripts/l3_kvspec_qwen35_v2.py |
| 数据 | data/codecontests_qwen35_4b_1024_256.jsonl（13/14 PASS, 1 误报） |
| L1 AR log | logs/l1_ar_server_20260812.log |
| L1 DFlash log | logs/l1_dflash_server_20260812.log |

## 数据审核结果（Qwen3.5-4B tokenizer）

- 256 条，全部精确 1024 tokens（独立复算确认）
- raw length 1024-1279，254/256 content-level 截断
- chat template: enable_thinking=False（Qwen3.5 特有后缀 `</think>`）
- 13 PASS / 0 WARN / 1 FAIL（"solution" marker 全部为竞赛题正文自然用语，非答案泄漏）
