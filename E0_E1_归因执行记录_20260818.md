# E0/E1 执行记录:C_eff 归因(2026-08-18)

> **做了什么:** 完成 0817 设计的 E0(0816 原始 gauge 稳态重析)与 E1(vLLM 0.26.0 KV 分配源码侦察 + 离线 spec dump/合成准入 ledger),确认 DFlash 稳态 ceiling=9 铁证、H1 死亡、H3/H4 源码坐实、38 组完整解剖(24 单层 mamba + 14 单层注意力)、uniform-page 垫页机制(所有组 page 统一 2.28MiB)。

## E0:稳态重析(无卡,已完成)

方法:raw gauge(AR 20507 / DF 11745 条,0.21s 间隔)按 summary 累积时长切分,与 running+waiting 台阶检测 16/16 边界对齐(±10s);稳态=每段末 80%。

**结果(关键数字):**

| 模式 | 稳态 running | 稳态 waiting | 稳态 kv | 结论 |
|---|---|---|---|---|
| AR C=1..96 | 全部 =C(p50=p95=max) | **全部 0**(w>0%=0%) | 0.009→0.860 线性 | AR 无稳态排队 |
| DF C=8 | 8 | 0 | 0.843-0.862 | 最后一个无排队的点 |
| DF C=12..96 | **全部 9**(p50=p95=max) | =C-9 精确(3,7,11,14,14,16,27,32,41,57,67) | 0.953-0.975 饱和 | ceiling=9 铁证 |

- **AR "C=20起 waiting=2→62" 全部来自初始 burst**(full_wait_max 2→62 但稳态全 0)→ 0817 担心的 token-budget 混杂变量在稳态口径下消失,E0 目标 ③ 闭合
- DF waiting=C-9 精确成立且 ~80-94% 时间非零 → 第 10 个请求全程被拒
- scheduled_tokens 未记录(gauge schema 只有 running/waiting/kv_usage/wbr)
- preempt=0 全程(两模式)
- 产出:`outputs/e0_steady_state_20260818.json`;补点缺口:DF 缺 C=9/10/11,AR 缺 C=104-128

## E1:源码侦察 + 离线 spec dump(有卡,已完成)

### H1-H4 裁定(源码级)

| 假设 | 裁定 | 源码证据 |
|---|---|---|
| H1 watermark 放大 | **死亡** | watermark 默认 0.0(config/scheduler.py L146);且为全局池级非 per-group(kv_cache_manager L168) |
| H2 页对齐当量放大 | **加强版坐实 → 0818 E3 修正为"存在但非binding"** | uniform-page-size 算法把**所有组**的 page 垫到一致:DF 全部 2,392,064B(2.28MiB)——FA-target 单层真实只需 65,536B/页,被垫 **36.5×**;AR 全部 2,146,304B(FA 8层组真实 524,288B,垫 4.09×)。**E3 修正:runtime block_size=592 tok/block(非本 dump 的 16),attn 每请求仅 42 blocks、字节全被利用,"36.5× 浪费"系 block 口径误读,attn 不是瓶颈** |
| H3 SD 专属预留 | **坐实** | scheduler L263:dflash lookahead=num_spec+1=**16**(比 eagle/dspark 多 1);MambaManager 非 align 模式 num_tokens += bs×num_speculative_blocks(L1444);admit 时 attn 组需求 cdiv(1024+16,16)=65 |
| H4 mamba 常数 state | **坐实且量化** | MambaSpec num_speculative_blocks=15(dflash 下);离线 ledger:mamba 组每请求 16 blocks(=1+15);AR 下 mamba 3 组×8 层合并,每请求仅 **1 block/组** |

另:`scheduler_reserve_full_isl` 默认 True 但 full_num_tokens=request.num_tokens(=prompt 1024),非 max_model_len → 解释 AR 按实长 admission(C=96>61)。

### 38 组完整解剖(离线 dump,0816 serving 同配置)

| 组 | 类型 | 层数/组 | block_size | page(垫后) | num_spec_blocks | 每请求blocks@1280 |
|---|---|---|---|---|---|---|
| 0-23 | MambaSpec(GDN) | 1 | 4096 | 2,392,064 | 15 | 16(常数) |
| 24-31 | FullAttention(target) | 1 | 16 | 2,392,064 | - | 65(admit)→81 |
| 32-36 | SlidingWindow(drafter,window=4096) | 1 | 16 | 2,392,064 | - | 65→81 |
| 37 | FullAttention(drafter,8h/128) | 1 | 16 | 2,392,064 | - | 65→81 |

AR 对照:4 组 = FA(8层合并,page 524,288→垫 2,146,304)+ 3×Mamba(各8层,bs=4096);每请求 [1,1,1,80] = **83 blocks**。

**DF 每请求 1294 blocks = 24×16 + 14×65+16(growth 不全受池余量限制)** vs AR 83 → 15.6×。

### 池账本:离线与 runtime 的未闭合差

- capacity 公式(kv_cache_utils L937):`memory_per_block = page × num_layer_per_group;nbpr = cdiv(mmur, mpb);conc = num_blocks/nbpr`
  - AR 离线:mpb = 2,146,304×8 = 17.17MB(**正是 0817 勘误笔记的"17.2MB"**);nbpr=259(=3+256);conc=676/259=2.61x
  - DF 离线:mmur≈9.4GiB(24×2.28MiB×16 + 14×256×2.28MiB),需 8.85GiB>avail 报错(与 runtime 压线启动一致,runtime 4039 blocks / 502 nbpr = 8.05x)
- **离线重建无法复现 runtime 的 61.00x/8.05x 与 502**——n_spec 语义(mamba 1+1 还是 1+16)、attn 页是否真垫、离线 spec 与 runtime spec 的差异需现场裁决
- 离线 AR 池 676 blocks 复现的是 0812 配置(max_len 2048),0816 AR(4096/128)runtime 为 249,856 tokens/61.00x,同样未闭合

### 关键机制结论(可写入口径)

1. DFlash 组碎片化不只是"38 组":**24 个 GDN 层全部拆成单层组**(AR 下 8 层/组共享一页),每请求 mamba 块需求从 3(AR)暴涨到 384(DF)
2. ~~**uniform-page 垫页是独立的第二重放大**:垫页后每 block 成本统一为 max 页,FA-target 真实 KV 需求被垫 36.5×~~ **0818 E3 改写:block 当量放大真实存在(runtime 592 tok/block)但它让 attn 组每请求只需 42 块——垫页在字节层被用满,不构成独立浪费放大;第二重放大实为 mamba 的 spec 常数 state(16 blocks/组,24 组)**
3. lookahead=16 直接进 attn 组 cdiv 分子(admit 65 vs AR 64,+1 block/组/请求)
4. DF per-req@1280=1294 blocks(离线上界)vs runtime 反解 ~449 blocks(4039/9)→ ~~runtime 有回收机制未建模(推测 mamba 实际持 2 非 16、attn 按当前长度)~~ **0818 E3 裁决:无回收机制——mamba 实持 16✓,attn 按 592 tok/block 实持 3,合计 426(772 请求零方差),离线 1294 高估全部来自 attn 的 block_size=16 假设**

## 下一步(待确认)

- **E3(优先)**:monkey-patch 包装器(不动 site-packages)在 LIVE dflash server 上,admission 失败路径打点 + per-group 持块打印,C=8/9/10 三点 → 钉死"第 10 个请求被哪个组挡住" + 502 精确分解
- E2:补点 DF C=9/10/11(knee onset)+ AR C=104-128(AR knee),~40min GPU
- E4(可选):max_model_len=1280 启动 DF,分解常数项/长度项

## 产出文件

- `outputs/e0_steady_state_20260818.json`(稳态表)
- `outputs/e1_specdump_ledger_20260818.json`(组表+ledger)
- `scripts/e0_final.py` / `scripts/e1_specdump_ledger_20260818.py`
- 踩坑:LD_LIBRARY_PATH 必须在启动命令导出(进程内 os.environ 对动态加载器无效)

## 关联

- [[实验设计_C_eff归因_20260817]](E0-E4 计划)
- [[阶段二_C_eff拐点扫描_20260816]](原始数据)
- [[三层容量实验_Qwen3.5-4B_20260812]](38 组发现)
