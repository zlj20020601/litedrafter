# KV 容量 Ledger（账本）与 Mamba 状态分析

> 本文量化各类 KV state 的物理占用，重点确认 Mamba/GDN recurrent state 对容量的影响。

> **做了什么:** 在 LIVE dflash server 上以 monkey-patch 包装器(零改动 site-packages)打点 KVCacheManager 准入路径,跑 C=8/9/10 三点,钉死"DFlash 第 10 个请求被哪个组的需块挡住"与稳态 per-group 实际持块,闭合因果链④(38 groups → capacity=9)。

## 目的(对应 0817 设计验收标准)


## 打点设计(零改动 site-packages)

包装器 `scripts/e3_ledger_server.py`,argv 与 c_eff_scan.build_cmd("dflash") 完全一致:

| 事件 | 挂点 | 内容 |
|---|---|---|
| group_table | KVCacheManager.__init__ 后 | 静态组表 gid/type/block_size/page/layers + 池总量 |
| admit_fail | allocate_slots 返回 None | 完整 per-group 需求分解(每 rid 第 1/10/1000 次限流) |
| admit_ok | 首次分配成功(computed==0, new>=256) | per-group 需求 + 当时池 free |
| finish | free(request) 前 | coordinator.get_blocks() 每组实际持块 |

关键源码事实(打点依据):
- 准入失败 = allocate_slots 返回 None;判定为 sum(per-manager need) > block_pool.get_num_free_blocks()(共享池,uniform page)
- SingleTypeKVCacheManager 每组一个(38 个),get_num_blocks_to_allocate 逐组算 need
- free(self, request: Request);get_blocks(request_id) 返回 tuple[list[KVCacheBlock], ...] 按组

## 执行步骤(逐步更新)

- [x] 源码复核:allocate_slots 失败路径(L283-460)、coordinator(L130-190)、manager 属性(python3.12 路径)
- [x] wrapper 方案定稿:sitecustomize+PYTHONPATH 注入(EngineCore spawn 子进程自动装补丁),e3_patches.py 幂等
- [x] server 启动(pid 11671,ledger=outputs/e3_ledger_dflash_20260818_1123.jsonl)
- [x] c_eff_scan --mode dflash --keep-server --concurrencies 8,9,10 → C=8/9 全进(run=8/9,wait=0),C=10 run=9/wait=1
- [x] ledger 分析:Q1/Q2/502 全闭合(scripts/e3_analyze.py)
- [x] 结论回写 + 同步服务器

## 结果(核心数字)


| C | max_run | max_wait | kv | 结论 |
|---|---|---|---|---|
| 8 | 8 | 0 | 0.866 | 全准入 |
| 10 | 9 | 1 | 0.975 | **第 10 个被拒 → queueing knee=10** |

→ 正式口径:**active concurrency ceiling=9,queueing knee=10**(0817 术语裁定条件满足)

**Q1:第 10 个请求被谁挡住(admit_fail 分解,245 条一致):**

- 需求:总 412 blocks = **mamba 384(93%)** + attn 28(7%)
- mamba:24 组 × 16 blocks(常数,与长度无关)= speculative states(1+15)
- attn:14 组 × 2 blocks = cdiv(1012+16, 592)(runtime block_size=592,非离线 dump 的 16)
- 池 free=169 < 412 → 拒。9 稳态请求 × 426 = 3834,池 3919,余量恒 <412 → 第 10 个永远进不去

**Q2:每请求稳态实际持块(772 完整请求,p50=mean=max=426,零方差):**

| 组类型 | 组数 | 每组持块 | 小计 | 字节(2.42MB/块) |
|---|---|---|---|---|
| MambaSpec(GDN 单层) | 24 | 16(常数) | 384 | **931MB(93%)** |
| FullAttention+SW | 14 | 3(cdiv(1268+16,592)) | 42 | 101MB(7%) |

**502/1294 之谜闭合:**

- 实测 426 ↔ runtime 反解 4039/9≈449(差值=启动账目杂项,无未建模机制)
- 0817 反解 ~1.05GB/请求 ↔ 实测 426×2.42MB≈0.97GB ✓


1. **④ 闭合:DFlash 容量惩罚的 binding constraint 是 mamba 组碎片化×spec 常数 state**:AR 3 组×8 层合并、每请求 3 blocks;DFlash 24 单层组、每请求 384 blocks(组数×8 × spec slots×16 = 128×),占准入需求 93%。第 10 个请求被 24 个 mamba 组的 384 块常数需求挡住,实锤。
3. H3(lookahead+1)在 592 粒度下不可见(只在精确边界跳块);H4(mamba 常数主导)为最终主因,且量化为 931MB/请求。
4. 池账本:3919 块(启动日志 4039 含 null/杂项);ceiling=9 = floor((3919-杂项)/426)。

## 产出文件

- `outputs/e3_ledger_dflash_20260818_1123.jsonl`(1790 事件:group_table/admit_ok×772/admit_fail×245/finish×772)
- `outputs/c_eff_dflash_20260818.json` + raw gauge gz(C=8/9/10)
- `scripts/e3_patches.py`(幂等补丁)/`sitecustomize.py`/`e3_ledger_server.py`/`e3_run.sh`/`e3_analyze.py`

## 踩坑

3. pkill -f 会匹配自身 ssh 命令行自杀(exit 255);用 pidfile+精确 pid 杀
4. 崩溃残留的 EngineCore 孤儿进程占 22GB 显存会让下次启动报"Free memory 不足"——启动前查 nvidia-smi compute-apps
5. 完整请求 ntok=1268(1024 prompt+256 out−4?以 finish 事件为准),不是预设的 1280

## 关联


