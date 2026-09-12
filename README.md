# LiteDrafter

LiteDrafter 研究资源受限 GPU 上的 speculative decoding serving。项目关注的不是单请求 decode 是否更快，而是：DFlash 的单请求加速能否在 vLLM continuous batching 中转化为真实吞吐，以及容量损失来自哪里。

当前重点是单卡 RTX 4090、vLLM 0.26.0、V2 Model Runner、Qwen3.5-4B Hybrid 和 DFlash drafter。

## 核心发现

Hybrid 模型的主要瓶颈是 Mamba/GDN speculative state，而不是 attention kernel 或 page 对齐。当前 DFlash KV layout 约包含 24 个 Mamba group、8 个 target FullAttention group、5 个 SlidingWindow group 和 1 个 drafter FullAttention group。

在 `n_spec=15` 时，每个请求约需要 426 blocks，其中约 384 个来自 Mamba，占 admission 需求约 93%。因此 speculative token 数增加会近似线性放大每请求容量成本。

## vLLM 动态 K 的容量盲区

vLLM 0.26.0 支持按当前 scheduler batch 选择 speculative token 数：

```json
{"num_speculative_tokens":15,"num_speculative_tokens_per_batch_size":[[1,4,15],[5,64,3]]}
```

每个三元组是 `[最小 batch, 最大 batch, K]`。上述配置表示当前 scheduler step 有 1~4 个 request 时使用 K=15，有 5~64 个 request 时使用 K=3。这里的 batch 是当前 scheduler step 选中的 request 数。

原始 Hybrid 路径中，动态表改变了 model runner 的计算宽度，但 Mamba speculative admission 仍按启动时最大值初始化，形成：

```text
实际计算：K=3
容量预留：K=15
```

Phase A 实测表明，高并发时只有前三个 speculative position 有效，但 running ceiling 仍约为固定 `n_spec=15` 的 9 个请求，吞吐约 462 tok/s。

## Admission patch

补丁把 scheduler 当前选择的有效 K 接入 Mamba speculative block 的 admission 和释放路径：

```text
当前 scheduler K -> 按当前 K 申请 blocks -> target 验证
-> accepted state 提交 -> rejected state 释放 -> 下一轮重新选择 K
```

只修改 `num_speculative_tokens_to_schedule` 会再次出现“省计算、不省容量”的假优化；真正修改必须同时影响 Mamba block accounting 和 release。

## Patch 后效果

固定 workload 约为输入 1024 tokens、输出 256 tokens：

| 配置 | C | 结果 |
|---|---:|---|
| patch 前动态 K | 16 | running 约 9，capacity waiting，约 462 tok/s |
| patch G2 | 16 | running=16，无 capacity waiting，约 699 tok/s |
| patch G2 | 32 | 稳态容量包络约 31 running，约 955 tok/s |

C=32 时 running 会因请求完成和新请求准入在约 27~31 之间锯齿；稳态 KV 使用率约 0.92~1.00。prefill 爬坡和最后 drain 不属于稳态容量。

## 复现实验

修改 `scripts/c_eff_scan.py` 中的模型和环境路径后运行：

```bash
python scripts/c_eff_scan.py --mode dflash --num-spec-tokens 7 --concurrencies 1,4,8,16,24,32
python scripts/c_eff_scan.py --mode dflash --num-spec-tokens 15 --spec-schedule '[[1,4,15],[5,64,3]]' --concurrencies 4,8,16
```

默认数据集是 `data/codecontests_qwen35_4b_1024_256.jsonl`。结果写入 `outputs/`，日志写入 `logs/`，默认不上传。

## 统计口径

- `C` 是客户端 closed-loop 并发度；`running` 和 `waiting` 是时变 gauge。
- `max_running` 和 `max_waiting` 是独立峰值，可能发生在不同阶段，不能直接相加。
- 容量结论使用 raw telemetry 的稳态窗口，排除 prefill 爬坡和 drain。
- server speculative counter 包含 settling 请求；客户端吞吐排除前 C 个完成请求。
- acceptance length 和吞吐不能证明输出正确性，需要 token-level 或 task-level 校验。

## 仓库结构

`scripts/` 是实验脚本，`data/` 是 metadata 和小型输入，`env/` 是环境快照，`docs/` 是归因文档，`results/` 是结果摘要。模型权重、密码、运行中的服务和远端绝对路径不属于仓库。

## 后续工作

继续验证 admission patch 在完成、拒绝 token、EOS、preemption 和 K 改变时的 state 提交与释放；之后评估 target/drafter 独立 physical KV layout，再做 Mamba state 压缩和 D-Cut。

详细项目过程与归因复盘见 [docs/项目总结.md](docs/项目总结.md)。
