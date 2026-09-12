# LiteDrafter

LiteDrafter studies speculative decoding for memory constrained LLM serving. The goal is to explain why DFlash can improve single request latency while failing to improve, or even reducing, throughput under vLLM continuous batching.

The current focus is Qwen3.5-4B Hybrid with a DFlash drafter on one RTX 4090, using vLLM 0.26.0 and the V2 model runner.

## Main finding

In the Hybrid model, Mamba/GDN speculative state is the dominant per request capacity cost. The DFlash KV layout contains approximately 24 Mamba groups, 8 target FullAttention groups, 5 SlidingWindow groups, and 1 drafter FullAttention group. At `n_spec=15`, one request uses about 426 blocks; about 384 are Mamba blocks, roughly 93% of the admission requirement.

## The vLLM dynamic K problem

vLLM supports a schedule such as:

```json
{
  "num_speculative_tokens": 15,
  "num_speculative_tokens_per_batch_size": [[1, 4, 15], [5, 64, 3]]
}
```

Each tuple is `[minimum_batch_size, maximum_batch_size, K]`. The example selects `K=15` when the current scheduler step contains 1--4 requests and `K=3` when it contains 5--64 requests. Here batch size means the number of requests selected in that scheduler step, not the total requests seen by the server.

The problem in the original Hybrid path is that the schedule changes model computation width but Mamba speculative admission is initialized from the startup maximum. At high load the service therefore computes with `K=3` while reserving state as if `K=15`:

```text
computation: K=3
admission:   K=15
```

The unused positions are already physical admission cost. Phase A measured this at runtime: high-load speculative execution used only three active positions, but the running ceiling remained the fixed-`n_spec=15` ceiling of about 9 requests. Throughput was about 462 tok/s, worse than using a correctly started fixed K.

## What the admission patch does

The patch connects the effective K selected by the scheduler to Mamba speculative block accounting. The lifecycle is:

```text
current scheduler K
-> admit Mamba speculative blocks for that K
-> verify draft tokens
-> commit accepted state
-> release rejected state
-> choose K again at the next scheduler step
```

Changing only `num_speculative_tokens_to_schedule` would reduce computation without releasing capacity. The patch changes the admission and release path as well.

## Runtime effect

The fixed workload uses approximately 1024 input tokens and 256 output tokens.

| configuration | target C | result |
|---|---:|---|
| dynamic K before admission patch | 16 | running capped at 9; capacity waiting; about 462 tok/s |
| admission patch G2 | 16 | running 16; no capacity waiting; about 699 tok/s |
| admission patch G2 | 32 | steady capacity envelope about 31 running; about 955 tok/s |

At C=32, running naturally breathes between roughly 27 and 31 as requests complete and waiting requests are admitted. KV usage is roughly 0.92--1.00 in steady state. A prefill ramp-up waiting peak and the final drain must be excluded when interpreting capacity.

## Reproduction

Set the local model paths and environment in `scripts/c_eff_scan.py`, then run:

```bash
python scripts/c_eff_scan.py --mode dflash --num-spec-tokens 7 --concurrencies 1,4,8,16,24,32
```

Dynamic-K diagnostic:

```bash
python scripts/c_eff_scan.py --mode dflash --num-spec-tokens 15 --spec-schedule '[[1,4,15],[5,64,3]]' --concurrencies 4,8,16
```

The standard workload is `data/codecontests_qwen35_4b_1024_256.jsonl`. Results go to `outputs/` and server logs to `logs/`.

## Measurement rules

- `C` is client closed-loop concurrency; server `running` and `waiting` are time-varying gauges.
- `max_running` and `max_waiting` are independent time-window maxima and may occur in different phases.
- Use raw telemetry and a steady-state window; exclude prefill ramp-up and final drain.
- Server speculative counters include settling requests; client throughput follows the established protocol and excludes the first `C` completed requests.
- Throughput and acceptance length do not prove correctness; use token-level or task-level checks.

## Repository contents

- `scripts/`: launchers, capacity scans, KV ledgers, audits, and analysis;
- `data/`: benchmark inputs and metadata;
- `env/`: environment snapshots;
- `E0_E1_归因执行记录_20260818.md` and `E3_现场Ledger执行记录_20260818.md`: attribution evidence;
- `outputs/` and `logs/`: local artifacts, excluded from a source-only GitHub release.

Model weights, credentials, running services, and remote absolute paths are not part of this repository.

## Next work

Harden the admission patch across completion, rejection, EOS, preemption, and changing K. Then evaluate separate target/drafter physical KV layouts. Mamba state compression and D-Cut are later experiments because they address secondary costs after admission is correct.
