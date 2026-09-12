# Hybrid Patch Summary

Workload: Qwen3.5-4B Hybrid, approximately 1024 input tokens and 256 output tokens, vLLM 0.26.0, RTX 4090.

| Configuration | C | Result |
|---|---:|---|
| Dynamic K before admission patch | 16 | running capped at 9, capacity waiting, about 462 tok/s |
| Admission patch G2 | 16 | running 16, no capacity waiting, about 699 tok/s |
| Admission patch G2 | 32 | steady capacity envelope about 31 running, about 955 tok/s |

At C=32, max_running and max_waiting are independent window maxima. The steady-state running count breathes near 27--31 as requests finish and waiting requests are admitted. Prefill ramp-up and final drain are excluded from capacity interpretation.
