#!/bin/bash
# 手动启动 n_spec=3 dflash server (与 c_eff_scan.py build_cmd 逐参数一致)
CONDA_ENV=/root/autodl-tmp/conda_envs/env_vllm026
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V2_MODEL_RUNNER=1
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:$LD_LIBRARY_PATH
export PATH=$(dirname ${CONDA_ENV}/bin/python):$PATH
exec ${CONDA_ENV}/bin/python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen3.5-4B --port 8301 \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching --tensor-parallel-size 1 \
  --served-model-name qwen35-4b --enforce-eager \
  --speculative-config '{"method": "dflash", "model": "/root/autodl-tmp/models/Qwen3.5-4B-DFlash", "num_speculative_tokens": 3}'
