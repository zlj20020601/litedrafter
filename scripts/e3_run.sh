#!/bin/bash
# e3_run.sh — E3 现场 ledger:带打点启动 dflash server(nohup 分离,写 manifest)
# 参数与 c_eff_scan.build_cmd("dflash") 完全一致;保持 0816 同款多进程架构。
# PYTHONPATH 注入 sitecustomize → EngineCore spawn 子进程自动装补丁(仅 E3_LEDGER 设置时)。
set -e
export PATH=/root/autodl-tmp/conda_envs/env_vllm026/bin:$PATH
ROOT=/root/autodl-tmp/litedrafter
PY=/root/autodl-tmp/conda_envs/env_vllm026/bin/python
STAMP=$(date +%Y%m%d_%H%M)
LEDGER=$ROOT/outputs/e3_ledger_dflash_$STAMP.jsonl
LOG=$ROOT/logs/e3_server_dflash_$STAMP.log

export HF_ENDPOINT=https://hf-mirror.com
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V2_MODEL_RUNNER=1
export LD_LIBRARY_PATH=/root/autodl-tmp/conda_envs/env_vllm026/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/scripts:${PYTHONPATH:-}
export E3_LEDGER=$LEDGER

nohup $PY $ROOT/scripts/e3_ledger_server.py \
  --model /root/autodl-tmp/models/Qwen3.5-4B \
  --port 8000 \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --served-model-name qwen35-4b \
  --enforce-eager \
  --speculative-config '{"method":"dflash","model":"/root/autodl-tmp/models/Qwen3.5-4B-DFlash","num_speculative_tokens":15}' \
  > $LOG 2>&1 &

echo $! > $ROOT/e3_server.pid
cat > $ROOT/e3_manifest.json <<EOF
{"pid": $(cat $ROOT/e3_server.pid), "ledger": "$LEDGER", "log": "$LOG", "stamp": "$STAMP"}
EOF
echo "server pid=$(cat $ROOT/e3_server.pid)"
echo "ledger=$LEDGER"
echo "log=$LOG"
