# sitecustomize.py — E3 打点自动注入点(放 scripts/,经 PYTHONPATH 生效)
# 仅当 E3_LEDGER 设置时激活;EngineCore spawn 子进程启动时自动执行本文件。
import os

if os.environ.get("E3_LEDGER"):
    try:
        import e3_patches
        e3_patches.install()
    except Exception as _e:  # 打点失败不阻断 server
        print(f"[e3] sitecustomize install failed: {_e!r}", flush=True)
