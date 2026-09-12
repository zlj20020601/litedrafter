#!/usr/bin/env python3
"""e3_ledger_server.py — E3 wrapper:装补丁后原样启动 api_server(参数不变)"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_patches

if not os.environ.get("E3_LEDGER"):
    print("FATAL: E3_LEDGER env not set", flush=True)
    sys.exit(1)
e3_patches.install()  # 父进程;EngineCore 子进程经 sitecustomize 注入

runpy.run_module("vllm.entrypoints.openai.api_server",
                 run_name="__main__", alter_sys=True)
