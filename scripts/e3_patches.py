#!/usr/bin/env python3
"""
e3_patches.py — E3 现场 per-group KV ledger 补丁(幂等,2026-08-18)

由 sitecustomize.py(EngineCore spawn 子进程)与 e3_ledger_server.py(父进程)共同调用,
install() 带幂等标记,双进程重复调用安全。激活条件:环境变量 E3_LEDGER 已设置。

事件(JSONL,由 E3_LEDGER 指定路径,行缓冲追加):
  group_table / admit_fail / admit_ok / finish
详见 E3_现场Ledger执行记录_20260818.md。
"""
import json
import os
import threading
import time

MARKER = "_e3_installed"


def install():
    import vllm.v1.core.kv_cache_manager as kcm_mod
    import vllm.v1.core.single_type_kv_cache_manager as stk

    if getattr(kcm_mod.KVCacheManager.allocate_slots, MARKER, False):
        return False  # 已装(幂等)

    ledger_path = os.environ.get("E3_LEDGER")
    if not ledger_path:
        return False

    _f = open(ledger_path, "a", buffering=1)
    _lock = threading.Lock()

    state = {"cur": [], "fail_n": {}}

    def _write(rec):
        with _lock:
            rec["t"] = round(time.time(), 3)
            _f.write(json.dumps(rec, default=str) + "\n")

    def _spec_info(spec):
        return {
            "type": type(spec).__name__,
            "block_size": getattr(spec, "block_size", None),
            "page": getattr(spec, "page_size_bytes", None),
            "layers": len(getattr(spec, "layer_names", []) or []),
        }

    # 1) 静态组表 + 池总量
    _orig_init = kcm_mod.KVCacheManager.__init__

    def _init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        try:
            groups = self.kv_cache_config.kv_cache_groups
            _write({
                "ev": "group_table", "pid": os.getpid(),
                "num_groups": len(groups),
                "pool_free_at_init": self.block_pool.get_num_free_blocks(),
                "groups": [{"gid": i, **_spec_info(g.kv_cache_spec)}
                           for i, g in enumerate(groups)],
            })
        except Exception as e:
            _write({"ev": "group_table_error", "err": repr(e)})

    kcm_mod.KVCacheManager.__init__ = _init

    # 2) per-group 需求(挂 SingleType 基类,覆盖 FA/SW/Mamba 全部子类)
    _orig_st_get = stk.SingleTypeKVCacheManager.get_num_blocks_to_allocate

    def _st_get(self, *a, **kw):
        need = _orig_st_get(self, *a, **kw)
        try:
            state["cur"].append({
                "gid": self.kv_cache_group_id,
                **_spec_info(self.kv_cache_spec),
                "need": need,
            })
        except Exception:
            pass
        return need

    stk.SingleTypeKVCacheManager.get_num_blocks_to_allocate = _st_get

    # 3) allocate_slots:失败分解 / 首次成功
    _orig_alloc = kcm_mod.KVCacheManager.allocate_slots

    def _alloc(self, request, num_new_tokens=0, *a, **kw):
        state["cur"] = []
        ret = _orig_alloc(self, request, num_new_tokens, *a, **kw)
        try:
            look = kw.get("num_lookahead_tokens", 0)
            if ret is None:
                n = state["fail_n"].get(request.request_id, 0) + 1
                state["fail_n"][request.request_id] = n
                if n in (1, 10) or n % 1000 == 0:
                    _write({
                        "ev": "admit_fail", "rid": request.request_id,
                        "status": str(getattr(request, "status", "")),
                        "ntok": request.num_tokens,
                        "nprompt": request.num_prompt_tokens,
                        "computed": request.num_computed_tokens,
                        "new": num_new_tokens, "lookahead": look,
                        "pool_free": self.block_pool.get_num_free_blocks(),
                        "fail_n": n,
                        "decomp": list(state["cur"]),
                    })
            elif request.num_computed_tokens == 0 and num_new_tokens >= 256:
                _write({
                    "ev": "admit_ok", "rid": request.request_id,
                    "ntok": request.num_tokens,
                    "nprompt": request.num_prompt_tokens,
                    "new": num_new_tokens, "lookahead": look,
                    "pool_free": self.block_pool.get_num_free_blocks(),
                    "decomp": list(state["cur"]),
                })
        except Exception as e:
            _write({"ev": "alloc_log_error", "err": repr(e)})
        return ret

    _alloc.__dict__[MARKER] = True
    kcm_mod.KVCacheManager.allocate_slots = _alloc

    # 4) 请求结束:per-group 实际持块
    _orig_free = kcm_mod.KVCacheManager.free

    def _free(self, request, *a, **kw):
        try:
            blocks = self.coordinator.get_blocks(request.request_id)
            _write({
                "ev": "finish", "rid": request.request_id,
                "status": str(getattr(request, "status", "")),
                "ntok": request.num_tokens,
                "held": [len(b) for b in blocks],
                "pool_free": self.block_pool.get_num_free_blocks(),
            })
        except Exception as e:
            _write({"ev": "free_log_error", "err": repr(e)})
        return _orig_free(self, request, *a, **kw)

    kcm_mod.KVCacheManager.free = _free

    print(f"[e3] patches installed (pid={os.getpid()}), ledger -> {ledger_path}",
          flush=True)
    return True
