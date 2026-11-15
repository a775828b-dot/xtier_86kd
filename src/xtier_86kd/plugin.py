"""SGLang plugin entry point; use SGLang's MoE runner unchanged."""

import logging
import os
import sys

logger = logging.getLogger(__name__)
_installed = False


def _in_scheduler():
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "run_scheduler_process":
            return True
        frame = frame.f_back
    return False


def _require_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry

    apply_hooks = HookRegistry.apply_hooks

    def checked():
        apply_hooks()
        failed = sorted(
            target
            for target, hooks in HookRegistry._hooks.items()
            if target not in HookRegistry._patched
            and any(getattr(h[-1], "plugin_name", None) == "xtier_86kd" for h in hooks)
        )
        if failed:
            raise SystemExit(f"xtier_86kd: SGLang hooks not applied: {failed}")

    if hasattr(HookRegistry, "_hooks") and hasattr(HookRegistry, "_patched"):
        HookRegistry.apply_hooks = checked


def load():
    global _installed
    if os.environ.get("XTIER_86KD") != "1" or not _in_scheduler() or _installed:
        return
    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
        from xtier_86kd.mxfp8_prefill import install as install_mxfp8_prefill
        from xtier_86kd.packs.deepseek_v4 import install

        _require_hooks()
        install_mxfp8_prefill(HookRegistry, HookType)
        install(HookRegistry, HookType)
        _installed = True
        logger.info("xtier_86kd: DeepSeek V4.1 hooks registered in scheduler PID %d", os.getpid())
        print(f"[xtier_86kd] DeepSeek hooks registered in scheduler PID {os.getpid()}", file=sys.stderr, flush=True)
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise SystemExit(f"xtier_86kd plugin setup failed: {type(exc).__name__}: {exc}") from exc

