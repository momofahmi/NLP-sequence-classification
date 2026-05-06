"""Environment shims for running on Colab.

Colab pre-installs torchao==0.10.0. Recent peft (>=0.13) calls
``is_torchao_available`` while injecting LoRA adapters and raises
``ImportError`` when it finds an old torchao. We do not use torchao
for plain LoRA, so ``ensure_peft_compat`` simply removes it.

Call this function before importing or using ``peft.PeftModel``.
"""

from __future__ import annotations

_DONE = False


def ensure_peft_compat() -> None:
    global _DONE
    if _DONE:
        return
    _DONE = True

    import importlib
    import importlib.util
    import subprocess
    import sys

    spec = importlib.util.find_spec("torchao")
    if spec is None:
        return

    try:
        import torchao
        from packaging.version import Version

        if Version(torchao.__version__) >= Version("0.16.0"):
            return
    except Exception:
        pass

    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
        check=False,
    )
    sys.modules.pop("torchao", None)
    importlib.invalidate_caches()
