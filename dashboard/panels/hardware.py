"""Hardware panel — live GPU/CPU/disk stats polled on each rerun.

Reads:
- NVML via pynvml (GPU VRAM + util + temp)
- psutil (CPU, RAM, disk)

Writing these values into the run's metrics.jsonl is left to an optional
side-process; this panel shows current live values each refresh.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st


def _gpu_stats() -> dict | None:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        return {
            "name": name,
            "vram_used_gb": mem.used / 1e9,
            "vram_total_gb": mem.total / 1e9,
            "gpu_util_pct": util.gpu,
            "mem_util_pct": util.memory,
            "temp_c": temp,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _sys_stats(disk_paths: list[Path]) -> dict:
    try:
        import psutil  # type: ignore
    except ImportError:
        return {}
    vm = psutil.virtual_memory()
    out = {
        "cpu_pct": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": vm.used / 1e9,
        "ram_total_gb": vm.total / 1e9,
        "ram_pct": vm.percent,
    }
    for p in disk_paths:
        try:
            usage = shutil.disk_usage(p)
            out[f"disk_{p}_free_gb"] = usage.free / 1e9
            out[f"disk_{p}_total_gb"] = usage.total / 1e9
        except OSError:
            pass
    return out


def render(disk_paths: list[Path]) -> None:
    st.subheader("Hardware")

    gpu = _gpu_stats()
    sys = _sys_stats(disk_paths)

    g1, g2, g3 = st.columns(3)
    if gpu is None:
        g1.warning("pynvml not installed")
    elif "error" in gpu:
        g1.warning(f"NVML error: {gpu['error']}")
    else:
        g1.metric(
            f"GPU: {gpu['name']}",
            f"{gpu['vram_used_gb']:.2f} / {gpu['vram_total_gb']:.1f} GB",
            f"util {gpu['gpu_util_pct']}% | {gpu['temp_c']}°C" if gpu['temp_c'] else f"util {gpu['gpu_util_pct']}%",
        )

    g2.metric("CPU", f"{sys.get('cpu_pct', 0):.0f} %")
    g3.metric(
        "RAM",
        f"{sys.get('ram_used_gb', 0):.1f} / {sys.get('ram_total_gb', 0):.1f} GB",
        f"{sys.get('ram_pct', 0):.0f}%",
    )

    if disk_paths:
        cols = st.columns(len(disk_paths))
        for col, p in zip(cols, disk_paths):
            free = sys.get(f"disk_{p}_free_gb")
            total = sys.get(f"disk_{p}_total_gb")
            if free is not None and total is not None:
                col.metric(f"Disk {p}",
                           f"{free:.1f} GB free",
                           f"of {total:.0f} GB")
            else:
                col.caption(f"{p}: n/a")
