"""Hardware probe — mirrors Hypura's `HardwareProfile` but for PyTorch/Windows.

The profile drives `placement.plan_placement` and is also emitted as telemetry
at training start so the dashboard can show the configuration that was used.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class HardwareProfile:
    gpu_name: str
    gpu_vram_bytes: int
    gpu_bandwidth_gbps: float        # approximate, for planning — not measured
    ram_bytes: int
    ram_bandwidth_gbps: float        # approximate
    nvme_path: Path
    nvme_free_bytes: int
    nvme_bandwidth_mbps: float       # measured read bandwidth of a 16 MB probe

    def to_dict(self) -> dict:
        d = asdict(self)
        d["nvme_path"] = str(self.nvme_path)
        return d


def _probe_nvme_bandwidth_mbps(tmp_dir: Path, size_mb: int = 16) -> float:
    """One-shot write+read probe on the target drive. Small (16 MB) so probe is
    cheap; result is a coarse estimate good enough for placement planning.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    probe = tmp_dir / ".nvme_probe.bin"
    data = b"\xA5" * (size_mb * 1024 * 1024)
    try:
        with open(probe, "wb", buffering=0) as f:
            f.write(data)
            f.flush()
        t0 = time.perf_counter()
        with open(probe, "rb", buffering=0) as f:
            # Chunked readinto to simulate the prefetcher's own IO pattern.
            chunk = bytearray(1 * 1024 * 1024)
            got = 0
            while True:
                n = f.readinto(chunk)
                if not n:
                    break
                got += n
        elapsed = max(1e-6, time.perf_counter() - t0)
        return (got / 1e6) / elapsed
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def probe_hardware(nvme_path: Path | str = "H:/elt_data/offload_scratch") -> HardwareProfile:
    """Build a HardwareProfile for the current machine.

    `nvme_path` should point at a writable directory on the target offload
    drive; the directory is created if it doesn't exist.
    """
    nvme_path = Path(nvme_path)

    gpu_name = "cpu"
    gpu_vram = 0
    gpu_bw = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_vram = props.total_memory
        # very coarse estimate for compute budget planning
        gpu_bw = 360.0 if "3060" in gpu_name else 500.0

    try:
        import psutil
        ram_bytes = psutil.virtual_memory().total
    except ImportError:
        # psutil is optional; fall back to a conservative 16 GB guess.
        ram_bytes = 16 * (1024 ** 3)

    free_bytes = shutil.disk_usage(str(nvme_path.drive) + "\\"
                                   if nvme_path.drive else str(nvme_path.anchor)).free \
        if nvme_path.anchor else shutil.disk_usage(str(nvme_path)).free

    nvme_mbps = _probe_nvme_bandwidth_mbps(nvme_path)

    return HardwareProfile(
        gpu_name=gpu_name,
        gpu_vram_bytes=gpu_vram,
        gpu_bandwidth_gbps=gpu_bw,
        ram_bytes=ram_bytes,
        ram_bandwidth_gbps=50.0,     # DDR4 dual channel ~50 GB/s typical
        nvme_path=nvme_path,
        nvme_free_bytes=free_bytes,
        nvme_bandwidth_mbps=nvme_mbps,
    )
