"""
Resource Monitoring Utilities
==============================
Detect available CPU, RAM, and GPU memory for intelligent batch sizing.
"""

from __future__ import annotations

import psutil
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemResources:
    """System resource availability snapshot."""
    cpu_percent: float
    ram_available_gb: float
    ram_total_gb: float
    gpu_vram_available_gb: Optional[float] = None
    gpu_vram_total_gb: Optional[float] = None

    def ram_available_percent(self) -> float:
        """Percentage of RAM available (0-100)."""
        if self.ram_total_gb == 0:
            return 0.0
        return (self.ram_available_gb / self.ram_total_gb) * 100

    def gpu_available_percent(self) -> float:
        """Percentage of GPU VRAM available (0-100)."""
        if self.gpu_vram_total_gb is None or self.gpu_vram_total_gb == 0:
            return 0.0
        return (self.gpu_vram_available_gb / self.gpu_vram_total_gb) * 100


def get_system_resources() -> SystemResources:
    """
    Detect current system resources: CPU, RAM, and optional GPU VRAM.

    Returns
    -------
    SystemResources
        Current resource availability.
    """
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # RAM usage
    vm = psutil.virtual_memory()
    ram_available_gb = vm.available / (1024 ** 3)
    ram_total_gb = vm.total / (1024 ** 3)

    # GPU VRAM (try nvidia-smi if available)
    gpu_vram_available_gb = None
    gpu_vram_total_gb = None

    try:
        import torch
        if torch.cuda.is_available():
            # NVIDIA GPU
            device_idx = 0  # Default to first GPU
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
            available = total_memory - allocated

            gpu_vram_total_gb = total_memory
            gpu_vram_available_gb = available
    except Exception:
        pass

    return SystemResources(
        cpu_percent=cpu_percent,
        ram_available_gb=ram_available_gb,
        ram_total_gb=ram_total_gb,
        gpu_vram_available_gb=gpu_vram_available_gb,
        gpu_vram_total_gb=gpu_vram_total_gb,
    )


def calculate_batch_size(
    frame_size_mb: float = 0.004,  # 4KB per frame
    embedding_size_mb: float = 0.002,  # 2KB per embedding (CLAP 512-dim)
    avg_frames_per_file: int = 100,
    ram_reserve_percent: float = 0.3,
) -> int:
    """
    Auto-calculate optimal batch size (files per batch) based on available RAM.

    Parameters
    ----------
    frame_size_mb : float
        Estimated size of each audio frame in MB.
    embedding_size_mb : float
        Estimated size of each embedding in MB.
    avg_frames_per_file : int
        Average number of frames per audio file.
    ram_reserve_percent : float
        Reserve this fraction of available RAM for system/other processes (0.0-1.0).

    Returns
    -------
    int
        Recommended batch size (number of files to process before flushing to HDF5).
        Minimum 1, maximum 100.
    """
    resources = get_system_resources()

    # Usable RAM (reserve some for system)
    usable_ram_gb = resources.ram_available_gb * (1 - ram_reserve_percent)
    usable_ram_mb = usable_ram_gb * 1024

    # Memory per file (frames + embeddings)
    memory_per_file_mb = (frame_size_mb * avg_frames_per_file) + (embedding_size_mb * avg_frames_per_file)

    if memory_per_file_mb <= 0:
        return 1

    batch_size = max(1, min(100, int(usable_ram_mb / memory_per_file_mb)))

    return batch_size


def check_memory_pressure(threshold_percent: float = 85.0) -> dict[str, float | bool]:
    """
    Check if system is under memory pressure.

    Parameters
    ----------
    threshold_percent : float
        Memory usage above this % is considered pressure (0-100).

    Returns
    -------
    dict
        Keys: 'under_pressure' (bool), 'ram_usage_percent' (float), 'gpu_usage_percent' (float or None)
    """
    resources = get_system_resources()
    ram_usage_percent = 100 - resources.ram_available_percent()

    result = {
        "under_pressure": ram_usage_percent > threshold_percent,
        "ram_usage_percent": ram_usage_percent,
        "gpu_usage_percent": (100 - resources.gpu_available_percent()) if resources.gpu_vram_total_gb else None,
    }

    return result
