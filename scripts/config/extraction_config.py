"""
Extraction Configuration
========================
Specifies which embeddings and features to extract, window lengths, and output settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


@dataclass
class ExtractionConfig:
    """
    Configuration for the extraction pipeline.

    Parameters
    ----------
    extractors : list[str]
        List of extractor names to run. Valid names: 'vggish', 'clap', 'whisper', 'opensmile'.
        Examples: ['vggish', 'clap'], ['whisper', 'opensmile']

    extractor_params : dict[str, dict]
        Optional per-extractor parameters. Overrides defaults for specific extractors.
        Example: {
            'clap': {'version': '2024'},
            'whisper': {'model_name': 'base'}
        }

    window_lengths : list[float]
        List of frame durations (in seconds) to extract for.
        The dataset config's frame_duration will be overridden for each value.
        Example: [3.0, 5.0] → extracts embeddings with 3s and 5s frames

    output_dir : str | Path
        Parent directory for all output parquet files.
        Files are named: {extractor_name}_window_{duration}s.parquet

    batch_size : int | str
        Number of files to process before flushing to HDF5.
        - Positive int: Use this batch size explicitly
        - 'auto': Calculate from available RAM (recommended)
        Default: 'auto'

    gpu_cache_cleanup : bool
        If True, empty GPU cache between batches (slower but prevents fragmentation).
        If False (default), keep models in GPU memory for speed.

    num_workers : int
        Number of thread workers for parallel file I/O (0 = sequential).
        Default: 4 (recommended for most systems)
    """

    extractors: List[str]
    output_dir: str | Path
    window_lengths: List[float] = field(default_factory=lambda: [3.0])
    extractor_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    batch_size: int | str = "auto"
    gpu_cache_cleanup: bool = False
    num_workers: int = 4

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_extractors = {"vggish", "clap", "whisper", "opensmile"}

        for extractor in self.extractors:
            if extractor not in valid_extractors:
                raise ValueError(
                    f"Invalid extractor '{extractor}'. Must be one of {valid_extractors}"
                )

        if not self.extractors:
            raise ValueError("At least one extractor must be specified")

        if not self.window_lengths or any(wl <= 0 for wl in self.window_lengths):
            raise ValueError("window_lengths must be non-empty and all values > 0")

        if isinstance(self.batch_size, int) and self.batch_size < 1:
            raise ValueError(
                f"batch_size must be 'auto' or a positive int, got {self.batch_size}"
            )

        if isinstance(self.batch_size, str) and self.batch_size != "auto":
            raise ValueError(
                f"batch_size must be 'auto' or a positive int, got {self.batch_size}"
            )

        if self.num_workers < 0 or self.num_workers > 32:
            raise ValueError(f"num_workers must be in [0, 32], got {self.num_workers}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionConfig":
        """Create from dictionary."""
        return cls(**data)

    def to_yaml_dict(self) -> dict:
        """Convert to YAML-friendly dictionary."""
        return self.to_dict()

    @classmethod
    def from_yaml_dict(cls, data: dict) -> "ExtractionConfig":
        """Create from YAML dictionary."""
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        data = self.to_yaml_dict()
        data["output_dir"] = str(data["output_dir"])

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "ExtractionConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_yaml_dict(data)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        data = self.to_dict()
        data["output_dir"] = str(data["output_dir"])

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExtractionConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)


def get_batch_size(config: ExtractionConfig) -> int:
    """
    Resolve batch_size from extraction config.
    If 'auto', calculate from available RAM.

    Parameters
    ----------
    config : ExtractionConfig
        Extraction configuration.

    Returns
    -------
    int
        Resolved batch size in files.
    """
    if isinstance(config.batch_size, int):
        return config.batch_size

    # Auto-calculate
    from scripts.resource_monitor import calculate_batch_size

    return calculate_batch_size()
