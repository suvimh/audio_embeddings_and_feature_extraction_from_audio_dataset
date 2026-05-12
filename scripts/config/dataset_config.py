from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

# ---------------------------------------------------------------------------
# Dataset Configuration to specify the input dataset
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """
    Describes how to interpret a dataset's folder structure.

    Parameters
    ----------
    name : str
        Human-readable dataset name (used in output filenames).
    root_dir : str | Path
        Root directory of the dataset.
    level_names : list[str]
        Labels for each folder depth level BELOW root_dir.
        e.g. ['experience', 'speaker', 'phonation', 'condition', 'scale', 'take']
    participant_level : int
        Which folder depth (0-indexed from root) contains the participant/speaker folder.
    subgroup_configs : dict[str, list[str]] | None
        If folder structure differs by subgroup, provide per-subgroup level_names.
        Key: value at `subgroup_split_level` folder, Value: list of level_names for that subgroup.
    subgroup_split_level : int | None
        The folder depth at which to look up subgroup_configs.
    level_allowed_values : dict[str, list[str]] | None
        Optional. Documents the known/expected values for each level.
        Used for validation + readable reference. Does NOT filter files.
        e.g. {'phonation': ['undefined', 'breathy'],
               'condition': ['arched_back', 'chin_up', ...]}
    file_suffix_filter : str | None
        If set, only process files whose stem ends with this string.
        e.g. '-mic-audio' matches 'inex-4-...-mic-audio.wav' only.
        Case-insensitive. None = process all files matching audio_extensions.
    audio_extensions : list[str]
        Audio file extensions to collect.
    skip_prefixes : list[str]
        Skip files/folders starting with these strings.
    sample_rate : int
        Target sample rate (VGGish requires 16000).
    frame_duration : float
        Frame duration in seconds.
    overlap_percentage : float
        Overlap between frames (0.0-1.0).
    enable_silence_trimming : bool
        If True, trim silence from edges and interior of audio before framing.
    silence_threshold_db : float
        Silence threshold in dB (relative to RMS). Only used if enable_silence_trimming=True.
        Typical values: -40 to -20 dB. More negative = more aggressive.
    silence_min_duration_ms : float
        Minimum duration of silence (in ms) to be trimmed. Prevents clipping short pauses.
        Only used if enable_silence_trimming=True.
    normalization_mode : str
        Audio normalization strategy: 'peak' (0dB), 'rms', 'lufs', or 'none'.
        Applied to each frame independently.
    """

    name: str
    root_dir: str | Path
    level_names: list[str]
    participant_level: int = 1
    subgroup_configs: Optional[dict[str, list[str]]] = None
    subgroup_split_level: Optional[int] = None
    level_allowed_values: Optional[dict[str, list[str]]] = None
    file_suffix_filter: Optional[str] = None
    audio_extensions: list[str] = field(default_factory=lambda: [".wav"])
    skip_prefixes: list[str] = field(default_factory=lambda: ["._", ".DS"])
    sample_rate: int = 16000
    frame_duration: float = 3.0
    overlap_percentage: float = 0.25
    enable_silence_trimming: bool = True
    silence_threshold_db: float = -40.0
    silence_min_duration_ms: float = 250.0
    normalization_mode: str = "none"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_normalization_modes = {"peak", "rms", "lufs", "none"}
        if self.normalization_mode not in valid_normalization_modes:
            raise ValueError(
                f"normalization_mode must be one of {valid_normalization_modes}, "
                f"got '{self.normalization_mode}'"
            )
        if not 0 <= self.overlap_percentage < 1:
            raise ValueError(
                f"overlap_percentage must be in [0, 1), got {self.overlap_percentage}"
            )
        if self.silence_threshold_db > 0:
            raise ValueError(
                f"silence_threshold_db should be negative (dB), got {self.silence_threshold_db}"
            )
        if self.silence_min_duration_ms < 0:
            raise ValueError(
                f"silence_min_duration_ms must be non-negative, got {self.silence_min_duration_ms}"
            )

    def matches_file_filter(self, file_path: Path) -> bool:
        """
        Returns True if this file should be processed.
        Checks extension and optional suffix filter.
        """
        if file_path.suffix.lower() not in self.audio_extensions:
            return False
        if self.file_suffix_filter is not None:
            if not file_path.stem.lower().endswith(self.file_suffix_filter.lower()):
                return False
        return True

    def level_names_for_path(self, relative_parts: list[str]) -> dict[str, str]:
        """
        Given the folder parts of a file path (relative to root_dir),
        return a dict mapping level_name -> folder value.

        Handles irregular depth by checking subgroup_configs if set.
        Missing levels are filled with None.
        """
        if self.subgroup_configs and self.subgroup_split_level is not None:
            try:
                subgroup_key = relative_parts[self.subgroup_split_level]
                names = self.subgroup_configs.get(subgroup_key, self.level_names)
            except IndexError:
                names = self.level_names
        else:
            names = self.level_names

        result = {}
        for i, name in enumerate(names):
            try:
                result[name] = relative_parts[i]
            except IndexError:
                result[name] = None
        return result

    def validate_metadata(self, metadata: dict) -> list[str]:
        """
        Check extracted metadata against level_allowed_values.
        Returns a list of warning strings (empty = all good).
        Only runs if level_allowed_values is set.
        """
        if not self.level_allowed_values:
            return []
        warnings = []
        for level, allowed in self.level_allowed_values.items():
            val = metadata.get(level)
            if val is not None and val not in allowed:
                warnings.append(
                    f"Unexpected value '{val}' at level '{level}' (known values: {allowed})"
                )
        return warnings

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["root_dir"] = str(data["root_dir"])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetConfig":
        """Create from dictionary."""
        return cls(**data)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        data = self.to_dict()

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "DatasetConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def save_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "DatasetConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)
