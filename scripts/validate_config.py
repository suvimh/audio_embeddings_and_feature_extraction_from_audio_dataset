"""
Configuration Validation Utilities
===================================
Pre-flight checks and validation for extraction pipeline configs.
"""

from pathlib import Path
from typing import List, Tuple

from scripts.config.dataset_config import DatasetConfig
from scripts.config.extraction_config import ExtractionConfig
from scripts.extractors_registry import get_extractor_info


def validate_dataset_config(config: DatasetConfig) -> Tuple[bool, List[str]]:
    """
    Validate a dataset config.

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration to validate.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_warnings_and_errors)
    """
    issues = []

    # Check root directory exists
    root_dir = Path(config.root_dir)
    if not root_dir.exists():
        issues.append(f"ERROR: Dataset root directory does not exist: {root_dir}")
        return False, issues

    if not root_dir.is_dir():
        issues.append(f"ERROR: Dataset root is not a directory: {root_dir}")
        return False, issues

    # Check for audio files
    audio_files = list(root_dir.rglob("*"))
    audio_files = [
        f
        for f in audio_files
        if f.is_file() and f.suffix.lower() in config.audio_extensions
    ]

    if not audio_files:
        issues.append(
            f"WARNING: No audio files found with extensions {config.audio_extensions}"
        )

    # Validate parameters
    if config.frame_duration <= 0:
        issues.append(f"ERROR: frame_duration must be > 0, got {config.frame_duration}")

    if config.sample_rate <= 0:
        issues.append(f"ERROR: sample_rate must be > 0, got {config.sample_rate}")

    if not (0 <= config.overlap_percentage < 1):
        issues.append(
            f"ERROR: overlap_percentage must be in [0, 1), got {config.overlap_percentage}"
        )

    if config.normalization_mode not in {"peak", "rms", "lufs", "none"}:
        issues.append(f"ERROR: Invalid normalization_mode: {config.normalization_mode}")

    if config.enable_silence_trimming:
        if config.silence_threshold_db > 0:
            issues.append(
                f"WARNING: silence_threshold_db is positive (expected negative): {config.silence_threshold_db}"
            )
        if config.silence_min_duration_ms < 0:
            issues.append(
                f"ERROR: silence_min_duration_ms must be >= 0, got {config.silence_min_duration_ms}"
            )

    return len([e for e in issues if e.startswith("ERROR")]) == 0, issues


def validate_extraction_config(config: ExtractionConfig) -> Tuple[bool, List[str]]:
    """
    Validate an extraction config.

    Parameters
    ----------
    config : ExtractionConfig
        Extraction configuration to validate.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_warnings_and_errors)
    """
    issues = []

    # Check extractors
    for extractor in config.extractors:
        try:
            get_extractor_info(extractor)
        except ValueError:
            issues.append(f"ERROR: Unknown extractor: {extractor}")

    if not config.extractors:
        issues.append("ERROR: No extractors specified")

    # Check window lengths
    if not config.window_lengths:
        issues.append("ERROR: No window_lengths specified")

    for wl in config.window_lengths:
        if wl <= 0:
            issues.append(f"ERROR: window_length must be > 0, got {wl}")

    # Check batch size
    if isinstance(config.batch_size, int) and config.batch_size < 1:
        issues.append(
            f"ERROR: batch_size must be >= 1 or 'auto', got {config.batch_size}"
        )

    # Check output directory is writable
    output_dir = Path(config.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Try to write a test file
        test_file = output_dir / ".write_test"
        test_file.write_text("")
        test_file.unlink()
    except Exception as e:
        issues.append(f"ERROR: Output directory not writable: {e}")

    return len([e for e in issues if e.startswith("ERROR")]) == 0, issues


def validate_configs(
    dataset_config: DatasetConfig, extraction_config: ExtractionConfig
) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of both configs together.

    Parameters
    ----------
    dataset_config : DatasetConfig
        Dataset configuration.
    extraction_config : ExtractionConfig
        Extraction configuration.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_warnings_and_errors)
    """
    issues = []

    # Validate individual configs
    dataset_valid, dataset_issues = validate_dataset_config(dataset_config)
    issues.extend(dataset_issues)

    extraction_valid, extraction_issues = validate_extraction_config(extraction_config)
    issues.extend(extraction_issues)

    # Cross-config checks
    min_expected_samples = int(
        min(extraction_config.window_lengths) * dataset_config.sample_rate
    )
    if min_expected_samples < 1000:
        issues.append(
            f"WARNING: Minimum expected frame size ({min_expected_samples} samples) is very short. "
            f"Some extractors may fail."
        )

    is_valid = dataset_valid and extraction_valid
    return is_valid, issues
