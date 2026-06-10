"""
Main Extraction Orchestrator
=============================
Coordinates the extraction pipeline: loads configs, instantiates extractors,
manages window lengths, and orchestrates extraction execution.
"""

import time
from pathlib import Path
from dataclasses import replace
from typing import Optional
import importlib.util

from scripts.config.dataset_config import DatasetConfig
from scripts.config.extraction_config import ExtractionConfig, get_batch_size
from scripts.extractors_registry import make_extractor
from scripts.logging_config import ExtractionLogger
from scripts.resource_monitor import get_system_resources
from scripts.utils import extract_dataset_features


def load_config_from_python(file_path: str | Path) -> object:
    """
    Load a configuration from a Python file by importing the first class/instance found.

    Parameters
    ----------
    file_path : str | Path
        Path to Python file containing config class or instance.

    Returns
    -------
    object
        Loaded configuration object (DatasetConfig or ExtractionConfig).

    Raises
    ------
    ValueError
        If no valid config found in file.
    """
    file_path = Path(file_path)
    spec = importlib.util.spec_from_file_location("config_module", file_path)
    if not spec or not spec.loader:
        raise ValueError(f"Cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Look for DatasetConfig or ExtractionConfig instances/classes
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)

        # Check if it's an instance of a config class
        if isinstance(attr, (DatasetConfig, ExtractionConfig)):
            return attr

    raise ValueError(f"No DatasetConfig or ExtractionConfig found in {file_path}")


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    """
    Load dataset configuration from file (auto-detect format).

    Parameters
    ----------
    config_path : str | Path
        Path to config file (.py or .yml).

    Returns
    -------
    DatasetConfig
        Loaded configuration.
    """
    config_path = Path(config_path)

    if config_path.suffix.lower() in {".yml", ".yaml"}:
        return DatasetConfig.load_yaml(config_path)
    elif config_path.suffix.lower() == ".py":
        return load_config_from_python(config_path)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def load_extraction_config(config_path: str | Path) -> ExtractionConfig:
    """
    Load extraction configuration from file (auto-detect format).

    Parameters
    ----------
    config_path : str | Path
        Path to config file (.py or .yml).

    Returns
    -------
    ExtractionConfig
        Loaded configuration.
    """
    config_path = Path(config_path)

    if config_path.suffix.lower() in {".yml", ".yaml"}:
        return ExtractionConfig.load_yaml(config_path)
    elif config_path.suffix.lower() == ".py":
        config_obj = load_config_from_python(config_path)
        if not isinstance(config_obj, ExtractionConfig):
            raise ValueError(f"Expected ExtractionConfig, got {type(config_obj)}")
        return config_obj
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def run_extraction(
    dataset_config_path: str | Path,
    extraction_config_path: str | Path,
    log_file: Optional[str | Path] = None,
    console_output: bool = True,
    overwrite: bool = False,
) -> dict:
    """
    Run the full extraction pipeline.

    Parameters
    ----------
    dataset_config_path : str | Path
        Path to dataset configuration file.
    extraction_config_path : str | Path
        Path to extraction configuration file.
    log_file : str | Path | None
        Path to write JSON-lines log file. If None, uses timestamp-based name in output_dir.
    console_output : bool
        If True, also print logs to console.

    Returns
    -------
    dict
        Summary report with keys: total_files, total_frames, errors, warnings, duration_sec, output_files.

    Raises
    ------
    ValueError
        If configurations are invalid or incompatible.
    """
    # Load configurations
    dataset_config = load_dataset_config(dataset_config_path)
    extraction_config = load_extraction_config(extraction_config_path)

    # Setup logging
    if log_file is None:
        output_dir = Path(extraction_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"extraction_{timestamp}.jsonl"

    logger = ExtractionLogger(log_file=log_file, console=console_output)

    # Resolve batch size if auto
    batch_size = get_batch_size(extraction_config)
    logger.log_info(
        f"Resolved batch_size: {batch_size} files",
        stage="initialization",
    )

    # Create output directory
    output_dir = Path(extraction_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.log_info(
        f"Extraction pipeline starting",
        stage="initialization",
        dataset=dataset_config.name,
        extractors=",".join(extraction_config.extractors),
        window_lengths=",".join(str(w) for w in extraction_config.window_lengths),
    )

    # Summary trackers
    total_files_processed = 0
    total_frames_processed = 0
    total_errors = 0
    total_warnings = 0
    output_files = []
    start_time = time.time()

    # Process each window length
    for window_length in extraction_config.window_lengths:
        # Create dataset config with this window length
        dataset_config_for_window = replace(
            dataset_config, frame_duration=window_length
        )

        # Process each extractor
        for extractor_name in extraction_config.extractors:
            try:
                logger.log_extraction_start(
                    extractor=extractor_name,
                    window_length=window_length,
                    dataset_name=dataset_config.name,
                )

                # Instantiate extractor
                extractor_params = extraction_config.extractor_params.get(
                    extractor_name, {}
                )
                extractor_config = make_extractor(
                    name=extractor_name,
                    sample_rate=dataset_config.sample_rate,
                    extractor_params=extractor_params,
                )

                # Run extraction
                extraction_start = time.time()

                try:
                    df = extract_dataset_features(
                        dataset_config=dataset_config_for_window,
                        extractor_config=extractor_config,
                        output_dir=output_dir,
                        checkpoint_dir=output_dir,
                        overwrite=overwrite,
                        batch_size=batch_size,
                    )

                    files_processed = df["filepath"].nunique()
                    frames_processed = len(df)
                    total_files_processed += files_processed
                    total_frames_processed += frames_processed

                    extraction_time = time.time() - extraction_start

                    # Log completion
                    logger.log_extraction_complete(
                        extractor=extractor_name,
                        window_length=window_length,
                        files_processed=files_processed,
                        frames_processed=frames_processed,
                        errors=0,
                        warnings=0,
                        duration_sec=extraction_time,
                    )

                    # Track output file
                    output_filename = (
                        f"{extractor_name}_window_{window_length}s.parquet"
                    )
                    output_file = output_dir / output_filename
                    output_files.append(str(output_file))

                    # Log resource checkpoint
                    resources = get_system_resources()
                    logger.log_resource_checkpoint(
                        extractor=extractor_name,
                        window_length=window_length,
                        cpu_percent=resources.cpu_percent,
                        ram_used_gb=resources.ram_total_gb - resources.ram_available_gb,
                        ram_available_gb=resources.ram_available_gb,
                        gpu_vram_used_gb=(
                            resources.gpu_vram_total_gb
                            - resources.gpu_vram_available_gb
                            if resources.gpu_vram_total_gb
                            else None
                        ),
                        gpu_vram_available_gb=resources.gpu_vram_available_gb,
                    )

                except Exception as e:
                    total_errors += 1
                    logger.log_error(
                        f"Extraction failed for {extractor_name} @ {window_length}s: {str(e)}",
                        stage="extraction_error",
                        extractor=extractor_name,
                        window_length=window_length,
                    )

            except Exception as e:
                total_errors += 1
                logger.log_error(
                    f"Failed to instantiate extractor {extractor_name}: {str(e)}",
                    stage="extractor_instantiation_error",
                    extractor=extractor_name,
                )

    # Final summary
    total_time = time.time() - start_time
    summary = {
        "total_files_processed": total_files_processed,
        "total_frames_processed": total_frames_processed,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "duration_sec": total_time,
        "output_files": output_files,
        "log_file": str(log_file),
    }

    logger.log_info(
        f"Extraction pipeline complete | {total_files_processed} files, {total_frames_processed} frames | "
        f"{total_errors} errors | {total_time:.1f}s",
        stage="pipeline_complete",
        **summary,
    )

    return summary
