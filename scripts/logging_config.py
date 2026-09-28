"""
Structured JSON-Lines Logging
=============================
Provides JSON-lines format logging for extraction pipeline with resource monitoring.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict
import sys


class JSONLineFormatter(logging.Formatter):
    """Custom formatter that outputs JSON-lines format."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert log record to JSON-lines format."""
        log_dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add custom attributes if present
        if hasattr(record, "stage"):
            log_dict["stage"] = record.stage
        if hasattr(record, "extractor"):
            log_dict["extractor"] = record.extractor
        if hasattr(record, "window_length"):
            log_dict["window_length"] = record.window_length
        if hasattr(record, "file_path"):
            log_dict["file_path"] = record.file_path
        if hasattr(record, "files_processed"):
            log_dict["files_processed"] = record.files_processed
        if hasattr(record, "frames_processed"):
            log_dict["frames_processed"] = record.frames_processed
        if hasattr(record, "errors"):
            log_dict["errors"] = record.errors
        if hasattr(record, "warnings"):
            log_dict["warnings"] = record.warnings
        if hasattr(record, "resource_cpu_percent"):
            log_dict["resource_cpu_percent"] = record.resource_cpu_percent
        if hasattr(record, "resource_ram_used_gb"):
            log_dict["resource_ram_used_gb"] = record.resource_ram_used_gb
        if hasattr(record, "resource_ram_available_gb"):
            log_dict["resource_ram_available_gb"] = record.resource_ram_available_gb
        if hasattr(record, "resource_gpu_vram_used_gb"):
            log_dict["resource_gpu_vram_used_gb"] = record.resource_gpu_vram_used_gb
        if hasattr(record, "resource_gpu_vram_available_gb"):
            log_dict["resource_gpu_vram_available_gb"] = (
                record.resource_gpu_vram_available_gb
            )
        if hasattr(record, "exc_info"):
            if record.exc_info:
                log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict)


class ExtractionLogger:
    """Structured logger for extraction pipeline."""

    def __init__(self, log_file: Optional[str | Path] = None, console: bool = True):
        """
        Initialize extraction logger.

        Parameters
        ----------
        log_file : str | Path | None
            Path to write JSON-lines log file. If None, only console output.
        console : bool
            If True, also log to console.
        """
        self.logger = logging.getLogger("audio_extraction")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear any existing handlers

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(JSONLineFormatter())
            self.logger.addHandler(file_handler)

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(JSONLineFormatter())
            self.logger.addHandler(console_handler)

        self.log_file = log_file

    def log_extraction_start(
        self, extractor: str, window_length: float, dataset_name: str
    ) -> None:
        """Log start of extraction for a specific (extractor, window_length) pair."""
        record = self.logger.makeRecord(
            name="audio_extraction",
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"Starting extraction: {extractor} @ {window_length}s for {dataset_name}",
            args=(),
            exc_info=None,
        )
        record.stage = "extraction_start"
        record.extractor = extractor
        record.window_length = window_length
        self.logger.handle(record)

    def log_extraction_complete(
        self,
        extractor: str,
        window_length: float,
        files_processed: int,
        frames_processed: int,
        errors: int,
        warnings: int,
        duration_sec: float,
    ) -> None:
        """Log completion of extraction."""
        record = self.logger.makeRecord(
            name="audio_extraction",
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"Extraction complete: {extractor} @ {window_length}s | {files_processed} files, {frames_processed} frames | {errors} errors, {warnings} warnings | {duration_sec:.1f}s",
            args=(),
            exc_info=None,
        )
        record.stage = "extraction_complete"
        record.extractor = extractor
        record.window_length = window_length
        record.files_processed = files_processed
        record.frames_processed = frames_processed
        record.errors = errors
        record.warnings = warnings
        self.logger.handle(record)

    def log_file_processed(
        self, file_path: str, frames: int, extractor: str, window_length: float
    ) -> None:
        """Log successful processing of a file."""
        record = self.logger.makeRecord(
            name="audio_extraction",
            level=logging.DEBUG,
            fn="",
            lno=0,
            msg=f"File processed: {Path(file_path).name} ({frames} frames)",
            args=(),
            exc_info=None,
        )
        record.stage = "file_processed"
        record.file_path = str(file_path)
        record.frames_processed = frames
        record.extractor = extractor
        record.window_length = window_length
        self.logger.handle(record)

    def log_file_error(
        self, file_path: str, error: str, extractor: str, window_length: float
    ) -> None:
        """Log error processing a file (non-fatal)."""
        record = self.logger.makeRecord(
            name="audio_extraction",
            level=logging.WARNING,
            fn="",
            lno=0,
            msg=f"Failed to extract {Path(file_path).name}: {error}",
            args=(),
            exc_info=None,
        )
        record.stage = "file_error"
        record.file_path = str(file_path)
        record.extractor = extractor
        record.window_length = window_length
        self.logger.handle(record)

    def log_resource_checkpoint(
        self,
        extractor: str,
        window_length: float,
        cpu_percent: float,
        ram_used_gb: float,
        ram_available_gb: float,
        gpu_vram_used_gb: Optional[float] = None,
        gpu_vram_available_gb: Optional[float] = None,
    ) -> None:
        """Log resource usage snapshot."""
        record = self.logger.makeRecord(
            name="audio_extraction",
            level=logging.DEBUG,
            fn="",
            lno=0,
            msg=f"Resource checkpoint | CPU: {cpu_percent}% | RAM: {ram_used_gb:.1f}GB used, {ram_available_gb:.1f}GB available"
            + (
                f" | GPU: {gpu_vram_used_gb:.1f}GB used, {gpu_vram_available_gb:.1f}GB available"
                if gpu_vram_used_gb
                else ""
            ),
            args=(),
            exc_info=None,
        )
        record.stage = "resource_checkpoint"
        record.extractor = extractor
        record.window_length = window_length
        record.resource_cpu_percent = cpu_percent
        record.resource_ram_used_gb = ram_used_gb
        record.resource_ram_available_gb = ram_available_gb
        if gpu_vram_used_gb is not None:
            record.resource_gpu_vram_used_gb = gpu_vram_used_gb
        if gpu_vram_available_gb is not None:
            record.resource_gpu_vram_available_gb = gpu_vram_available_gb
        self.logger.handle(record)

    def log_info(self, message: str, **kwargs) -> None:
        """Log info message with optional custom fields."""
        record = self.logger.makeRecord(
            name="audio_extraction",
            level=logging.INFO,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        for key, value in kwargs.items():
            setattr(record, key, value)
        self.logger.handle(record)

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional custom fields."""
        record = self.logger.makeRecord(
            name="audio_extraction",
            level=logging.WARNING,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        for key, value in kwargs.items():
            setattr(record, key, value)
        self.logger.handle(record)

    def log_error(self, message: str, **kwargs) -> None:
        """Log error message with optional custom fields."""
        record = self.logger.makeRecord(
            name="audio_extraction",
            level=logging.ERROR,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        for key, value in kwargs.items():
            setattr(record, key, value)
        self.logger.handle(record)
