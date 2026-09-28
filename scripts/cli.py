"""
Command-line interface for the audio extraction pipeline.
"""

import argparse
import sys
from pathlib import Path

from scripts.run_extraction import run_extraction
from scripts.extractors_registry import list_extractors


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="audio_extraction",
        description="Extract audio embeddings and features from datasets using modular extractors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction run
  python -m audio_extraction extract \\
    --dataset-config configs/dataset.py \\
    --extraction-config configs/extraction.py

  # With custom log file
  python -m audio_extraction extract \\
    --dataset-config configs/dataset.py \\
    --extraction-config configs/extraction.py \\
    --log-file logs/extraction_run.jsonl

  # Validate configs before running
  python -m audio_extraction validate \\
    --dataset-config configs/dataset.py \\
    --extraction-config configs/extraction.py

  # List available extractors
  python -m audio_extraction list-extractors

Available extractors:
  - vggish: VGGish embeddings (128-dim)
  - clap: MS-CLAP audio-text embeddings (512/1024-dim)
  - whisper: Whisper encoder embeddings (384-1280-dim)
  - opensmile: Traditional acoustic features (62-6373-dim)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Run extraction pipeline",
    )
    extract_parser.add_argument(
        "--dataset-config",
        required=True,
        type=str,
        help="Path to dataset config file (.py or .yml)",
    )
    extract_parser.add_argument(
        "--extraction-config",
        required=True,
        type=str,
        help="Path to extraction config file (.py or .yml)",
    )
    extract_parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Output JSON-lines log file (default: auto-generated in output_dir)",
    )
    extract_parser.add_argument(
        "--no-console",
        action="store_true",
        help="Suppress console output (only write to log file)",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate dataset and extraction configs",
    )
    validate_parser.add_argument(
        "--dataset-config",
        required=True,
        type=str,
        help="Path to dataset config file (.py or .yml)",
    )
    validate_parser.add_argument(
        "--extraction-config",
        required=True,
        type=str,
        help="Path to extraction config file (.py or .yml)",
    )

    # List extractors command
    list_parser = subparsers.add_parser(
        "list-extractors",
        help="List available extractors with descriptions",
    )

    args = parser.parse_args()

    if args.command == "extract":
        try:
            console_output = not args.no_console
            summary = run_extraction(
                dataset_config_path=args.dataset_config,
                extraction_config_path=args.extraction_config,
                log_file=args.log_file,
                console_output=console_output,
            )
            print("\n" + "=" * 60)
            print("EXTRACTION COMPLETE")
            print("=" * 60)
            print(f"Files processed: {summary['total_files_processed']}")
            print(f"Frames processed: {summary['total_frames_processed']}")
            print(f"Errors: {summary['total_errors']}")
            print(f"Duration: {summary['duration_sec']:.1f}s")
            print(f"Log file: {summary['log_file']}")
            print("Output files:")
            for output_file in summary["output_files"]:
                print(f"  - {output_file}")
            print("=" * 60)

        except Exception as e:
            print(f"ERROR: {str(e)}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "validate":
        try:
            from scripts.config.dataset_config import DatasetConfig
            from scripts.config.extraction_config import ExtractionConfig
            from scripts.run_extraction import (
                load_dataset_config,
                load_extraction_config,
            )

            print("Validating configs...")
            dataset_config = load_dataset_config(args.dataset_config)
            extraction_config = load_extraction_config(args.extraction_config)

            # Validate dataset
            print(f"✓ Dataset config loaded: {dataset_config.name}")
            print(f"  - Root dir: {dataset_config.root_dir}")
            print(f"  - Sample rate: {dataset_config.sample_rate} Hz")
            print(f"  - Frame duration: {dataset_config.frame_duration}s")
            print(f"  - Silence trimming: {dataset_config.enable_silence_trimming}")
            print(f"  - Normalization: {dataset_config.normalization_mode}")

            # Validate extraction
            print(f"✓ Extraction config loaded")
            print(f"  - Extractors: {', '.join(extraction_config.extractors)}")
            print(f"  - Window lengths: {extraction_config.window_lengths}s")
            print(f"  - Output dir: {extraction_config.output_dir}")
            print(f"  - Batch size: {extraction_config.batch_size}")

            # Check dataset folder exists
            dataset_root = Path(dataset_config.root_dir)
            if not dataset_root.exists():
                print(f"✗ Dataset root does not exist: {dataset_root}")
                sys.exit(1)

            print("✓ All validations passed!")

        except Exception as e:
            print(f"✗ Validation failed: {str(e)}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "list-extractors":
        extractors = list_extractors()
        print("Available extractors:")
        for name, description in sorted(extractors.items()):
            print(f"  - {name}: {description}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
