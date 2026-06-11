from __future__ import annotations

import json
import numpy as np
import pandas as pd
import librosa
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from tqdm import tqdm
from scripts.config.dataset_config import DatasetConfig

# ---------------------------------------------------------------------------
# Audio processing utilities
# ---------------------------------------------------------------------------


def normalize_audio(
    audio: np.ndarray,
    mode: str = "none",
) -> np.ndarray:
    """
    Normalize audio to a reference level.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal (mono or multi-channel).
    mode : str
        Normalization mode: 'peak' (0dB), 'rms', 'lufs', or 'none'.

    Returns
    -------
    np.ndarray
        Normalized audio.
    """

    if mode == "none":
        return audio

    if mode == "peak":
        return librosa.util.normalize(audio, axis=None)

    elif mode == "rms":
        rms = np.sqrt(np.mean(audio**2))
        return audio / rms if rms > 1e-6 else audio

    else:
        return audio


def trim_silence_smart(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40,
    min_duration_ms: float = 500,
) -> np.ndarray:
    """
    Trim silence from audio using a configurable threshold and minimum duration.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    threshold_db : float
        Silence threshold in dB (relative to RMS).
    min_duration_ms : float
        Minimum duration of silence (ms) to be considered for trimming.

    Returns
    -------
    np.ndarray
        Audio with silence trimmed from edges and long interior gaps.
    """
    # Use librosa's trim with a dynamic threshold based on RMS
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-6:
        return audio

    # Convert dB to linear scale
    threshold_linear = rms * (10 ** (threshold_db / 20))

    # Trim silence from edges
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=abs(threshold_db),
        ref=np.max,
    )

    return trimmed


# ---------------------------------------------------------------------------
# Feature extractor abstraction
# ---------------------------------------------------------------------------


@dataclass
class FeatureExtractorConfig:
    """Wraps a callable embedding model with metadata."""

    name: str
    extract_fn: Callable
    embedding_dim: int
    min_input_samples: Optional[int] = None


# ---------------------------------------------------------------------------
# Audio loading + framing
# ---------------------------------------------------------------------------


def load_and_frame_audio(
    audio_file: str | Path,
    frame_duration: float,
    overlap_percentage: float,
    target_sr: int = 16000,
    min_frame_samples: Optional[int] = None,
    enable_silence_trimming: bool = True,
    silence_threshold_db: float = -40.0,
    silence_min_duration_ms: float = 500.0,
    normalization_mode: str = "none",
) -> Optional[list[np.ndarray]]:
    """
    Load an audio file, resample, trim silence, normalize, and split into overlapping frames.
    Frames shorter than min_frame_samples are tiled to meet the minimum length.
    Returns a list of float32 numpy arrays, one per frame. Returns None on error.

    Parameters
    ----------
    audio_file : str | Path
        Path to the audio file.
    frame_duration : float
        Duration of each frame in seconds.
    overlap_percentage : float
        Overlap between frames (0.0-1.0).
    target_sr : int
        Target sample rate.
    min_frame_samples : int | None
        Minimum samples required per frame (pads with tiling if smaller).
    enable_silence_trimming : bool
        If True, trim silence from audio.
    silence_threshold_db : float
        Silence threshold in dB.
    silence_min_duration_ms : float
        Minimum silence duration in ms.
    normalization_mode : str
        Normalization mode: 'peak', 'rms', 'lufs', or 'none'.
    """
    try:
        audio, sr = librosa.load(str(audio_file), sr=None)

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        if enable_silence_trimming:
            audio = trim_silence_smart(
                audio,
                sr=sr,
                threshold_db=silence_threshold_db,
                min_duration_ms=silence_min_duration_ms,
            )

        audio = normalize_audio(audio, mode=normalization_mode)

        frame_len = int(frame_duration * sr)
        hop_length = int(frame_len * (1 - overlap_percentage))

        if len(audio) < frame_len:
            repeat_factor = int(np.ceil(frame_len / len(audio)))
            audio = np.tile(audio, repeat_factor)[:frame_len]

        frames_matrix = librosa.util.frame(
            audio, frame_length=frame_len, hop_length=hop_length
        )

        result = []
        for frame in frames_matrix.T:
            frame = np.ascontiguousarray(frame, dtype=np.float32)
            if min_frame_samples and len(frame) < min_frame_samples:
                repeat_factor = int(np.ceil(min_frame_samples / len(frame)))
                frame = np.tile(frame, repeat_factor)[:min_frame_samples]
            result.append(frame)

        return result if result else None

    except Exception as e:
        print(f"[WARN] Error loading {audio_file}: {e}")
        return None


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_audio_files(
    root_dir: str | Path,
    config: DatasetConfig,
) -> list[Path]:
    """
    Walk root_dir and return all audio file paths that pass config filters:
      - extension in config.audio_extensions
      - name does not start with any prefix in config.skip_prefixes
      - stem ends with config.file_suffix_filter (if set)
    """
    root_dir = Path(root_dir)
    files = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(path.name.startswith(p) for p in config.skip_prefixes):
            continue
        if config.matches_file_filter(path):
            files.append(path)
    return sorted(files)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def extract_file_metadata(
    file_path: Path,
    root_dir: Path,
    config: DatasetConfig,
) -> dict:
    """
    Extract all available metadata from a file's path relative to root_dir.
    Prints warnings for any unexpected folder values.
    """
    relative = file_path.relative_to(root_dir)
    parts = list(relative.parts[:-1])
    level_labels = config.level_names_for_path(parts)

    for w in config.validate_metadata(level_labels):
        print(f"[WARN] {file_path.name}: {w}")

    return {
        **level_labels,
        "filename": file_path.stem,
        "filepath": str(file_path),
    }


# ---------------------------------------------------------------------------
# HDF5 helpers  (used during extraction as accumulation store)
# ---------------------------------------------------------------------------


def _hdf5_path(checkpoint_dir: Path, out_stem: str) -> Path:
    return checkpoint_dir / f"{out_stem}.h5"


def _checkpoint_path(checkpoint_dir: Path, out_stem: str) -> Path:
    return checkpoint_dir / f"{out_stem}.checkpoint.jsonl"


def _load_checkpoint(checkpoint_dir: Path, out_stem: str, extractor_name: str = None) -> set[str]:
    """
    Return set of already-processed file paths from checkpoint file.
    
    If extractor_name is provided, only returns files processed by that specific extractor.
    This allows running different extractors on the same files without conflicts.
    """
    cp = _checkpoint_path(checkpoint_dir, out_stem)
    if not cp.exists():
        return set()
    completed = set()
    with open(cp, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                filepath = entry.get("filepath", "")
                entry_extractor = entry.get("extractor", None)
                
                # If extractor_name is specified, only include this extractor's files
                # If not specified (backwards compatibility), include all
                if extractor_name is None or entry_extractor == extractor_name:
                    completed.add(filepath)
    
    print(f"[INFO] Checkpoint found: {len(completed)} files already processed for {extractor_name or 'any extractor'}.")
    return completed


def _append_checkpoint(checkpoint_dir: Path, out_stem: str, filepath: str, extractor_name: str = None) -> None:
    """
    Record a completed filepath in the checkpoint file.
    
    If extractor_name is provided, records which extractor processed this file.
    This allows resuming different extractors independently.
    """
    with open(_checkpoint_path(checkpoint_dir, out_stem), "a") as f:
        entry = {"filepath": filepath}
        if extractor_name:
            entry["extractor"] = extractor_name
        f.write(json.dumps(entry) + "\n")


def _append_rows_to_hdf5(
    rows: list[dict], hdf5_file: Path, label_columns: list[str]
) -> None:
    """
    Append a batch of rows to the HDF5 accumulation file.

    HDF5 layout:
      /embeddings          : float32 array, shape (N, embedding_dim), resizable
      /metadata/<col>      : variable-length string dataset for each label column
      /metadata/filename   : variable-length string
      /metadata/filepath   : variable-length string
      /metadata/frame_index: int32 array
      /metadata/extractor  : variable-length string
    """
    if not rows:
        return

    embeddings = np.stack([r["embedding"] for r in rows]).astype(np.float32)
    n_new = len(rows)
    str_dt = h5py.special_dtype(vlen=str)

    with h5py.File(hdf5_file, "a") as f:
        # --- embeddings ---
        if "embeddings" not in f:
            emb_dim = embeddings.shape[1]
            f.create_dataset(
                "embeddings",
                data=embeddings,
                maxshape=(None, emb_dim),
                chunks=(64, emb_dim),
            )
        else:
            ds = f["embeddings"]
            old_len = ds.shape[0]
            ds.resize(old_len + n_new, axis=0)
            ds[old_len:] = embeddings

        # --- metadata ---
        if "metadata" not in f:
            f.create_group("metadata")

        meta_cols = label_columns + ["filename", "filepath", "extractor"]
        for col in meta_cols:
            values = [str(r.get(col) or "") for r in rows]
            if col not in f["metadata"]:
                f["metadata"].create_dataset(
                    col,
                    data=np.array(values, dtype=object),
                    maxshape=(None,),
                    dtype=str_dt,
                    chunks=(256,),
                )
            else:
                ds = f["metadata"][col]
                old_len = ds.shape[0]
                ds.resize(old_len + n_new, axis=0)
                ds[old_len:] = values

        # frame_index (int)
        frame_indices = np.array([r["frame_index"] for r in rows], dtype=np.int32)
        if "frame_index" not in f["metadata"]:
            f["metadata"].create_dataset(
                "frame_index", data=frame_indices, maxshape=(None,), chunks=(256,)
            )
        else:
            ds = f["metadata"]["frame_index"]
            old_len = ds.shape[0]
            ds.resize(old_len + n_new, axis=0)
            ds[old_len:] = frame_indices


def _hdf5_to_parquet(hdf5_file: Path, parquet_file: Path) -> pd.DataFrame:
    """
    Read the completed HDF5 accumulation file and write final Parquet output.
    Returns the loaded DataFrame.
    """
    with h5py.File(hdf5_file, "r") as f:
        embeddings = f["embeddings"][:]  # (N, dim) float32
        meta = {}
        for col in f["metadata"]:
            raw = f["metadata"][col][:]
            if raw.dtype.kind in ("S", "O"):  # string datasets
                meta[col] = [
                    v.decode() if isinstance(v, bytes) else str(v) for v in raw
                ]
            else:
                meta[col] = raw.tolist()

    df = pd.DataFrame(meta)

    # Serialise embeddings to bytes for parquet storage
    df["embedding"] = list(embeddings)
    df["embedding_dtype"] = embeddings.dtype.str
    df["embedding_shape"] = [
        json.dumps(list(embeddings.shape[1:])) for _ in range(len(df))
    ]
    df["embedding"] = [row.tobytes() for row in embeddings]

    df.to_parquet(parquet_file, index=False)
    print(f"[INFO] Parquet saved to {parquet_file}")
    return load_parquet(parquet_file)


# ---------------------------------------------------------------------------
# Parquet I/O (final output format)
# ---------------------------------------------------------------------------


def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.frombuffer(x, dtype=np.float32)
    )
    return df


def save_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame with numpy array 'embedding' column to Parquet.
    Embeddings are serialised to bytes. Use load_parquet() to reconstruct.
    """
    df_save = df.copy()
    df_save["embedding_dtype"] = df_save["embedding"].iloc[0].dtype.str
    df_save["embedding_shape"] = df_save["embedding"].apply(
        lambda x: json.dumps(list(x.shape))
    )
    df_save["embedding"] = df_save["embedding"].apply(lambda x: x.tobytes())
    df_save.to_parquet(path, index=False)
    print(f"[INFO] Saved to {path}")


def save_metadata(
    df: pd.DataFrame,
    dataset_config: DatasetConfig,
    extractor_config: FeatureExtractorConfig,
    out_file: Path,
) -> None:
    """Save a JSON sidecar with extraction config for reproducibility."""
    label_cols = [
        c
        for c in df.columns
        if c not in ("embedding", "extractor", "frame_index", "filepath", "filename")
    ]
    meta = {
        "dataset": dataset_config.name,
        "extractor": extractor_config.name,
        "embedding_dim": extractor_config.embedding_dim,
        "frame_duration": dataset_config.frame_duration,
        "overlap_percentage": dataset_config.overlap_percentage,
        "sample_rate": dataset_config.sample_rate,
        "file_suffix_filter": dataset_config.file_suffix_filter,
        "total_frames": len(df),
        "total_files": df["filepath"].nunique(),
        "label_columns": label_cols,
        "unique_values": {
            col: df[col].dropna().unique().tolist() for col in label_cols
        },
    }
    meta_file = out_file.with_suffix(".json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata saved to {meta_file}")


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_dataset_features(
    dataset_config: DatasetConfig,
    extractor_config: FeatureExtractorConfig,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    overwrite: bool = False,
    batch_size=5,  # append to HDF5 every N files
) -> pd.DataFrame:
    """
    Extract embeddings for all audio files in the dataset.

    Crash-safe strategy:
      - Rows are appended to a single HDF5 file (on Drive) every file
      - A checkpoint .jsonl tracks which files are fully done PER EXTRACTOR
      - On completion, HDF5 is converted to final Parquet and HDF5 + checkpoint are cleaned up
      - If interrupted, re-running resumes automatically from the checkpoint
      - Different extractors can be run independently on the same dataset

    Parameters
    ----------
    output_dir      : where the final .parquet and .json sidecar are written
    checkpoint_dir  : where the .h5 and .checkpoint.jsonl live during extraction
                      (defaults to output_dir — set this to a Drive path on Colab
                      so progress survives session crashes)
    overwrite       : if True, ignore existing output and re-extract from scratch
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    out_stem = f"{dataset_config.name}_{extractor_config.name}_{dataset_config.frame_duration}s"
    out_file = output_dir / f"{out_stem}.parquet"
    hdf5_file = _hdf5_path(checkpoint_dir, out_stem)

    # Already fully complete
    if out_file.exists() and not overwrite:
        print(f"[INFO] Output already exists: {out_file}. Loading from cache.")
        return load_parquet(out_file)

    # Resume from checkpoint if available
    completed_paths = _load_checkpoint(checkpoint_dir, out_stem, extractor_name=extractor_config.name)

    root_dir = Path(dataset_config.root_dir)
    all_files = collect_audio_files(root_dir, dataset_config)
    remaining = [f for f in all_files if str(f) not in completed_paths]
    already_done = len(all_files) - len(remaining)

    # Infer label columns from dataset config (union of all subgroup level names)
    all_level_names = (
        list(
            dict.fromkeys(
                name
                for names in (dataset_config.subgroup_configs or {}).values()
                for name in names
            )
        )
        or dataset_config.level_names
    )

    print(
        f"[INFO] {len(all_files)} total files | "
        f"{already_done} already processed | "
        f"{len(remaining)} remaining"
    )
    print(f"[INFO] HDF5 accumulation file: {hdf5_file}")
    print(f"[INFO] Final output: {out_file}")

    batch_rows = []
    extraction_errors = []
    extraction_warnings = []

    with tqdm(
        total=len(all_files),
        initial=already_done,
        desc=f"[{extractor_config.name}] {dataset_config.name}",
        unit="file",
    ) as pbar:

        for i, file_path in enumerate(remaining):
            try:
                metadata = extract_file_metadata(file_path, root_dir, dataset_config)

                frames = load_and_frame_audio(
                    file_path,
                    frame_duration=dataset_config.frame_duration,
                    overlap_percentage=dataset_config.overlap_percentage,
                    target_sr=dataset_config.sample_rate,
                    min_frame_samples=extractor_config.min_input_samples,
                    enable_silence_trimming=dataset_config.enable_silence_trimming,
                    silence_threshold_db=dataset_config.silence_threshold_db,
                    silence_min_duration_ms=dataset_config.silence_min_duration_ms,
                    normalization_mode=dataset_config.normalization_mode,
                )

                if frames is None:
                    extraction_warnings.append(
                        {
                            "file": str(file_path),
                            "reason": "Failed to load and frame audio",
                        }
                    )
                    _append_checkpoint(checkpoint_dir, out_stem, str(file_path), extractor_name=extractor_config.name)
                    pbar.update(1)
                    pbar.set_postfix(
                        {"frames": len(batch_rows), "file": file_path.name[:35]}
                    )
                    continue

                file_frames_extracted = 0
                for frame_idx, frame in enumerate(frames):
                    try:
                        embedding = extractor_config.extract_fn(frame)
                        if embedding is None:
                            extraction_warnings.append(
                                {
                                    "file": str(file_path),
                                    "frame": frame_idx,
                                    "reason": "Extractor returned None",
                                }
                            )
                            continue

                        if embedding.ndim > 1:
                            embedding = embedding.mean(axis=0)

                        batch_rows.append(
                            {
                                **metadata,
                                "frame_index": frame_idx,
                                "embedding": embedding,
                                "extractor": extractor_config.name,
                            }
                        )
                        file_frames_extracted += 1

                    except Exception as e:
                        extraction_errors.append(
                            {
                                "file": str(file_path),
                                "frame": frame_idx,
                                "error": str(e),
                            }
                        )
                        print(
                            f"[WARN] Frame extraction error for {file_path.name} frame {frame_idx}: {e}"
                        )

                # Mark file as done in checkpoint
                _append_checkpoint(checkpoint_dir, out_stem, str(file_path), extractor_name=extractor_config.name)
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "frames": len(batch_rows),
                        "file": file_path.name[:35],
                        "extracted": file_frames_extracted,
                    }
                )

                # Flush batch to HDF5 every N files
                if (i + 1) % batch_size == 0 and batch_rows:
                    _append_rows_to_hdf5(batch_rows, hdf5_file, all_level_names)
                    batch_rows = []

            except Exception as e:
                extraction_errors.append(
                    {
                        "file": str(file_path),
                        "stage": "file_processing",
                        "error": str(e),
                    }
                )
                print(f"[WARN] Error processing {file_path}: {e}")
                _append_checkpoint(checkpoint_dir, out_stem, str(file_path), extractor_name=extractor_config.name)
                pbar.update(1)

    # Flush remaining rows
    if batch_rows:
        _append_rows_to_hdf5(batch_rows, hdf5_file, all_level_names)

    # Convert HDF5 -> Parquet and clean up
    print("[INFO] Converting HDF5 to Parquet...")
    df = _hdf5_to_parquet(hdf5_file, out_file)
    print(
        f"[INFO] {len(df)} total frame embeddings from {df['filepath'].nunique()} files."
    )

    save_metadata(df, dataset_config, extractor_config, out_file)

    # Print error and warning summary
    if extraction_errors:
        print(f"\n[WARN] {len(extraction_errors)} extraction errors occurred:")
        for error in extraction_errors[:10]:  # Show first 10
            print(
                f"  - {error['file']}: {error.get('error', error.get('reason', 'Unknown error'))}"
            )
        if len(extraction_errors) > 10:
            print(f"  ... and {len(extraction_errors) - 10} more")

    if extraction_warnings:
        print(f"\n[WARN] {len(extraction_warnings)} extraction warnings:")
        for warning in extraction_warnings[:5]:  # Show first 5
            print(f"  - {warning['file']}: {warning.get('reason', 'Unknown warning')}")
        if len(extraction_warnings) > 5:
            print(f"  ... and {len(extraction_warnings) - 5} more")

    # Clean up HDF5 and checkpoint now that parquet is complete
    hdf5_file.unlink()
    _checkpoint_path(checkpoint_dir, out_stem).unlink(missing_ok=True)
    print("[INFO] Cleaned up HDF5 and checkpoint files.")

    return df
