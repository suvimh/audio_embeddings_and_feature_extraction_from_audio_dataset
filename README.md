# Audio Embeddings and Feature Extraction Pipeline

A modular, production-ready Python framework for extracting deep learning embeddings (VGGish, CLAP, Whisper) and traditional acoustic features (OpenSMILE) from audio datasets. Designed for research workflows with flexible dataset configurations, batch processing, resource monitoring, and comprehensive logging.

**Features:**

- 🎯 **Multiple Extractors**: VGGish (128-dim), CLAP (512/1024-dim), Whisper (384-1280-dim), OpenSMILE (62-6373-dim)
- 📊 **Multi-Window Extraction**: Process multiple frame durations in a single run
- 🎛️ **Flexible Audio Processing**: Configurable silence trimming, normalization (peak/RMS/LUFS)
- 💾 **Scalable I/O**: HDF5 accumulation → Parquet output with crash recovery
- 📝 **Structured Logging**: JSON-lines format with resource monitoring
- ⚡ **Resource-Aware**: Auto-calculate batch sizes, GPU memory management, thread-pool I/O parallelism
- 🔧 **User-Friendly CLI**: Python or YAML config files with validation utilities
- 📈 **Production-Ready**: Comprehensive error handling, type hints, extensive documentation

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/suvimh/audio_embeddings_and_feature_extraction_from_audio_dataset.git
cd audio_embeddings_and_feature_extraction_from_audio_dataset

# Create conda environment
conda create -n audio_extraction python=3.12
conda activate audio_extraction

# Install dependencies
pip install -r requirements.txt

# Install extractor models (VGGish, CLAP, Whisper)
pip install -r requirements-extractors.txt
```

**Dataset-specific guides:** [TUNI Emotion Dataset](TUNI_emotion_data_extraction/README.md)

### 2. Configure Your Dataset

Create a Python file `configs/my_dataset.py`:

```python
from scripts.config.dataset_config import DatasetConfig

dataset = DatasetConfig(
    name="my_audio_dataset",
    root_dir="/path/to/audio/data",
    level_names=["speaker", "condition"], # these need to be in the correct level order
    participant_level=0,
    sample_rate=16000,
    frame_duration=3.0,
    overlap_percentage=0.25,
    enable_silence_trimming=True,
    silence_threshold_db=-40.0,
    normalization_mode="peak",
)
```

**Dataset Folder Structure Example:**

```
/path/to/audio/data/
  singer_001/
    condition_A/
      audio_001.wav
      audio_002.wav
    condition_B/
      audio_003.wav
  singer_002/
    condition_A/
      audio_004.wav
```

### 3. Configure Extraction

Create a Python file `configs/my_extraction.py`:

```python
from scripts.config.extraction_config import ExtractionConfig

extraction = ExtractionConfig(
    extractors=["vggish", "clap", "whisper"],
    output_dir="/path/to/output",
    window_lengths=[3.0, 5.0],  # Extract for both 3s and 5s frames
    extractor_params={
        "clap": {"version": "2023"},
        "whisper": {"model_name": "openai/whisper-base"},
    },
    batch_size="auto",
    num_workers=4,
)
```

### 4. Run Extraction

```bash
# Validate configs first
python -m scripts validate \
  --dataset-config configs/my_dataset.py \
  --extraction-config configs/my_extraction.py

# Run extraction
python -m scripts extract \
  --dataset-config configs/my_dataset.py \
  --extraction-config configs/my_extraction.py

# Monitor progress in JSON-lines logs
tail -f extraction_TIMESTAMP.jsonl
```

**Output Files:**

Naming pattern: `{dataset_name}_{extractor_name}_{duration}s.parquet`

```
/path/to/output/
  my_audio_dataset_vggish_3.0s.parquet
  my_audio_dataset_clap-2023_3.0s.parquet
  my_audio_dataset_whisper_whisper-base_3.0s.parquet
  my_audio_dataset_opensmile-compare-2016_3.0s.parquet
  extraction_TIMESTAMP.jsonl
```

---

## Complete Documentation

### 1. Dataset Configuration

Define your dataset structure and audio preprocessing parameters in `scripts/config/dataset_config.py`.

**Key Parameters:**

| Parameter                 | Type      | Default  | Description                                                  |
| ------------------------- | --------- | -------- | ------------------------------------------------------------ |
| `name`                    | str       | —        | Human-readable dataset identifier                            |
| `root_dir`                | str/Path  | —        | Root directory containing audio files                        |
| `level_names`             | list[str] | —        | Folder hierarchy labels (e.g., `["speaker", "condition"]`)   |
| `participant_level`       | int       | 1        | Which folder depth contains the speaker/subject ID           |
| `sample_rate`             | int       | 16000    | Target sample rate (Hz)                                      |
| `frame_duration`          | float     | 3.0      | Frame duration (seconds)                                     |
| `overlap_percentage`      | float     | 0.25     | Overlap between frames (0.0–1.0)                             |
| `enable_silence_trimming` | bool      | True     | Trim silence from audio                                      |
| `silence_threshold_db`    | float     | -40.0    | Silence threshold (dB, relative to RMS)                      |
| `silence_min_duration_ms` | float     | 500.0    | Minimum silence gap to trim (ms)                             |
| `normalization_mode`      | str       | "peak"   | Audio normalization: "peak", "rms", "lufs", or "none"        |
| `file_suffix_filter`      | str/None  | None     | Only process files with specific suffix (e.g., "-mic-audio") |
| `audio_extensions`        | list[str] | [".wav"] | Audio file extensions to process                             |

**Example: Hierarchical Dataset with Subgroups**

```python
from scripts.config.dataset_config import DatasetConfig

dataset = DatasetConfig(
    name="voice_quality_vte",
    root_dir="/data/VTE",
    level_names=["experience", "speaker", "phonation", "condition", "scale", "take"], # these need to be in the correct level order
    participant_level=1,
    subgroup_configs={
        "professional": ["experience", "speaker", "phonation", "condition", "scale", "take"],
        "inexperienced": ["experience", "speaker", "condition", "scale", "take"],  # No phonation level
    },
    subgroup_split_level=0,
    sample_rate=16000,
    frame_duration=3.0,
    overlap_percentage=0.25,
    enable_silence_trimming=True,
    silence_threshold_db=-35.0,
    silence_min_duration_ms=300.0,
    normalization_mode="rms",
    file_suffix_filter="-mic-audio",
)
```

### 2. Extraction Configuration

Define which embeddings/features to extract and output settings in `scripts/config/extraction_config.py`.

**Key Parameters:**

| Parameter           | Type        | Default | Description                                                     |
| ------------------- | ----------- | ------- | --------------------------------------------------------------- |
| `extractors`        | list[str]   | —       | Extractors to run: `["vggish", "clap", "whisper", "opensmile"]` |
| `output_dir`        | str/Path    | —       | Output directory for `.parquet` files                           |
| `window_lengths`    | list[float] | `[3.0]` | Frame durations to extract (seconds)                            |
| `extractor_params`  | dict        | `{}`    | Per-extractor parameters (see below)                            |
| `batch_size`        | int/"auto"  | "auto"  | Files per batch before HDF5 flush; "auto" = calculate from RAM  |
| `gpu_cache_cleanup` | bool        | False   | Empty GPU cache between batches (slower but safer)              |
| `num_workers`       | int         | 4       | Thread pool workers for file I/O (0 = sequential)               |

**Extractor Parameters:**

```python
extractor_params = {
    "clap": {
        "version": "2023",  # or "2024" (1024-dim)
    },
    "whisper": {
        "model_name": "openai/whisper-base",  # Options: tiny, base, small, medium, large
        "device": None,  # or "cuda", "mps", "cpu" (None = auto-detect)
    },
    "opensmile": {
        "feature_set": "ComParE_2016",  # Options: ComParE_2016 (6373-dim), eGeMAPSv02 (88-dim), GeMAPSv01b (62-dim)
        "feature_level": "Functionals",  # Statistics over time (current only option)
    },
}
```

### 3. Available Extractors

```bash
python -m scripts list-extractors
```

| Extractor     | Type              | Dim      | Device  | Description                                     |
| ------------- | ----------------- | -------- | ------- | ----------------------------------------------- |
| **VGGish**    | DL Embedding      | 128      | GPU/CPU | Audio tagging from Google (TensorFlow)          |
| **CLAP**      | DL Embedding      | 512/1024 | GPU     | Audio-text embeddings (MS Research)             |
| **Whisper**   | DL Embedding      | 384–1280 | GPU     | Speech encoder (OpenAI)                         |
| **OpenSMILE** | Acoustic Features | 62–6373  | CPU     | Traditional acoustic features (paralinguistics) |

### 4a. Command-Line Interface

```bash
# Show help
python -m scripts --help

# Validate configs
python -m scripts validate \
  --dataset-config configs/dataset.py \
  --extraction-config configs/extraction.py

# Run extraction
python -m scripts extract \
  --dataset-config configs/dataset.py \
  --extraction-config configs/extraction.py \
  --log-file /path/to/logs/run.jsonl \
  --no-console  # Optional: suppress console output

# List available extractors
python -m scripts list-extractors
```

### 4b. Using from Jupyter Notebooks

If you prefer working in a Jupyter notebook instead of the command line, you can use the pipeline directly in Python:

#### Setup

```python
# Install in your notebook environment (run once)
# !pip install -r requirements.txt

from scripts.config.dataset_config import DatasetConfig
from scripts.config.extraction_config import ExtractionConfig
from scripts.run_extraction import run_extraction
import pandas as pd
import json
from pathlib import Path
```

#### Define Your Dataset Configuration

```python
# Define dataset configuration directly in notebook
dataset = DatasetConfig(
    name="my_audio_dataset",
    root_dir="/path/to/audio/data",
    level_names=["speaker", "condition"], # these need to be in the correct level order
    participant_level=0,
    sample_rate=16000,
    frame_duration=3.0,
    overlap_percentage=0.25,
    enable_silence_trimming=True,
    silence_threshold_db=-40.0,
    normalization_mode="peak",
)

# Validate dataset config
print(f"Dataset: {dataset.name}")
print(f"Root directory: {dataset.root_dir}")
print(f"Levels: {dataset.level_names}")
```

#### Define Your Extraction Configuration

```python
# Define extraction configuration
extraction = ExtractionConfig(
    extractors=["vggish", "clap", "whisper"],
    output_dir="/path/to/output",
    window_lengths=[3.0, 5.0],
    extractor_params={
        "clap": {"version": "2023"},
        "whisper": {"model_name": "openai/whisper-base"},
    },
    batch_size="auto",
    num_workers=4,
)

# View configuration
print(f"Extractors: {extraction.extractors}")
print(f"Output directory: {extraction.output_dir}")
print(f"Window lengths: {extraction.window_lengths}")
```

#### Validate Configurations

```python
# Validate both configurations before running
from scripts.validate_config import validate_config

try:
    validate_config(dataset, extraction)
    print("✓ Configurations are valid!")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
```

#### Run Extraction

**Option 1: Pass file paths (if you have config files)**

```python
from scripts.run_extraction import run_extraction

# Run extraction using config file paths
run_extraction(
    dataset_config_path="configs/my_dataset.py",
    extraction_config_path="configs/my_extraction.py",
    log_file="extraction_run.jsonl",
)

print(f"✓ Extraction complete!")
```

**Option 2: Save configs and then extract (if you defined them in notebook)**

```python
from scripts.run_extraction import run_extraction

# Save your programmatically-created configs to files
dataset.save_yaml("my_dataset_config.yml")
extraction.save_yaml("my_extraction_config.yml")

# Run extraction using the saved files
run_extraction(
    dataset_config_path="my_dataset_config.yml",
    extraction_config_path="my_extraction_config.yml",
    log_file="extraction_run.jsonl",
)

print(f"✓ Extraction complete!")
```

#### Load and Explore Results

```python
# Load extraction results
output_dir = Path(extraction.output_dir)

# List all output files
output_files = list(output_dir.glob("*.parquet"))
print("Generated output files:")
for f in output_files:
    print(f"  - {f.name}")

# Load a specific embedding (adjust dataset name, extractor, and duration)
df_vggish = pd.read_parquet(output_dir / "my_audio_dataset_vggish_3.0s.parquet")

print(f"\nVGGish embeddings shape: {df_vggish.shape}")
print(f"Columns: {list(df_vggish.columns)}")
print(f"\nFirst few rows:")
df_vggish.head()
```

#### Access Embeddings

```python
# Get numpy arrays of embeddings
embeddings = df_vggish["embedding"].values
print(f"Number of frames: {len(embeddings)}")
print(f"Embedding dimension: {embeddings[0].shape}")

# Example: Compute embedding statistics
import numpy as np

embedding_matrix = np.array([e for e in embeddings])
print(f"\nEmbedding statistics:")
print(f"Mean: {embedding_matrix.mean(axis=0)[:5]}...")  # First 5 dims
print(f"Std: {embedding_matrix.std(axis=0)[:5]}...")
```

#### Parse Logs and Monitor Progress

```python
# Read JSON-lines log file
logs = []
with open(log_file, 'r') as f:
    for line in f:
        logs.append(json.loads(line))

# Convert to dataframe for easier analysis
logs_df = pd.DataFrame(logs)

# View all INFO level messages
info_logs = logs_df[logs_df['level'] == 'INFO']
print(info_logs[['timestamp', 'stage', 'message']])

# Extract completion statistics
completion_logs = logs_df[logs_df['stage'] == 'extraction_complete']
print("\nExtraction Summary:")
for _, log in completion_logs.iterrows():
    print(f"{log['extractor']} @ {log['window_length']}s:")
    print(f"  Files: {log['files_processed']}")
    print(f"  Frames: {log['frames_processed']}")
    print(f"  Errors: {log['errors']}")
    print(f"  Warnings: {log['warnings']}")
```

#### Check Resource Usage

```python
# View resource checkpoints
resource_logs = logs_df[logs_df['stage'] == 'resource_checkpoint']
print("Resource Usage Over Time:")
print(resource_logs[['timestamp', 'ram_used_percent', 'gpu_memory_used_mb']])
```

#### Save Configuration as YAML (for CLI use later)

```python
# Convert configs to YAML for reproducibility
dataset.save_yaml("dataset_config_generated.yml")
extraction.save_yaml("extraction_config_generated.yml")

print("Saved YAML configs:")
print("  - dataset_config_generated.yml")
print("  - extraction_config_generated.yml")

# Later, you can load these from CLI:
# python -m scripts extract \
#   --dataset-config dataset_config_generated.yml \
#   --extraction-config extraction_config_generated.yml
```

#### Complete Notebook Example

```python
# Complete workflow in one cell
from pathlib import Path
from scripts.config.dataset_config import DatasetConfig
from scripts.config.extraction_config import ExtractionConfig
from scripts.run_extraction import run_extraction
from scripts.validate_config import validate_config
import pandas as pd
import json

# Configure
dataset = DatasetConfig(
    name="my_data",
    root_dir="/tmp/audio",
    level_names=["speaker"], # these need to be in the correct level order
    participant_level=0,
)

extraction = ExtractionConfig(
    extractors=["vggish"],
    output_dir="/tmp/output",
    window_lengths=[3.0],
)

# Validate
validate_config(dataset, extraction)

# Save configs to files (required by run_extraction)
dataset.save_yaml("dataset_config.yml")
extraction.save_yaml("extraction_config.yml")

# Extract using file paths
run_extraction(
    dataset_config_path="dataset_config.yml",
    extraction_config_path="extraction_config.yml",
    log_file="run.jsonl"
)

# Load results
df = pd.read_parquet(Path(extraction.output_dir) / "my_audio_dataset_vggish_3.0s.parquet")
print(f"Processed {len(df)} frames")
print(f"Columns: {list(df.columns)}")
```

### 5. Output Format

All embeddings and features are saved as **Apache Parquet** with metadata:

```python
# Load outputs
import pandas as pd
from scripts.utils import load_parquet

df = load_parquet("output/my_audio_dataset_vggish_3.0s.parquet")

print(df.columns)
# Index(['speaker', 'condition', 'filename', 'filepath', 'frame_index',
#        'embedding', 'extractor', ...]

print(df.shape)
# (N_frames, n_columns)

# Access embeddings
embeddings = df["embedding"].values  # List of numpy arrays
print(embeddings[0].shape)  # (128,) for VGGish

# Metadata JSON sidecar
import json
with open("output/my_audio_dataset_vggish_3.0s.json") as f:
    metadata = json.load(f)
```

### 6. Logging & Monitoring

Logs are written as JSON-lines (one JSON object per line) for easy parsing:

```bash
# Monitor live logs
tail -f extraction_TIMESTAMP.jsonl | jq .

# Parse all errors
cat extraction_TIMESTAMP.jsonl | jq 'select(.level=="ERROR")'

# Extract resource usage
cat extraction_TIMESTAMP.jsonl | jq 'select(.stage=="resource_checkpoint")'
```

**Example Log Entry:**

```json
{
  "timestamp": "2026-05-07T10:30:00.123456",
  "level": "INFO",
  "stage": "extraction_complete",
  "extractor": "vggish",
  "window_length": 3.0,
  "files_processed": 1250,
  "frames_processed": 12500,
  "errors": 0,
  "warnings": 2,
  "message": "Extraction complete: vggish @ 3.0s | 1250 files, 12500 frames | 0 errors, 2 warnings | 2345.6s"
}
```

### 7. Resource Management

The pipeline automatically:

- **Calculates batch sizes** from available RAM if `batch_size="auto"`
- **Monitors GPU memory** and logs peak usage
- **Parallelizes file I/O** with thread pools (configurable via `num_workers`)
- **Checks memory pressure** and warns if >85% utilized
- **Logs resource checkpoints** every N files for debugging

**Memory Tuning:**

```python
# For very large datasets (>100GB)
ExtractionConfig(
    batch_size=5,  # Explicit small batch size
    gpu_cache_cleanup=True,  # Clear GPU cache between batches
    num_workers=2,  # Reduce I/O parallelism
)

# For small datasets (<1GB) on powerful machines
ExtractionConfig(
    batch_size="auto",  # Let system calculate
    gpu_cache_cleanup=False,  # Keep models loaded
    num_workers=8,  # Max I/O parallelism
)
```

### 8. Error Handling & Recovery

- **Non-fatal file errors** (corrupt files, missing metadata) are logged as WARNING; pipeline continues
- **Fatal errors** (invalid config, model loading failure) stop the pipeline with clear error messages
- **Crash recovery**: Completed files are checkpointed; re-running resumes from the last checkpoint
- **All errors logged** to JSON-lines file for post-mortem analysis

---

## Examples

See the `examples/` directory for complete config templates:

- [examples/dataset_config_example.py](examples/dataset_config_example.py) — Dataset config examples
- [examples/extraction_config_example.py](examples/extraction_config_example.py) — Extraction config examples
- [examples/dataset_config_example.yml](examples/dataset_config_example.yml) — YAML format (auto-generated)
- [examples/extraction_config_example.yml](examples/extraction_config_example.yml) — YAML format (auto-generated)

### Example: Simple 5-Minute Extraction

```bash
# 1. Create minimal configs
cat > dataset.py << 'EOF'
from scripts.config.dataset_config import DatasetConfig
config = DatasetConfig(
    name="my_data",
    root_dir="/tmp/audio",
    level_names=["speaker"], # these need to be in the correct level order
    participant_level=0,
)
EOF

cat > extraction.py << 'EOF'
from scripts.config.extraction_config import ExtractionConfig
config = ExtractionConfig(
    extractors=["vggish"],
    output_dir="/tmp/output",
    window_lengths=[3.0],
)
EOF

# 2. Validate
python -m scripts validate --dataset-config dataset.py --extraction-config extraction.py

# 3. Run
python -m scripts extract --dataset-config dataset.py --extraction-config extraction.py
```

---

## Architecture

```
scripts/
├── config/
│   ├── dataset_config.py          # DatasetConfig dataclass + validation
│   └── extraction_config.py        # ExtractionConfig + YAML serialization
├── dl_embeddings/
│   ├── extract_vggish_embeddings.py
│   ├── extract_clap_embeddings.py
│   ├── extract_whisper_embeddings.py
│   └── ...
├── feature_sets/
│   ├── extract_opensmile.py
│   └── ...
├── utils.py                        # Core extraction pipeline
├── extractors_registry.py          # Extractor factory & metadata
├── resource_monitor.py             # RAM/GPU monitoring
├── logging_config.py               # JSON-lines logger
├── run_extraction.py               # Main orchestrator
├── validate_config.py              # Config validation
├── cli.py                          # CLI interface
└── __main__.py                     # python -m scripts entry point
```

**Data Flow:**

```
dataset_config.py + extraction_config.py
           ↓
    run_extraction()
           ↓
  For each (extractor, window_length):
    - Instantiate extractor via registry
    - Load audio files (with thread-pool I/O)
    - Frame + preprocess (silence trimming, normalization)
    - Extract embeddings
    - Accumulate to HDF5 (batch flushing)
    - Log progress (JSON-lines)
           ↓
    HDF5 → Parquet conversion
           ↓
    output/{dataset_name}_{extractor_name}_{duration}s.parquet
```

---

## Contributing

To add a new extractor:

1. Create extraction function in `scripts/<category>/extract_<name>.py`
2. Implement `make_<name>_extractor()` factory returning `FeatureExtractorConfig`
3. Register in `scripts/extractors_registry.py` inside `make_extractor()` and `EXTRACTOR_METADATA`
4. Add example in `examples/extraction_config_example.py`
5. Update this README

See [DEVELOPMENT.md](DEVELOPMENT.md) for full details.

---

## Citation

If you use this pipeline in research, please cite: TBD
