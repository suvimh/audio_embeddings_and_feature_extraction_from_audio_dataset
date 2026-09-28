# Quick Start Guide

This 5-minute guide gets you from zero to your first extraction.

## Prerequisites

- Python 3.12+
- ~2GB RAM available
- Audio dataset in WAV format

## Installation

```bash
# Clone and enter directory
git clone <repo_url>
cd audio_embeddings_and_feature_extraction_from_audio_dataset

# Create conda environment
conda create -n audio_extraction python=3.12
conda activate audio_extraction

# Install dependencies
pip install -r requirements.txt

# (Optional) Install CLAP if using CLAP extractor
pip install git+https://github.com/microsoft/CLAP.git
```

## Step 1: Prepare Your Dataset

Organize your audio files in a folder structure. Example:

```
/data/my_audio/
  speaker_001/
    audio_001.wav
    audio_002.wav
  speaker_002/
    audio_003.wav
    audio_004.wav
```

**Note:** Folder names become metadata (e.g., "speaker_001" → speaker="speaker_001")

## Step 2: Create Dataset Config

Create `my_dataset_config.py`:

```python
from scripts.config.dataset_config import DatasetConfig

config = DatasetConfig(
    name="my_audio_data",
    root_dir="/data/my_audio",          # ← Change to your path
    level_names=["speaker"],             # ← Folder depths
    participant_level=0,                 # ← Where speaker ID is
    sample_rate=16000,
    frame_duration=3.0,
    overlap_percentage=0.25,
    enable_silence_trimming=True,
    silence_threshold_db=-40.0,
    normalization_mode="peak",
)
```

### Common Folder Structures

**Simple (1 level):**

```python
level_names=["speaker"]
participant_level=0
```

**Hierarchical (2+ levels):**

```python
level_names=["speaker", "condition"]
participant_level=0
```

**3 levels:**

```python
level_names=["speaker", "session", "recording"]
participant_level=0
```

## Step 3: Create Extraction Config

Create `my_extraction_config.py`:

```python
from scripts.config.extraction_config import ExtractionConfig

config = ExtractionConfig(
    extractors=["vggish"],              # ← Simplest: VGGish only
    output_dir="/data/output",          # ← Where to save results
    window_lengths=[3.0],               # ← 3-second frames
    batch_size="auto",                  # ← Auto-calculate from RAM
    num_workers=4,
)
```

### Other Quick Options

**Multiple embeddings:**

```python
ExtractionConfig(
    extractors=["vggish", "clap", "whisper"],
    output_dir="/data/output",
    window_lengths=[3.0],
    batch_size="auto",
    num_workers=4,
)
```

**Multiple window lengths:**

```python
ExtractionConfig(
    extractors=["vggish"],
    output_dir="/data/output",
    window_lengths=[3.0, 5.0, 7.0],  # ← Three different frame sizes
    batch_size="auto",
    num_workers=4,
)
```

**OpenSMILE features (no GPU needed):**

```python
ExtractionConfig(
    extractors=["opensmile"],
    output_dir="/data/output",
    window_lengths=[3.0],
    batch_size="auto",
    num_workers=8,
)
```

## Step 4: Validate Configs

```bash
python -m scripts validate \
  --dataset-config my_dataset_config.py \
  --extraction-config my_extraction_config.py
```

**Expected output:**

```
Validating configs...
✓ Dataset config loaded: my_audio_data
  - Root dir: /data/my_audio
  - Sample rate: 16000 Hz
  - Frame duration: 3.0s
  - Silence trimming: True
  - Normalization: peak
✓ Extraction config loaded
  - Extractors: vggish
  - Window lengths: [3.0]s
  - Output dir: /data/output
  - Batch size: auto
✓ All validations passed!
```

## Step 5: Run Extraction

```bash
python -m scripts extract \
  --dataset-config my_dataset_config.py \
  --extraction-config my_extraction_config.py
```

**Watch live logs:**

```bash
tail -f extraction_TIMESTAMP.jsonl | jq .
```

**What to expect:**

```
[INFO] Extraction pipeline starting
[INFO] Resolved batch_size: 8 files
[INFO] X total files | Y already processed | Z remaining
[INFO] File processed: audio_001.wav (N frames)
...
[INFO] Extraction complete: vggish @ 3.0s | 10 files, 100 frames | 0 errors | 12.3s
[INFO] Extraction pipeline complete | 10 files, 100 frames | 0 errors | 15.6s

EXTRACTION COMPLETE
============================================================
Files processed: 10
Frames processed: 100
Errors: 0
Duration: 15.6s
Log file: /data/output/extraction_20260507_103000.jsonl
Output files:
  - /data/output/vggish_window_3s.parquet
============================================================
```

## Step 6: Load Results

```python
from scripts.utils import load_parquet
import pandas as pd

df = load_parquet("/data/output/vggish_window_3s.parquet")

print(f"Shape: {df.shape}")  # (N_frames, n_columns)
print(df.columns)           # Show all columns
print(f"Speakers: {df['speaker'].unique()}")

# Get embeddings
embeddings = df["embedding"].values  # List of numpy arrays
print(f"Embedding shape: {embeddings[0].shape}")  # e.g., (128,) for VGGish

# Save custom features
df.to_csv("embeddings_with_metadata.csv", index=False)
```

---

## Common Issues & Solutions

### Q: "Dataset root directory does not exist"

**A:** Update `root_dir` in dataset config to correct path:

```python
config = DatasetConfig(
    root_dir="/absolute/path/to/audio",  # Use absolute path
    # ...
)
```

### Q: "No audio files found"

**A:** Check:

1. Files are `.wav` format (or update `audio_extensions`)
2. Folder structure matches `level_names`
3. File permissions allow reading

### Q: "Out of Memory"

**A:** Reduce batch size or enable GPU cache cleanup:

```python
config = ExtractionConfig(
    batch_size=2,             # Explicit small batch
    gpu_cache_cleanup=True,   # Clean GPU between batches
    # ...
)
```

### Q: "Extractor model failed to load"

**A:** Ensure dependencies are installed:

```bash
pip install transformers torch  # For Whisper
pip install git+https://github.com/microsoft/CLAP.git  # For CLAP
pip install opensmile  # For OpenSMILE
```

### Q: "ModuleNotFoundError: No module named '...'"

**A:** Reinstall requirements:

```bash
pip install -r requirements.txt --upgrade
```

---

## Next Steps

- **See more examples:** `examples/` folder
- **Understand architecture:** [DEVELOPMENT.md](DEVELOPMENT.md)
- **Full documentation:** [README.md](README.md)
- **Add new extractors:** See DEVELOPMENT.md → "Adding a New Extractor"
- **Optimize for large datasets:** See README.md → "Resource Management"

---

## Cheat Sheet: Common Commands

```bash
# List available extractors
python -m scripts list-extractors

# Validate configs only (no extraction)
python -m scripts validate --dataset-config d.py --extraction-config e.py

# Run extraction with custom log file
python -m scripts extract --dataset-config d.py --extraction-config e.py --log-file my_log.jsonl

# Suppress console output (only file logging)
python -m scripts extract --dataset-config d.py --extraction-config e.py --no-console

# Convert Python config to YAML (in Python script)
from scripts.config.dataset_config import DatasetConfig
config = DatasetConfig.load_yaml("config.yml")  # Load from YAML
config.save_yaml("output.yml")                  # Save to YAML
```

---

## Support

- **Errors in logs:** Check `extraction_TIMESTAMP.jsonl` for detailed error messages
- **Performance questions:** See resource monitoring logs with `jq 'select(.stage=="resource_checkpoint")'`
- **Issues:** Open GitHub issue with logs and config files

**Happy extracting! 🎵**
