# Development Guide

This document explains the architecture and how to extend the pipeline with new extractors or features.

## Architecture Overview

### Configuration System

**DatasetConfig** (`scripts/config/dataset_config.py`):

- Describes folder structure and audio preprocessing parameters
- Validates configurations on instantiation
- Supports YAML serialization (see below)

**ExtractionConfig** (`scripts/config/extraction_config.py`):

- Specifies which extractors to run, window lengths, and output settings
- Supports YAML/JSON serialization and validation

### Extractor System

**FeatureExtractorConfig** (in `scripts/utils.py`):

```python
@dataclass
class FeatureExtractorConfig:
    name: str                           # Extractor name
    extract_fn: Callable               # Function: audio -> embedding
    embedding_dim: int                 # Output dimension
    min_input_samples: Optional[int]    # Minimum audio length required
```

**Extractor Factory Pattern** (`scripts/extractors_registry.py`):

```python
def make_<name>_extractor(...) -> FeatureExtractorConfig:
    """Load model and return extractor config."""
    model = load_model(...)
    return FeatureExtractorConfig(
        name="<name>",
        extract_fn=lambda audio: model.extract(audio),
        embedding_dim=<dim>,
        min_input_samples=<samples>,
    )
```

### Extraction Pipeline

**extract_dataset_features()** (`scripts/utils.py`):

- Orchestrates per-file extraction
- Manages batching and HDF5 accumulation
- Checkpoint-based crash recovery
- Converts HDF5 to Parquet output

**run_extraction()** (`scripts/run_extraction.py`):

- Top-level orchestrator
- Loops over (window_length, extractor) pairs
- Manages logging and resource monitoring
- Returns extraction summary

---

## Adding a New Extractor

### Step 1: Create Extraction Module

Create `scripts/<category>/extract_<name>.py`:

```python
"""
<Name> Feature Extractor
========================
"""

import numpy as np
from scripts.utils import FeatureExtractorConfig


def make_<name>_extractor(
    sample_rate: int = 16000,
    <other_params>: <type> = <default>,
) -> FeatureExtractorConfig:
    """
    Factory function to instantiate <name> extractor.

    Parameters
    ----------
    sample_rate : int
        Sample rate of input audio.
    <other_params> : <type>
        Parameter description.

    Returns
    -------
    FeatureExtractorConfig
        Extractor configuration.
    """
    # Load model
    model = load_model(sample_rate=sample_rate, <other_params>=<other_params>)

    def extract_fn(audio: np.ndarray) -> np.ndarray:
        """
        Extract embedding from audio frame.

        Parameters
        ----------
        audio : np.ndarray
            Audio frame (mono, float32, < 1 minute typically).

        Returns
        -------
        np.ndarray
            Embedding vector (1D array of floats).
        """
        embedding = model.extract(audio)

        # Flatten if multi-dimensional
        if embedding.ndim > 1:
            embedding = embedding.mean(axis=0)

        return embedding.astype(np.float32)

    return FeatureExtractorConfig(
        name="<name>",
        extract_fn=extract_fn,
        embedding_dim=<output_dimension>,
        min_input_samples=<min_samples_or_none>,  # Optional: min audio length
    )
```

### Step 2: Register Extractor

Edit `scripts/extractors_registry.py`:

```python
def make_extractor(
    name: str,
    sample_rate: int = 16000,
    extractor_params: Optional[Dict[str, Any]] = None,
) -> FeatureExtractorConfig:
    """..."""
    if extractor_params is None:
        extractor_params = {}

    name_lower = name.lower().strip()

    # ... existing extractors ...

    elif name_lower == "<name>":
        from scripts.<category>.extract_<name> import make_<name>_extractor
        <param1> = extractor_params.get("<param1>", "<default1>")
        <param2> = extractor_params.get("<param2>", "<default2>")
        return make_<name>_extractor(<param1>=<param1>, <param2>=<param2>)

    else:
        # ...
```

And add metadata:

```python
EXTRACTOR_METADATA = {
    # ... existing ...

    "<name>": {
        "description": "Human-readable description",
        "embedding_dim": <dim>,  # or dict if multiple variants
        "default_params": {
            "<param1>": "<default1>",
        },
        "params": {
            "<param1>": {
                "type": "str",
                "choices": ["option1", "option2"],
                "default": "option1",
                "description": "Parameter description"
            },
        }
    }
}
```

### Step 3: Add to Example Configs

Edit `examples/extraction_config_example.py`:

```python
example_with_new_extractor = ExtractionConfig(
    extractors=["<name>"],
    output_dir="/path/to/output/<name>",
    extractor_params={
        "<name>": {
            "<param1>": "<value1>",
        }
    },
)
```

### Step 4: Test Extraction

```python
from scripts.config.dataset_config import DatasetConfig
from scripts.extractors_registry import make_extractor
from scripts.utils import extract_dataset_features

dataset = DatasetConfig(
    name="test",
    root_dir="/path/to/small/dataset",
    level_names=["speaker"],
    participant_level=0,
)

extractor = make_extractor("<name>", sample_rate=16000)

df = extract_dataset_features(
    dataset_config=dataset,
    extractor_config=extractor,
    output_dir="/tmp/test_output",
    batch_size=1,
)

print(f"Extracted {len(df)} frames")
print(f"Embedding shape: {df['embedding'].iloc[0].shape}")
```

### Step 5: Document

Update `README.md`:

- Add row to "Available Extractors" table
- Add parameter options to "Extractor Parameters" section
- Update CLI examples if needed

---

## Extending Dataset Configuration

To add new audio preprocessing parameter:

1. **Add field to DatasetConfig**:

   ```python
   @dataclass
   class DatasetConfig:
       # ... existing fields ...
       my_new_param: str = "default_value"
   ```

2. **Add validation in **post_init****:

   ```python
   def __post_init__(self) -> None:
       # ... existing validations ...
       if self.my_new_param not in {"valid1", "valid2"}:
           raise ValueError(f"Invalid my_new_param: {self.my_new_param}")
   ```

3. **Update docstring** with parameter description

4. **Use in load_and_frame_audio()**:

   ```python
   def load_and_frame_audio(
       ...,
       my_new_param: str = "default_value",
   ):
       # Use my_new_param in processing
   ```

5. **Pass from extraction**:

   ```python
   # In extract_dataset_features()
   frames = load_and_frame_audio(
       ...,
       my_new_param=dataset_config.my_new_param,
   )
   ```

6. **Update examples**:
   - `examples/dataset_config_example.py`
   - `examples/dataset_config_example.yml`
   - `README.md` dataset config table

---

## Serialization: Python ↔ YAML

### Python to YAML

```python
from scripts.config.dataset_config import DatasetConfig

# Load from Python file
config = DatasetConfig.load_yaml("config.yml")

# Save to YAML
config.save_yaml("output.yml")
```

**Note:** YAML serialization methods need to be added to `DatasetConfig`:

```python
@dataclass
class DatasetConfig:
    # ... fields ...

    def save_yaml(self, path: str | Path) -> None:
        """Save to YAML."""
        import yaml
        data = asdict(self)
        data["root_dir"] = str(data["root_dir"])
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "DatasetConfig":
        """Load from YAML."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

---

## Testing

### Manual Integration Test

```bash
# 1. Create small test dataset
mkdir -p /tmp/test_data/speaker_001
cp /path/to/sample/audio.wav /tmp/test_data/speaker_001/

# 2. Create minimal configs
cat > /tmp/test_dataset.py << 'EOF'
from scripts.config.dataset_config import DatasetConfig
config = DatasetConfig(
    name="test",
    root_dir="/tmp/test_data",
    level_names=["speaker"],
    participant_level=0,
)
EOF

cat > /tmp/test_extraction.py << 'EOF'
from scripts.config.extraction_config import ExtractionConfig
config = ExtractionConfig(
    extractors=["vggish"],
    output_dir="/tmp/test_output",
    window_lengths=[3.0],
    batch_size=1,
)
EOF

# 3. Validate
python -m scripts validate --dataset-config /tmp/test_dataset.py --extraction-config /tmp/test_extraction.py

# 4. Run
python -m scripts extract --dataset-config /tmp/test_dataset.py --extraction-config /tmp/test_extraction.py

# 5. Check output
python << 'EOF'
from scripts.utils import load_parquet
df = load_parquet("/tmp/test_output/vggish_window_3s.parquet")
print(f"Extracted {len(df)} frames")
print(f"Columns: {df.columns.tolist()}")
print(f"Embedding shape: {df['embedding'].iloc[0].shape}")
EOF
```

---

## Performance Optimization

### Profiling Extraction

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run extraction
summary = run_extraction(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def extract_dataset_features(...):
    # ...
```

Run with:

```bash
python -m memory_profiler scripts/utils.py
```

---

## Common Issues & Solutions

### Issue: Extractor instantiation fails

**Solution:** Check that:

1. Extractor name matches registry exactly (case-insensitive)
2. Model can be downloaded/loaded (network, disk space)
3. Dependencies are installed (check requirements.txt)

### Issue: Runtime memory errors

**Solution:**

- Reduce `batch_size` in ExtractionConfig
- Enable `gpu_cache_cleanup=True`
- Reduce `num_workers` for I/O parallelism
- Check system RAM with `psutil.virtual_memory()`

### Issue: Slow extraction

**Solution:**

- Check CPU/GPU utilization in logs
- Profile to find bottleneck (I/O, model inference, memory allocation)
- Adjust `num_workers` and batch processing strategy

---

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/my-extractor`)
3. Implement extractor following patterns above
4. Test with small dataset
5. Update documentation and examples
6. Submit pull request with clear description

---

## References

- [Feature Extractor Config Pattern](scripts/utils.py#L15)
- [Registry Implementation](scripts/extractors_registry.py)
- [Extraction Pipeline](scripts/utils.py#L400)
- [CLI Interface](scripts/cli.py)
