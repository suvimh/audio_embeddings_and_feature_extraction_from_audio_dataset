"""
Example Extraction Configuration
=================================
This is a simple example showing how to define an extraction configuration.
Customize this for your own extraction needs.
"""

from pathlib import Path
from scripts.config.extraction_config import ExtractionConfig

# Example 1: Extract VGGish embeddings only with 3-second frames
vggish_only = ExtractionConfig(
    extractors=["vggish"],
    output_dir="/path/to/output/vggish",
    window_lengths=[3.0],
    batch_size="auto",
    gpu_cache_cleanup=False,
    num_workers=4,
)


# Example 2: Multi-window extraction with multiple embeddings
multi_window_multi_embeddings = ExtractionConfig(
    extractors=["vggish", "clap", "whisper"],
    output_dir="/path/to/output/multi_embeddings",
    window_lengths=[3.0, 5.0],  # Will extract for both 3s and 5s frames
    extractor_params={
        "clap": {"version": "2024"},  # Use CLAP 2024 model (1024-dim)
        "whisper": {"model_name": "openai/whisper-base"},  # Specify model variant
    },
    batch_size="auto",
    gpu_cache_cleanup=True,  # Enable cache cleanup for very large datasets
    num_workers=4,
)


# Example 3: Mix DL embeddings and traditional features
mixed_embeddings_and_features = ExtractionConfig(
    extractors=["vggish", "clap", "opensmile"],
    output_dir="/path/to/output/combined",
    window_lengths=[3.0],
    extractor_params={
        "clap": {"version": "2023"},
        "opensmile": {
            "feature_set": "eGeMAPSv02",  # Smaller, interpretable feature set
            "feature_level": "Functionals",
        },
    },
    batch_size=10,  # Explicit batch size for precise memory control
    gpu_cache_cleanup=False,
    num_workers=4,
)


# Example 4: OpenSMILE-only extraction (CPU-friendly, no GPU needed)
opensmile_only = ExtractionConfig(
    extractors=["opensmile"],
    output_dir="/path/to/output/acoustic_features",
    window_lengths=[2.0, 3.0, 5.0],  # Multiple window lengths
    extractor_params={
        "opensmile": {
            "feature_set": "ComParE_2016",  # Comprehensive feature set (6373 features)
            "feature_level": "Functionals",
        },
    },
    batch_size="auto",
    gpu_cache_cleanup=False,
    num_workers=8,  # Can use more workers since CPU-only
)


# Example 5: All extractors with comprehensive parameter control
comprehensive_extraction = ExtractionConfig(
    extractors=["vggish", "clap", "whisper", "opensmile"],
    output_dir="/path/to/output/comprehensive",
    window_lengths=[3.0, 5.0, 7.0],
    extractor_params={
        "clap": {"version": "2024"},
        "whisper": {"model_name": "openai/whisper-small"},
        "opensmile": {"feature_set": "ComParE_2016"},
    },
    batch_size="auto",  # Auto-detect based on RAM
    gpu_cache_cleanup=True,
    num_workers=4,
)


# Example 6: MFCC-only extraction (traditional spectral features)
mfcc_only = ExtractionConfig(
    extractors=["mfcc"],
    output_dir="/path/to/output/mfcc",
    window_lengths=[1.0, 3.0],
    extractor_params={"mfcc": {"n_mfcc": 13}},
    batch_size="auto",
    gpu_cache_cleanup=False,
    num_workers=4,
)
