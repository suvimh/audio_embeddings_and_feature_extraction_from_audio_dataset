"""
Example Dataset Configuration
=============================
This is a simple example showing how to define a dataset configuration.
Customize this for your own dataset.
"""

from pathlib import Path
from scripts.config.dataset_config import DatasetConfig

# Example 1: Simple flat dataset structure
# Folder structure:
#   /path/to/dataset/
#     speaker_001/audio1.wav, audio2.wav
#     speaker_002/audio3.wav, audio4.wav
simple_dataset = DatasetConfig(
    name="simple_audio_dataset",
    root_dir="/path/to/dataset",
    level_names=["speaker"],
    participant_level=0,
    sample_rate=16000,
    frame_duration=3.0,
    overlap_percentage=0.25,
    enable_silence_trimming=True,
    silence_threshold_db=-40.0,
    silence_min_duration_ms=500.0,
    normalization_mode="peak",
)


# Example 2: Hierarchical dataset (like VTE)
# Folder structure (professional/intermediate):
#   /path/to/dataset/
#     professional/
#       speaker_001/
#         phonation_A/condition_X/recording_1.wav
#         phonation_B/condition_Y/recording_2.wav
#     experienced/
#       speaker_002/
#         phonation_A/condition_X/recording_1.wav
hierarchical_dataset = DatasetConfig(
    name="voice_quality_dataset",
    root_dir="/path/to/hierarchical/dataset",
    level_names=["experience", "speaker", "phonation", "condition", "take"],
    participant_level=1,  # speaker is at depth 1
    sample_rate=16000,
    frame_duration=3.0,
    overlap_percentage=0.25,
    enable_silence_trimming=True,
    silence_threshold_db=-35.0,
    silence_min_duration_ms=300.0,
    normalization_mode="rms",
    file_suffix_filter="-mic-audio",  # Only process files ending with this
)


# Example 3: Dataset with irregular subgroup structures
# Use this when different top-level folders have different hierarchies
irregular_dataset = DatasetConfig(
    name="mixed_structure_dataset",
    root_dir="/path/to/mixed/dataset",
    level_names=["group", "speaker", "session", "recording"],  # default
    participant_level=1,
    subgroup_configs={
        "senior_speakers": ["group", "speaker", "recording"],  # No session level
        "junior_speakers": ["group", "speaker", "session", "recording"],
    },
    subgroup_split_level=0,  # Check 'group' folder to determine which config to use
    sample_rate=16000,
    frame_duration=5.0,
    overlap_percentage=0.50,
    enable_silence_trimming=False,  # Assume clean audio
    normalization_mode="lufs",
)
