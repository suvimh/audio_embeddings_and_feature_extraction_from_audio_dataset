# ---------------------------------------------------------------------------
# TUNI Emotion Extraction Config
# ---------------------------------------------------------------------------

from pathlib import Path

from scripts.config.extraction_config import ExtractionConfig

_DATA_DIR = Path(__file__).resolve().parent
DATA_OUTPUT_DIR = str(_DATA_DIR / "embeddings")

tuni_emotion_extraction = ExtractionConfig(
    extractors=["opensmile", "clap", "whisper"],
    output_dir=DATA_OUTPUT_DIR,
    window_lengths=[3.0, 0.5],  # Extract for both 3s and 0.5s frames
    batch_size="auto",
)
