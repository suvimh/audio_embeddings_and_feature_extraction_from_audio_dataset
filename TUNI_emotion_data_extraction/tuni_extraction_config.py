# ---------------------------------------------------------------------------
# TUNI Emotion Extraction Config
# ---------------------------------------------------------------------------

from scripts.config.extraction_config import ExtractionConfig

DATA_OUTPUT_DIR = (
    "/Users/pubert/audio_new/audio_embeddings_and_feature_extraction_from_audio_dataset/TUNI_emotion_dataset_og/embeddings/"
)

tuni_emotion_extraction = ExtractionConfig(
    extractors=["opensmile", "clap", "whisper", "vggish"],
    output_dir=DATA_OUTPUT_DIR,
    window_lengths=[3.0, 0.5],  # Extract for both 3s and 0.5s frames
    batch_size="auto",
)
