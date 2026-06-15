# ---------------------------------------------------------------------------
# TUNI Emotion Extraction Config
# ---------------------------------------------------------------------------

from scripts.config.extraction_config import ExtractionConfig

DATA_OUTPUT_DIR = (
    "/Users/pubert/Downloads/SkyNote/OTHER_DATASETS/TUNI_emotion_dataset/embeddings/tuni_embeddings")

tuni_emotion_extraction = ExtractionConfig(
    extractors=["opensmile", "clap", "whisper", "vggish"], #add extractor params? also re-add clap and vggish?
    output_dir=DATA_OUTPUT_DIR,
    window_lengths=[3.0, 0.5],  # Extract for both 3s and 0.5s frames
    batch_size="auto",
)
