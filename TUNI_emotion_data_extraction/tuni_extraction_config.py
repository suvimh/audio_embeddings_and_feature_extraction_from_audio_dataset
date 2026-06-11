# ---------------------------------------------------------------------------
# TUNI Emotion Extraction Config
# ---------------------------------------------------------------------------

from scripts.config.extraction_config import ExtractionConfig

DATA_OUTPUT_DIR = (
    "/Users/pubert/Downloads/SkyNote/OTHER_DATASETS/TUNI_emotion_dataset/embeddings/"
)

tuni_extraction = ExtractionConfig(
    extractors=["opensmile"],
    output_dir=DATA_OUTPUT_DIR,
    window_lengths=[3.0, 0.5],
    extractor_params={
        "opensmile": {
            "feature_set": "ComParE_2016",
            "feature_level": "Functionals",
        },
    },
    batch_size="auto",
)
