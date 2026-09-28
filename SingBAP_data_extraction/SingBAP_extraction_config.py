# ---------------------------------------------------------------------------
# VTE Extraction Config
# ---------------------------------------------------------------------------

DATA_OUTPUT_DIR = "/home/suvihaara/Documents/PhD/DATA/VTE/embeddings/"

from scripts.config.extraction_config import ExtractionConfig

SingBAP_extraction = ExtractionConfig(
    extractors=["opensmile", "clap", "whisper", "vggish"],
    output_dir=DATA_OUTPUT_DIR,
    window_lengths=[3.0, 0.5],  # Extract for both 3s and 0.5s frames
    batch_size="auto",
)
