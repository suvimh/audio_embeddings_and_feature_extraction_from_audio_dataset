# ---------------------------------------------------------------------------
# VocalSet Extraction Config
# ---------------------------------------------------------------------------

from scripts.config.extraction_config import ExtractionConfig

DATA_OUTPUT_DIR = (
    "/home/suvihaara/Documents/PhD/DATA/VocalSet/VocalSet_fixed/embeddings/"
)

vocalset_extraction = ExtractionConfig(
    # extractors=["opensmile", "clap", "whisper", "vggish"],
    extractors=["clap", "vggish"],
    output_dir=DATA_OUTPUT_DIR,
    window_lengths=[3.0, 0.5],  # Extract for both 3s and 0.5s frames
    batch_size="auto",
)
