# ---------------------------------------------------------------------------
# VocalSet Extraction Config
# ---------------------------------------------------------------------------

from scripts.config.extraction_config import ExtractionConfig

DATA_OUTPUT_DIR = (
    "/home/suvihaara/Documents/PhD/DATA/VocalSet/VocalSet_fixed/embeddings/3sec/"
)

vocalset_extraction = ExtractionConfig(
    extractors=[
        "mfcc",
        "opensmile",
        "vggish",
        "clap",
        "whisper",
    ],
    output_dir=DATA_OUTPUT_DIR,
    window_lengths=[3.0],
    batch_size="auto",
)
