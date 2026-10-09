# ---------------------------------------------------------------------------
# VocalSet Extraction Config — 3s non-overlapping rerun
# ---------------------------------------------------------------------------

from scripts.config.extraction_config import ExtractionConfig

DATA_OUTPUT_DIR = (
    "/home/suvihaara/Documents/PhD/DATA/VocalSet/VocalSet_fixed/embeddings/3sec_nonoverlap/"
)

vocalset_extraction = ExtractionConfig(
    extractors=[
        # "mfcc",
        # "opensmile",
        "vggish",
        # "ms-clap",
        # "laion-clap",
        # "whisper",
    ],
    output_dir=DATA_OUTPUT_DIR,
    window_lengths=[3.0],
    # LAION-CLAP is trained at 48 kHz; all other extractors use the dataset
    # config's sample_rate (16 kHz).
    extractor_params={
        "laion-clap": {"sample_rate": 48000},
    },
    batch_size="auto",
)