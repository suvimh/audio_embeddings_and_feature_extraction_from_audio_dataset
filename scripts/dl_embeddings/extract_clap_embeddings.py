"""
CLAP Feature Extractor
======================
Uses MS-CLAP audio encoder to extract embeddings.
Plugs into the same DatasetConfig / utils pipeline as VGGish and Whisper.

Embedding dim:
  CLAP 2023 : 512
  CLAP 2024 : 1024

Note: msclap requires file paths rather than raw arrays, so each frame is
written to a temp file, embedded, then deleted. This is unavoidable with
the msclap API but is kept as lightweight as possible (one tmp file reused
per extractor instance).
"""

import os
import numpy as np
import tempfile
import soundfile as sf
import torch
from typing import Optional
from msclap import CLAP

from scripts.utils import FeatureExtractorConfig, extract_dataset_features

# ---------------------------------------------------------------------------
# CLAP embedding dims by version
# ---------------------------------------------------------------------------

CLAP_EMBEDDING_DIMS = {
    "2023": 512,
    "2024": 1024,
}


# ---------------------------------------------------------------------------
# Extractor factory
# ---------------------------------------------------------------------------


def make_clap_extractor(
    version: str = "2023",
    sample_rate: int = 16000,
) -> FeatureExtractorConfig:
    """
    Load MS-CLAP and return a FeatureExtractorConfig.

    Parameters
    ----------
    version     : CLAP model version — '2023' or '2024'
    sample_rate : sample rate used when writing temp wav files (must match
                  what load_and_frame_audio resamples to — default 16000)
    """
    use_cuda = torch.cuda.is_available()
    print(f"[INFO] Loading CLAP {version} (cuda={use_cuda})")
    clap_model = CLAP(version=version, use_cuda=use_cuda)

    embedding_dim = CLAP_EMBEDDING_DIMS.get(version, 512)

    # One persistent temp file, reused for every frame to avoid repeated
    # file creation overhead
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    def extract(audio: np.ndarray) -> Optional[np.ndarray]:
        try:
            # Write frame to temp file
            sf.write(tmp_path, audio, samplerate=sample_rate)

            emb = clap_model.get_audio_embeddings([tmp_path])

            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            if emb.ndim > 1:
                emb = emb[0]  # (1, dim) -> (dim,)

            return emb.astype(np.float32)

        except Exception as e:
            print(f"[WARN] CLAP extraction failed: {e}")
            return None

    # Register cleanup so the temp file is removed when Python exits
    import atexit

    atexit.register(lambda: os.unlink(tmp_path) if os.path.exists(tmp_path) else None)

    return FeatureExtractorConfig(
        name=f"clap-{version}",  # e.g. vte_dataset_clap-2023_3.0s.parquet
        extract_fn=extract,
        embedding_dim=embedding_dim,
        min_input_samples=None,  # CLAP handles short audio internally
    )
