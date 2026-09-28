"""MFCC feature extractor compatible with the pipeline's `FeatureExtractorConfig`.

This module provides a factory `make_mfcc_extractor` that returns a
`FeatureExtractorConfig` suitable for use with
`scripts.utils.extract_dataset_features`. The extractor returns a 1-D
float32 vector per frame (mean MFCC across time frames) so it stores
cleanly in the repository's Parquet format.
"""

from typing import Optional
import numpy as np
import librosa
from scripts.utils import FeatureExtractorConfig


def make_mfcc_extractor(n_mfcc: int = 13, sample_rate: int = 16000) -> FeatureExtractorConfig:
    """
    Create an MFCC extractor config.

    Parameters
    ----------
    n_mfcc : int
        Number of MFCC coefficients to compute (output embedding dim).
    sample_rate : int
        Sample rate that frames are expected to be in (frames are resampled
        by `load_and_frame_audio` before being passed to the extractor).

    Returns
    -------
    FeatureExtractorConfig
        Config with `extract_fn(frame) -> np.ndarray` returning a 1-D
        `float32` vector of length `n_mfcc`.
    """

    def extract(frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            # librosa.feature.mfcc returns shape (n_mfcc, n_time)
            mfcc = librosa.feature.mfcc(y=frame.astype(np.float32), sr=sample_rate, n_mfcc=n_mfcc)
            # Reduce over time axis to a single vector (n_mfcc,)
            vec = mfcc.mean(axis=1).astype(np.float32)
            return vec
        except Exception as e:
            print(f"[WARN] MFCC extraction failed: {e}")
            return None

    return FeatureExtractorConfig(
        name=f"mfcc-{n_mfcc}d",
        extract_fn=extract,
        embedding_dim=n_mfcc,
        min_input_samples=None,
    )
