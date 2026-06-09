import numpy as np
from typing import Optional
from scripts.utils import (
    FeatureExtractorConfig,
)


def make_vggish_extractor(sample_rate: int = 16000) -> FeatureExtractorConfig:
    """
    Load VGGish from TF Hub and return a FeatureExtractorConfig.
    
    Parameters
    ----------
    sample_rate : int, optional
        Ignored (VGGish internally resamples to 16kHz). Included for API consistency.
    """
    import tensorflow_hub as hub

    model = hub.load("https://tfhub.dev/google/vggish/1")

    def extract(audio: np.ndarray) -> Optional[np.ndarray]:
        try:
            embedding = model(audio)
            return embedding.numpy()
        except Exception as e:
            print(f"[WARN] VGGish extraction failed: {e}")
            return None

    return FeatureExtractorConfig(
        name="vggish",
        extract_fn=extract,
        embedding_dim=128,
        min_input_samples=16000,  # VGGish requires at least 1 second of audio
    )
