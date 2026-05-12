"""
OpenSMILE ComParE Feature Extractor
=====================================
Extracts the ComParE 2016 feature set (6373 features) using opensmile.
Plugs into the same DatasetConfig / utils pipeline as VGGish, Whisper, CLAP.

ComParE is designed specifically for paralinguistic and voice quality tasks —
it captures MFCCs, prosody, voice quality (jitter, shimmer, HNR), spectral
features, and more. Good fit for postural/articulatory condition classification.

"""

import numpy as np
import opensmile
from typing import Optional
from scripts.utils import FeatureExtractorConfig, extract_dataset_features

# ---------------------------------------------------------------------------
# Available feature sets (most useful for your task)
# ---------------------------------------------------------------------------
# opensmile.FeatureSet.ComParE_2013   : 65 / 65 / 6373 features — full paralinguistic set
# opensmile.FeatureSet.eGeMAPSv02     : 25 / - / 88 features  — compact, interpretable
# opensmile.FeatureSet.GeMAPSv01b     : 18 / - / 62 features  — minimal, fast


def make_opensmile_extractor(
    feature_set: opensmile.FeatureSet = opensmile.FeatureSet.ComParE_2016,
    feature_level: opensmile.FeatureLevel = opensmile.FeatureLevel.Functionals,
    sample_rate: int = 16000,
) -> FeatureExtractorConfig:
    """
    Load an opensmile feature extractor and return a FeatureExtractorConfig.

    Parameters
    ----------
    feature_set   : which opensmile feature set to use.
                    ComParE_2013 (6373d) is the default — broadest coverage.
                    eGeMAPSv02 (88d) is a good compact alternative.
    feature_level : Functionals = one vector per clip (statistics over time).
                    This is what you want for frame-level classification.
    sample_rate   : must match what load_and_frame_audio resamples to.

    Output dim:
        ComParE_2013 : 6373
        eGeMAPSv02   : 88
        GeMAPSv01b   : 62
    """
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )

    # Infer embedding dim from a dummy signal
    dummy = np.zeros(sample_rate, dtype=np.float32)
    dummy_out = smile.process_signal(dummy, sample_rate)
    embedding_dim = dummy_out.shape[1]

    feature_set_name = feature_set.name.lower().replace("_", "-")
    print(f"[INFO] OpenSMILE {feature_set_name}: {embedding_dim} features")

    def extract(audio: np.ndarray) -> Optional[np.ndarray]:
        try:
            # opensmile expects float32, 1D
            audio = audio.astype(np.float32)
            result = smile.process_signal(audio, sample_rate)
            return result.values[0].astype(np.float32)  # (n_features,)
        except Exception as e:
            print(f"[WARN] OpenSMILE extraction failed: {e}")
            return None

    return FeatureExtractorConfig(
        name=f"opensmile-{feature_set_name}",
        extract_fn=extract,
        embedding_dim=embedding_dim,
        min_input_samples=None,
    )
