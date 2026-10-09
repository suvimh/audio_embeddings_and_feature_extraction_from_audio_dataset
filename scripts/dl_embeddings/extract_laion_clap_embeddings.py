"""
LAION-CLAP Feature Extractor
============================
Uses the HuggingFace Transformers LAION-CLAP audio encoder (ClapModel) to
extract audio embeddings. Plugs into the same DatasetConfig / utils pipeline
as VGGish, MS-CLAP and Whisper.

Mirrors the fine-tuning pipeline in audio_data_benchmarking_mml_lab
(scripts/models/clap.py): each frame is encoded by the ClapProcessor and
embedded via ClapModel.get_audio_features, i.e. CLAP's projected audio
embedding (same embedding the classifier head is trained on).

Embedding dim:
  laion/clap-htsat-fused : 512 (model.config.projection_dim)

LAION-CLAP is trained at 48 kHz; passing frames at any other sample rate
raises a ValueError (set extractor_params['laion-clap']['sample_rate']).
"""

import numpy as np
import torch
from typing import Optional

from transformers import ClapModel, ClapProcessor

from scripts.utils import FeatureExtractorConfig


def make_laion_clap_extractor(
    model_name: str = "laion/clap-htsat-fused",
    sample_rate: int = 48000,
    device: Optional[str] = None,
) -> FeatureExtractorConfig:
    """
    Load LAION-CLAP and return a FeatureExtractorConfig.

    Parameters
    ----------
    model_name  : HuggingFace LAION-CLAP model identifier, default
                  'laion/clap-htsat-fused'.
    sample_rate : sample rate the frames are resampled to before embedding.
                  LAION-CLAP is trained at 48 kHz, so any other value raises
                  a ValueError.
    device      : 'cuda', 'mps', 'cpu', or None (auto-detect).
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[INFO] Loading LAION-CLAP ({model_name}) on {device}")

    processor = ClapProcessor.from_pretrained(model_name)
    feat_sr = int(processor.feature_extractor.sampling_rate)
    if sample_rate != feat_sr:
        raise ValueError(
            f"LAION-CLAP requires {feat_sr} Hz audio, got sample_rate={sample_rate}. "
            f"Set extractor_params['laion-clap']['sample_rate'] = {feat_sr}."
        )

    model = ClapModel.from_pretrained(model_name).to(device)
    model.eval()

    # LAION-CLAP's final projected audio embedding dimension
    embedding_dim = int(model.config.projection_dim)

    @torch.no_grad()
    def extract(audio: np.ndarray) -> Optional[np.ndarray]:
        """
        audio : float32 numpy array at 48 kHz (a single frame / window).
                Returns the projected audio embedding of shape
                (embedding_dim,) — the same embedding used by the
                fine-tuning classifier head.
        """
        try:
            inputs = processor(
                audio=[np.asarray(audio, dtype=np.float32)],
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device)  # (1, 1, 1024, 64)
            is_longer = inputs.is_longer.to(device)  # (1,)

            audio_features = model.get_audio_features(
                input_features=input_features,
                is_longer=is_longer,
            )
            if audio_features.ndim > 1:
                audio_features = audio_features[0]

            return audio_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[WARN] LAION-CLAP extraction failed: {e}")
            return None

    return FeatureExtractorConfig(
        name=f"laion-{model_name.split('/')[-1]}",  # e.g. 'laion-clap-htsat-fused'
        extract_fn=extract,
        embedding_dim=embedding_dim,
        min_input_samples=None,  # short frames are zero-padded upstream via short_audio_handling='pad'
    )