"""
Whisper Feature Extractor
=========================
Uses only the Whisper encoder to extract embeddings — no classifier head,
no label handling. Plugs into the same DatasetConfig / utils pipeline as VGGish.

Embedding: mean-pooled encoder hidden states → shape (d_model,)
  whisper-tiny   : 384
  whisper-base   : 512
  whisper-small  : 768
  whisper-medium : 1024
  whisper-large  : 1280
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from transformers import WhisperModel, WhisperFeatureExtractor
from scripts.utils import (
    FeatureExtractorConfig,
)

# ---------------------------------------------------------------------------
# Whisper encoder wrapper
# ---------------------------------------------------------------------------


class WhisperEncoder(nn.Module):
    """
    Whisper encoder only — no decoder, no classifier.
    Returns mean-pooled hidden states as a 1D embedding.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.encoder.gradient_checkpointing = False
        self.d_model = self.model.config.d_model

    @torch.no_grad()
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        input_features : (batch, 80, time) or (80, time) — mel spectrogram
        returns        : (batch, d_model) or (d_model,) — mean-pooled embedding
        """
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)  # add batch dim
        encoder_outputs = self.model.encoder(input_features=input_features)
        hidden = encoder_outputs.last_hidden_state  # (batch, time, d_model)
        pooled = hidden.mean(dim=1)  # (batch, d_model)
        return pooled.squeeze(0)  # (d_model,) for single input


# ---------------------------------------------------------------------------
# Extractor factory
# ---------------------------------------------------------------------------


def make_whisper_extractor(
    model_name: str = "openai/whisper-base",
    device: Optional[str] = None,
) -> FeatureExtractorConfig:
    """
    Load Whisper encoder and return a FeatureExtractorConfig.

    Parameters
    ----------
    model_name : HuggingFace model string, e.g. 'openai/whisper-base'
    device     : 'cuda', 'mps', 'cpu', or None (auto-detect)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[INFO] Loading Whisper encoder ({model_name}) on {device}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    encoder = WhisperEncoder(model_name).to(device)
    encoder.eval()

    # Infer embedding dim from model config
    embedding_dim = encoder.d_model

    def extract(audio: np.ndarray) -> Optional[np.ndarray]:
        """
        audio : float32 numpy array at 16kHz (any length — Whisper pads/truncates to 30s internally)
        """
        try:
            inputs = feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device)  # (1, 80, time)
            embedding = encoder(input_features)
            return embedding.cpu().numpy()
        except Exception as e:
            print(f"[WARN] Whisper extraction failed: {e}")
            return None

    return FeatureExtractorConfig(
        name=f"whisper_{model_name.split('/')[-1]}",  # e.g. 'whisper_whisper-base'
        extract_fn=extract,
        embedding_dim=embedding_dim,
        min_input_samples=None,  # Whisper handles short audio internally via padding
    )
