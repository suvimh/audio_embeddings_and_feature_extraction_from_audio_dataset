"""
Extractor Registry
==================
Central registry mapping extractor names to factory functions and metadata.
"""

from typing import Dict, Callable, Optional, Any
from scripts.utils import FeatureExtractorConfig


def make_extractor(
    name: str,
    sample_rate: int = 16000,
    extractor_params: Optional[Dict[str, Any]] = None,
) -> FeatureExtractorConfig:
    """
    Factory function to instantiate an extractor by name.

    Parameters
    ----------
    name : str
        Extractor name: 'vggish', 'clap', 'whisper', or 'opensmile'.
    sample_rate : int
        Sample rate for audio. Default: 16000 Hz.
    extractor_params : dict | None
        Optional per-extractor parameters (e.g., {'version': '2024'} for CLAP).

    Returns
    -------
    FeatureExtractorConfig
        Instantiated extractor configuration.

    Raises
    ------
    ValueError
        If extractor name is not recognized.
    """
    if extractor_params is None:
        extractor_params = {}

    name_lower = name.lower().strip()

    if name_lower == "vggish":
        from scripts.dl_embeddings.extract_vggish_embeddings import (
            make_vggish_extractor,
        )

        return make_vggish_extractor(sample_rate=sample_rate)

    elif name_lower == "clap":
        from scripts.dl_embeddings.extract_clap_embeddings import make_clap_extractor

        version = extractor_params.get("version", "2023")
        return make_clap_extractor(version=version, sample_rate=sample_rate)

    elif name_lower == "whisper":
        from scripts.dl_embeddings.extract_whisper_embeddings import (
            make_whisper_extractor,
        )

        model_name = extractor_params.get("model_name", "openai/whisper-base")
        device = extractor_params.get("device", None)
        return make_whisper_extractor(model_name=model_name, device=device)

    elif name_lower == "opensmile":
        from scripts.feature_sets.extract_opensmile import make_opensmile_extractor

        feature_set = extractor_params.get("feature_set", "ComParE_2016")
        feature_level = extractor_params.get("feature_level", "Functionals")
        return make_opensmile_extractor(
            feature_set=feature_set,
            feature_level=feature_level,
            sample_rate=sample_rate,
        )

    else:
        valid_names = {"vggish", "clap", "whisper", "opensmile"}
        raise ValueError(f"Unknown extractor '{name}'. Valid names: {valid_names}")


# Registry metadata for introspection and documentation
EXTRACTOR_METADATA = {
    "vggish": {
        "description": "VGGish audio embeddings (TensorFlow Hub)",
        "embedding_dim": 128,
        "default_params": {},
        "params": {
            # VGGish is fixed, no parameters
        },
    },
    "clap": {
        "description": "MS-CLAP audio-text embeddings (PyTorch)",
        "embedding_dims": {"2023": 512, "2024": 1024},
        "default_params": {"version": "2023"},
        "params": {
            "version": {
                "type": "str",
                "choices": ["2023", "2024"],
                "default": "2023",
                "description": "CLAP model version",
            }
        },
    },
    "whisper": {
        "description": "Whisper audio-text encoder (OpenAI)",
        "embedding_dims": {
            "tiny": 384,
            "base": 512,
            "small": 768,
            "medium": 1024,
            "large": 1280,
        },
        "default_params": {"model_name": "openai/whisper-base"},
        "params": {
            "model_name": {
                "type": "str",
                "choices": [
                    "openai/whisper-tiny",
                    "openai/whisper-base",
                    "openai/whisper-small",
                    "openai/whisper-medium",
                    "openai/whisper-large",
                ],
                "default": "openai/whisper-base",
                "description": "Whisper model variant",
            },
            "device": {
                "type": "str or None",
                "choices": ["cuda", "mps", "cpu", None],
                "default": None,
                "description": "Force device (None = auto-detect)",
            },
        },
    },
    "opensmile": {
        "description": "Traditional acoustic features (openSMILE)",
        "feature_sets": {
            "ComParE_2016": 6373,
            "eGeMAPSv02": 88,
            "GeMAPSv01b": 62,
        },
        "default_params": {
            "feature_set": "ComParE_2016",
            "feature_level": "Functionals",
        },
        "params": {
            "feature_set": {
                "type": "str",
                "choices": ["ComParE_2016", "eGeMAPSv02", "GeMAPSv01b"],
                "default": "ComParE_2016",
                "description": "Feature set to extract",
            },
            "feature_level": {
                "type": "str",
                "choices": ["Functionals"],
                "default": "Functionals",
                "description": "Feature level (functionals = statistics over time)",
            },
        },
    },
}


def get_extractor_info(name: str) -> Dict[str, Any]:
    """
    Get metadata about an extractor.

    Parameters
    ----------
    name : str
        Extractor name.

    Returns
    -------
    dict
        Metadata dictionary.

    Raises
    ------
    ValueError
        If extractor name is not recognized.
    """
    name_lower = name.lower().strip()
    if name_lower not in EXTRACTOR_METADATA:
        raise ValueError(f"Unknown extractor: {name}")
    return EXTRACTOR_METADATA[name_lower]


def list_extractors() -> Dict[str, str]:
    """
    List all available extractors with short descriptions.

    Returns
    -------
    dict
        Mapping of extractor name to description.
    """
    return {name: info["description"] for name, info in EXTRACTOR_METADATA.items()}
