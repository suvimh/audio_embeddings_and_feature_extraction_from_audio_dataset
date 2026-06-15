# ---------------------------------------------------------------------------
# TUNI Emotion Dataset Config
# ---------------------------------------------------------------------------

from pathlib import Path

from scripts.config.dataset_config import DatasetConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_INPUT_DIR = str(_REPO_ROOT / "wav")


def tuni_emotion_dataset_config(
    root_dir: str = DATA_INPUT_DIR,
    frame_duration: float = 3.0,
    sample_rate: int = 16000,
    overlap_percentage: float = 0.25,
) -> DatasetConfig:
    """
    TUNI dataset.

    Folder structures
    -----------------
    TUNI singers (wav):
        Levels: singer / genre / emotion / wav files

    All recording files are processed (all .wav).

    Label columns in output DataFrame
    ----------------------------------
    singer              : participant name, e.g. 'Anu_Mattila'
    genre               : genre of the song, e.g. 'pop', 'classical'
    emotion             : emotion of the song, e.g. 'joy', 'sadness', 'anger', 'gentleness', 'neutral'
    """
    return DatasetConfig(
        name="tuni_emotion_dataset",
        root_dir=root_dir,
        # Fallback schema + union of all column names across subgroups
        level_names=["singer", "genre", "emotion"],
        participant_level=0,
        level_allowed_values={
            "singer": ["Anu_Mattila", "Anniina_Honkala", "Bianca_Hösli", "Elina_Lahtinen", "Liisi_Petterson", "Maarit_Aura", "Marja_Erdogan", "Saga_Ohlsson_1", "Sanna_Vähälä", "Tero_Ikävalko", "Tommi_Grönberg", "Tua_Hakanpää", "Veera_Tapanainen"],
            "genre": [ "pop", "classical"],
            "emotion": ["joy", "sadness", "anger", "gentleness", "neutral"],
        },
        frame_duration=frame_duration,
        sample_rate=sample_rate,
        overlap_percentage=overlap_percentage,
    )


tuni_emotion_dataset = tuni_emotion_dataset_config()
