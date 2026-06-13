# ---------------------------------------------------------------------------
# VocalSet Dataset Config
# ---------------------------------------------------------------------------

from scripts.config.dataset_config import DatasetConfig

DATA_INPUT_DIR = ""


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
    vocalset singers:
        <root>/female6/scales/fast_forte/f6_scales_c_fast_forte_a.wav
        Levels: singer_ID / exercise_type / vocal_technique / takes (all) .. _a.wav

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
