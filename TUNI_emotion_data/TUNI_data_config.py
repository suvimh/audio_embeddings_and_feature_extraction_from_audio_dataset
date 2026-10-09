# ---------------------------------------------------------------------------
# VocalSet Dataset Config
# ---------------------------------------------------------------------------

from scripts.config.dataset_config import DatasetConfig

DATA_INPUT_DIR = "/home/suvihaara/Documents/PhD/DATA/TUNI_emotion_dataset/wav"


def tuni_emotion_dataset_config(
    root_dir: str = DATA_INPUT_DIR,
    frame_duration: float = 3.0,
    sample_rate: int = 16000,
    overlap_percentage: float = 0.25,
) -> DatasetConfig:
    """
    TUNI dataset.
    All recording files are processed (all .wav).

    Label columns in output DataFrame
    ----------------------------------
    singer              : participant ID, e.g. 'FEM_1'
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
            "singer_id": [ 
                "FEM_1", 
                "FEM_2", 
                "FEM_3", 
                "FEM_4", 
                "FEM_5", 
                "FEM_6", 
                "FEM_7", 
                "FEM_8", 
                "FEM_9", 
                "MALE_1", 
                "MALE_2", 
                "FEM_10", 
                "FEM_11"
            ],
            "genre": [ "pop", "classical"],
            "emotion": ["joy", "sadness", "anger", "gentleness", "neutral"],
        },
        frame_duration=frame_duration,
        sample_rate=sample_rate,
        overlap_percentage=overlap_percentage,
        enable_silence_trimming=True,
    )

TUNI_dataset_config = tuni_emotion_dataset_config()
