# ---------------------------------------------------------------------------
# TUNI Emotion Dataset Config
# ---------------------------------------------------------------------------

from scripts.config.dataset_config import DatasetConfig

DATA_INPUT_DIR = (
    "/Users/pubert/Downloads/SkyNote/OTHER_DATASETS/TUNI_emotion_dataset/wav"
)


def tuni_dataset_config(
    root_dir: str = DATA_INPUT_DIR,
    frame_duration: float = 3.0,
    sample_rate: int = 16000,
    overlap_percentage: float = 0.25,
) -> DatasetConfig:
    """
    TUNI emotion recognition from singing dataset.

    Folder structure
    ----------------
        <root>/Bianca_Hösli/pop/Bianca_Hösli_1_neutraali_3a.wav
        Levels: singer / genre

    Label columns in output DataFrame
    ----------------------------------
    singer : participant name, e.g. 'Bianca_Hösli'
    genre  : 'classical' | 'pop'
    """
    return DatasetConfig(
        name="tuni_emotion_dataset",
        root_dir=root_dir,
        level_names=["singer", "genre"],
        participant_level=0,
        level_allowed_values={
            "genre": ["classical", "pop"],
        },
        frame_duration=frame_duration,
        sample_rate=sample_rate,
        overlap_percentage=overlap_percentage,
    )


tuni_dataset = tuni_dataset_config()
