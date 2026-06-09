# ---------------------------------------------------------------------------
# VocalSet Dataset Config
# ---------------------------------------------------------------------------

from scripts.config.dataset_config import DatasetConfig

DATA_INPUT_DIR = "/home/suvihaara/Documents/PhD/DATA/VocalSet/VocalSet_fixed/FULL"


def vocalset_dataset_config(
    root_dir: str = DATA_INPUT_DIR,
    frame_duration: float = 3.0,
    sample_rate: int = 16000,
    overlap_percentage: float = 0.25,
) -> DatasetConfig:
    """
    VocalSet dataset.

    Folder structures
    -----------------
    vocalset singers:
        <root>/female6/scales/fast_forte/f6_scales_c_fast_forte_a.wav
        Levels: singer_ID / exercise_type / vocal_technique / takes (all) .. _a.wav

    All recording files are processed (all .wav).

    Label columns in output DataFrame
    ----------------------------------
    singer              : participant ID, e.g. 'female6'
    vocal_technique     : e.g. 'breathy', 'belt', 'vibrato', etc.
    exercise_type       : vocal exercise type, e.g. 'scales', 'long_tones'
    """
    return DatasetConfig(
        name="vocalset_dataset",
        root_dir=root_dir,
        # Fallback schema + union of all column names across subgroups
        level_names=["singer", "exercise_type", "vocal_technique"],
        participant_level=0,
        level_allowed_values={
            "exercise_type": ["arpeggios", "excerpts", "long_tones", "scales"],
            "vocal_technique": [
                "belt",
                "breathy",
                "straight",
                "vibrato",
                "lip_trill",
                "trill",
                "trillo",
                "vocal_fry",
                "inhaled",
                "spoken",
                "fast_piano",
                "slow_piano",
                "fast_forte",
                "slow_forte",
                "forte",
                "messa",  # Messa di voce
                "pp",  # Pianissimo
            ],
        },
        frame_duration=frame_duration,
        sample_rate=sample_rate,
        overlap_percentage=overlap_percentage,
    )


vocalset_dataset = vocalset_dataset_config()
