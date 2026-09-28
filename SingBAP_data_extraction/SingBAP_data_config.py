# ---------------------------------------------------------------------------
# VTE Dataset Config
# ---------------------------------------------------------------------------

from scripts.config.dataset_config import DatasetConfig

DATA_INPUT_DIR = "/home/suvihaara/Documents/PhD/DATA/VTE/VOICE_DATA_CLEAN/"


def vte_dataset_config(
    root_dir: str = DATA_INPUT_DIR,
    frame_duration: float = 3.0,
    sample_rate: int = 16000,
    overlap_percentage: float = 0.25,
) -> DatasetConfig:
    """
    VTE voice dataset.

    Folder structures
    -----------------
    professional / intermediate  (have phonation level):
        <root>/professional/PROF-1/undefined/arched_back/glissando/1/prof-1-...-mic-audio.wav
        <root>/intermediate/INTR-3/breathy/arched_back/glissando/1/intr-3-...-mic-audio.wav
        Levels: experience / singer / phonation / condition / scale / take

    inexperienced  (no phonation level):
        <root>/inexperienced/INEX-4/after_instruction/glissando/2/inex-4-...-mic-audio.wav
        Levels: experience / singer / condition / scale / take

    Only mic recording files are processed (suffix filter: '-mic-audio').
    The 'phonation' column will be None for all inexperienced rows.

    Label columns in output DataFrame
    ----------------------------------
    experience  : 'professional' | 'intermediate' | 'inexperienced'
    singer     : participant ID, e.g. 'PROF-1', 'INTR-3', 'INEX-4'
    phonation   : 'undefined' | 'breathy'  (None for inexperienced)
    condition   : posture/articulation condition (experienced) or
                  'before_instruction' / 'after_instruction' (inexperienced)
    scale       : vocal exercise type, e.g. 'glissando'
    take        : recording take number
    """
    return DatasetConfig(
        name="SingBAP_vte_dataset",
        root_dir=root_dir,
        # Fallback schema + union of all column names across subgroups
        level_names=["experience", "singer", "phonation", "condition", "scale", "take"],
        participant_level=1,
        subgroup_split_level=0,
        subgroup_configs={
            "professional": [
                "experience",
                "singer",
                "phonation",
                "condition",
                "scale",
                "take",
            ],
            "intermediate": [
                "experience",
                "singer",
                "phonation",
                "condition",
                "scale",
                "take",
            ],
            "inexperienced": ["experience", "singer", "condition", "scale", "take"],
        },
        level_allowed_values={
            "experience": ["professional", "intermediate", "inexperienced"],
            "phonation": ["undefined", "breathy"],
            "condition": [
                # experienced/intermediate conditions
                "arched_back",
                "hunched_back",
                "sideways",
                "correct",
                "over_articulation",
                "under_articulation",
                "chest_breathing",
                # inexperienced conditions
                "before_instruction",
                "after_instruction",
            ],
            "scale": ["simple", "vowel", "sustained", "glissando"],
        },
        file_suffix_filter="-mic-audio",
        frame_duration=frame_duration,
        sample_rate=sample_rate,
        overlap_percentage=overlap_percentage,
    )
