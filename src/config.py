"""Shared paths, column definitions and constants.

Every script imports from here so that the feature lists, the subject-ID
derivation and the random seed cannot drift between phases.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

SEED = 27

# --- UCI / MDVP ----------------------------------------------------------
UCI_RAW = RAW / "parkinsons.csv"
UCI_OUT = PROCESSED / "uci_mdvp.csv"

# The 22 acoustic measures. `name` is the recording ID, `status` the target;
# neither is a predictor.
UCI_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# --- Sakar ---------------------------------------------------------------
SAKAR_TRAIN_RAW = RAW / "sakar" / "train_data.txt"
SAKAR_TEST_RAW = RAW / "sakar" / "test_data.txt"
SAKAR_TRAIN_OUT = PROCESSED / "sakar_train.csv"
SAKAR_TEST_OUT = PROCESSED / "sakar_test.csv"

# The 26 Praat measures, in raw-file column order.
SAKAR_FEATURES = [
    "Jitter_local", "Jitter_local_absolute", "Jitter_rap", "Jitter_ppq5", "Jitter_ddp",
    "Shimmer_local", "Shimmer_local_dB", "Shimmer_apq3", "Shimmer_apq5",
    "Shimmer_apq11", "Shimmer_dda",
    "AC", "NTH", "HTN",
    "Median_pitch", "Mean_pitch", "Standard_deviation", "Minimum_pitch", "Maximum_pitch",
    "Number_of_pulses", "Number_of_periods", "Mean_period",
    "Standard_deviation_of_period",
    "Fraction_of_locally_unvoiced_frames", "Number_of_voice_breaks",
    "Degree_of_voice_breaks",
]

# Raw-file layouts. train_data.txt carries UPDRS, test_data.txt does not.
SAKAR_TRAIN_COLS = ["id"] + SAKAR_FEATURES + ["UPDRS", "class_information"]
SAKAR_TEST_COLS = ["id"] + SAKAR_FEATURES + ["class_information"]

# TRAP 3: UPDRS is a target leak (1 for every HC row, 5-55 for every PD row,
# zero overlap). It is dropped in build_datasets.py and never written to
# data/processed/, so it cannot reach a model by accident.
SAKAR_FORBIDDEN = ["UPDRS"]

# Recording-type blocks, by 1-based within-subject recording index.
# Verified empirically in run_recording_type.py, not assumed.
SAKAR_TYPE_BLOCKS = {
    "vowel": range(1, 4),      # 1-3   sustained vowels
    "number": range(4, 14),    # 4-13  numbers 1-10
    "sentence": range(14, 18), # 14-17 short sentences
    "word": range(18, 27),     # 18-26 words
}


def sakar_rec_type(k: int) -> str:
    for name, rng in SAKAR_TYPE_BLOCKS.items():
        if k in rng:
            return name
    raise ValueError(f"recording index {k} outside 1..26")
