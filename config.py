"""
Central configuration for the AI-Based Intrusion Detection System for IoT Networks.
All other modules read settings from here (as config.X) so run_all.py can override them.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Folder that contains your dataset CSV files (searched recursively).
# Default: <project>/dataset/  (your MERGED_CSV folder lives inside it)
DATASET_DIR = Path(os.environ.get("IDS_DATASET_DIR", BASE_DIR / "dataset"))

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PROCESSED_DIR = RESULTS_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42

# ---- Data loading -------------------------------------------------------
# The full dataset has millions of rows. We randomly sample a fraction of each
# CSV so the whole pipeline finishes in minutes instead of hours.
SAMPLE_FRAC = 0.01        # 1% of every CSV file (1.0 = use everything)
MAX_ROWS = 300_000        # hard cap on rows kept after de-duplication (stratified)
TEST_SIZE = 0.20          # unseen test data (never used for training / feature ranking)
VAL_SIZE = 0.20           # part of TRAIN used to choose the number of features

# ---- Feature selection --------------------------------------------------
K_VALUES = [3, 5, 8, 10, 15, 20, 25, 30]   # candidate sizes for the reduced feature set
F1_TOLERANCE = 0.005      # pick smallest k whose validation F1 is within 0.5% of "all features"
FORCE_K = None            # set to e.g. 10 to force a specific number of features

# ---- Label handling -----------------------------------------------------
LABEL_CANDIDATES = ["label", "class", "attack_cat", "attack", "target", "category"]
NORMAL_NAMES = {"benign", "normal", "benigntraffic", "legitimate", "0", "0.0", "none"}


def ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
