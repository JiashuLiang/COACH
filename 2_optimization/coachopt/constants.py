"""Constants shared across preprocessing, optimization, and analysis."""

from __future__ import annotations

HARTREE_TO_KCAL_MOL = 627.5095
DEFAULT_A_ROWS = (64, 153, 166)
DEFAULT_GRID_KEY = "99000590"
DEFAULT_GRID_THRESHOLD = 0.015
DEFAULT_TOP_DIFF_PER_BETA = 100
DEFAULT_TOP_L1_ROWS = 200
DEFAULT_MAX_COEFFICIENT = 25.0
# CSV schemas maintained by the cleaned preprocessing pipeline.
REQUIRED_DATASET_EVAL_COLUMNS = ["Reaction", "Dataset", "Reference", "Stoichiometry"]
REQUIRED_TRAINING_WEIGHT_COLUMNS = ["Dataset", "datapoints", "weights"]
