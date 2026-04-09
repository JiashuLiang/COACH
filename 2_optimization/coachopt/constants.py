"""Constants shared across preprocessing, optimization, and analysis."""

from __future__ import annotations

HARTREE_TO_KCAL_MOL = 627.50947406
DEFAULT_A_ROWS = (64, 153, 166)
DEFAULT_GRID_KEY = "99590"
ANALYSIS_DIFF_GRIDS = ("99590", "75302")
DEFAULT_DIFF_MATRIX_NAME = f"diff_{DEFAULT_GRID_KEY}.npy"
DEFAULT_DIFF_NAMES_NAME = f"name_list_diff_{DEFAULT_GRID_KEY}.txt"
DEFAULT_SELECTED_DIFF_PREFIX = f"diff_constraint_{DEFAULT_GRID_KEY}"
DEFAULT_SELECTED_DIFF_NAME = f"{DEFAULT_SELECTED_DIFF_PREFIX}.npy"
DEFAULT_GRID_THRESHOLD = 0.015
DEFAULT_TOP_DIFF_PER_BETA = 100
DEFAULT_TOP_L1_ROWS = 200
DEFAULT_MAX_COEFFICIENT = 25.0
# CSV schemas maintained by the cleaned preprocessing pipeline.
REQUIRED_DATASET_EVAL_COLUMNS = ["Reaction", "Dataset", "Reference", "Stoichiometry"]
REQUIRED_TRAINING_WEIGHT_COLUMNS = ["Dataset", "datapoints", "weights"]
