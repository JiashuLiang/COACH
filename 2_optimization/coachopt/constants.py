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

# Baseline 289-parameter reference coefficients used as optional warm starts.
W_B97M_V_COEF_289 = [
    (0, 0.85),
    (8, 0.259),
    (1, 1.007),
    (96, 0.443),
    (104, -4.535),
    (112, -3.39),
    (131, 4.278),
    (100, -1.437),
    (192, 1.0),
    (200, 1.358),
    (208, 2.924),
    (240, -1.39),
    (209, -8.812),
    (241, 9.142),
    (288, 0.15),
]

W_B97X_V_COEF_289 = [
    (0, 0.833),
    (1, 0.603),
    (2, 1.194),
    (96, 0.556),
    (97, -0.257),
    (192, 1.219),
    (193, -1.850),
    (288, 0.167),
]
