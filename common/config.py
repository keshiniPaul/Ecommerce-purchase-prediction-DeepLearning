"""
Shared configuration for the SE4050 e-commerce purchase
prediction project.

All models should use these common settings.
"""

# Reproducibility
SEED = 42

# Sessionization
SESSION_GAP_SECONDS = 30 * 60

# Chronological dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Events allowed as model input.
# Transaction must NEVER be model input.
ALLOWED_INPUT_EVENTS = {"view", "addtocart"}

# Project paths
RAW_EVENTS_PATH = "data/raw/events.csv"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"