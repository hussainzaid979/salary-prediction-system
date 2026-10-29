"""
Configuration Module
====================
Defines project paths, column identifiers, model hyperparameters,
and ANSI terminal styling constants.
"""

from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
STATIC_DIR = BASE_DIR / "static"
STATIC_IMAGES_DIR = STATIC_DIR / "images"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, STATIC_DIR, STATIC_IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Paths
DEFAULT_DATA_PATH = DATA_DIR / "salary_data.csv"
MARKET_DATA_PATH = DATA_DIR / "salary_data_market.csv"

# Saved Model Path
DEFAULT_MODEL_PATH = MODELS_DIR / "salary_model.joblib"

# Visualization Output Path
FIGURE_PATH = FIGURES_DIR / "salary_prediction_analysis.png"
WEB_FIGURE_PATH = STATIC_IMAGES_DIR / "salary_prediction_analysis.png"

# Schema & Column Identifiers
EXPERIENCE_COLUMN = "Experience (Years)"
SALARY_COLUMN = "Salary (USD)"

# Model Hyperparameters
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# Experience Input Validation Limits
MIN_EXPERIENCE_YEARS = 0.0
MAX_EXPERIENCE_YEARS = 50.0

# Terminal UI Colors & Styling (ANSI Escape Sequences)
class UIColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"

    # Background colors
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"
