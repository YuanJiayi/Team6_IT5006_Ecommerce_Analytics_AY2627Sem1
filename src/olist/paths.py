"""Repository locations used by the modelling code, scripts, tests and apps."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GEO_DIR = DATA_DIR / "geo"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
PHASE2_OUTPUTS = OUTPUTS_DIR / "phase2"
EXPERIMENT_OUTPUTS = OUTPUTS_DIR / "experiments"
MODELS_DIR = REPO_ROOT / "deployment" / "api" / "models"
PHASE2_REPORT_DIR = REPO_ROOT / "docs" / "phase2" / "report"
