from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SALES_CSV = str(DATA / "sales.csv")
OUT = DATA / "output"
PREDICTIONS_CSV = str(OUT / "backtest_predictions.csv")
IMPORTANCE_CSV = str(OUT / "feature_importance.csv")

HORIZON = 14      # forecast 14 days ahead; every feature uses data at least 14 days old
N_FOLDS = 3       # rolling-origin backtest folds
SEED = 42
