import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyspark.sql import functions as F  # noqa: E402

from src import config as C  # noqa: E402
from src.backtest import run_backtest  # noqa: E402
from src.features import build_features  # noqa: E402
from src.metrics import summarize  # noqa: E402
from src.spark_session import get_spark  # noqa: E402

METHODS = ["naive_14", "avg3_same_weekday", "ma28", "linear", "gbt"]
pd.set_option("display.width", 160)

spark = get_spark()
sales = (spark.read.csv(C.SALES_CSV, header=True)
         .select(F.to_date("date").alias("date"), F.col("store_id").cast("int"), F.col("product_id").cast("int"),
                 "category", F.col("sales").cast("double"), F.col("promo").cast("int"),
                 F.col("price").cast("double"), F.col("base_price").cast("double"),
                 F.col("is_holiday").cast("int")))
feats = build_features(sales).cache()
print(f"feature rows={feats.count():,}  features={len(__import__('src.features', fromlist=['x']).FEATURES)}")

preds, imp = run_backtest(feats)
C.OUT.mkdir(parents=True, exist_ok=True)
preds.to_csv(C.PREDICTIONS_CSV, index=False)
imp.to_csv(C.IMPORTANCE_CSV, index=False)
spark.stop()

print("\n=== OVERALL (all 3 folds, 42 test days x 400 series) ===")
print(summarize(preds, METHODS).round(4).to_string(index=False))
print("\n=== BY FOLD (WAPE) ===")
print(summarize(preds, METHODS, by="fold").pivot(index="method", columns="fold", values="WAPE").round(4))
print("\n=== PROMO vs NON-PROMO DAYS (WAPE) ===")
print(summarize(preds, METHODS, by="promo").pivot(index="method", columns="promo", values="WAPE").round(4))
print("\n=== TOP FEATURES (GBT) ===")
print(imp.head(8).round(3).to_string(index=False))
