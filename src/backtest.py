"""Rolling-origin backtesting: train on everything up to a cutoff, forecast the next 14 days, repeat."""
from datetime import date, timedelta
from typing import List, Tuple

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import config as C
from src.features import FEATURES

KEEP = ["fold", "date", "series_id", "store_id", "product_id", "category", "promo", "is_holiday",
        "sales", "horizon_day", "naive_14", "avg3_same_weekday", "ma28", "linear", "gbt", "gbt_log"]


def make_folds(max_date: date, horizon: int = C.HORIZON, n_folds: int = C.N_FOLDS) -> List[Tuple[date, date, date]]:
    """[(cutoff, test_start, test_end)] oldest first. Test windows are contiguous and never overlap."""
    folds = []
    for k in range(n_folds, 0, -1):
        test_end = max_date - timedelta(days=horizon * (k - 1))
        cutoff = test_end - timedelta(days=horizon)
        folds.append((cutoff, cutoff + timedelta(days=1), test_end))
    return folds


def add_baselines(df: DataFrame) -> DataFrame:
    """Honest benchmarks that need no model; any ML model must beat these to earn its keep."""
    return (df.withColumn("naive_14", F.col("lag_14"))                       # same weekday, 2 weeks ago
            .withColumn("avg3_same_weekday", (F.col("lag_14") + F.col("lag_21") + F.col("lag_28")) / 3)
            .withColumn("ma28", F.col("roll_mean_28")))                      # 28-day average level


def _gbt() -> Pipeline:
    return Pipeline(stages=[
        VectorAssembler(inputCols=FEATURES, outputCol="features"),
        GBTRegressor(featuresCol="features", labelCol="y", predictionCol="gbt_log", maxIter=40, maxDepth=5,
                     stepSize=0.15, subsamplingRate=0.8, seed=C.SEED)])


def _linear() -> Pipeline:
    return Pipeline(stages=[
        VectorAssembler(inputCols=FEATURES, outputCol="features_lin"),     # distinct name: both models transform the same frame
        LinearRegression(featuresCol="features_lin", labelCol="y", predictionCol="linear_log", regParam=1e-3)])


def run_backtest(feats: DataFrame, log=print) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feats = add_baselines(feats).cache()
    max_date = feats.agg(F.max("date")).first()[0]
    parts, importances = [], None
    for i, (cutoff, t0, t1) in enumerate(make_folds(max_date), start=1):
        train = feats.filter(F.col("date") <= F.lit(cutoff))
        test = feats.filter((F.col("date") >= F.lit(t0)) & (F.col("date") <= F.lit(t1)))
        log(f"fold {i}: train <= {cutoff} ({train.count():,} rows) | test {t0}..{t1} ({test.count():,} rows)")
        gbt_model = _gbt().fit(train)
        lin_model = _linear().fit(train)
        out = lin_model.transform(gbt_model.transform(test))
        out = (out.withColumn("gbt", F.greatest(F.expm1("gbt_log"), F.lit(0.0)))
               .withColumn("linear", F.greatest(F.expm1("linear_log"), F.lit(0.0)))
               .withColumn("fold", F.lit(i))
               .withColumn("horizon_day", F.datediff("date", F.lit(cutoff)))
               .select(*KEEP))
        parts.append(pd.DataFrame([r.asDict() for r in out.collect()]))   # small (5.6k rows/fold); avoids Arrow/pandas-3 coupling
        if i == len(make_folds(max_date)):
            imp = gbt_model.stages[-1].featureImportances.toArray()
            importances = pd.DataFrame({"feature": FEATURES, "importance": imp}).sort_values("importance", ascending=False)
    return pd.concat(parts, ignore_index=True), importances
