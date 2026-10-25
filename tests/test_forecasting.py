from datetime import date, timedelta

import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src import metrics as M
from src.backtest import make_folds
from src.features import FEATURES, build_features


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("tests")
         .config("spark.sql.shuffle.partitions", "2").config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


def _series(spark, sales):
    start = date(2024, 1, 1)
    rows = [(start + timedelta(days=i), 1, 1, "Grocery", float(v), 0, 10.0, 10.0, 0) for i, v in enumerate(sales)]
    return spark.createDataFrame(rows, ["date", "store_id", "product_id", "category", "sales", "promo",
                                        "price", "base_price", "is_holiday"])


def _feature_row(spark, sales, idx):
    d = date(2024, 1, 1) + timedelta(days=idx)
    return build_features(_series(spark, sales)).filter(F.col("date") == F.lit(d)).first()


def test_features_never_use_sales_newer_than_14_days_before_the_row(spark):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 50, 80).tolist()
    target = 60
    tampered = list(base)
    for i in range(target - 14 + 1, 80):          # rewrite EVERYTHING newer than target-14
        tampered[i] = 9999
    a, b = _feature_row(spark, base, target), _feature_row(spark, tampered, target)
    for f in FEATURES:
        assert a[f] == pytest.approx(b[f]), f"feature {f} leaked information from the future"
    assert a["y"] == pytest.approx(np.log1p(base[target]))      # the label itself is the true sales


def test_leak_test_is_sensitive_to_the_allowed_boundary(spark):
    base = list(range(1, 81))
    changed = list(base)
    changed[60 - 14] = 5000                           # day target-14 IS allowed to influence the row
    a, b = _feature_row(spark, base, 60), _feature_row(spark, changed, 60)
    assert a["l_lag_14"] != pytest.approx(b["l_lag_14"])


def test_lag_and_rolling_values(spark):
    sales = list(range(100, 180))                      # strictly increasing
    r = _feature_row(spark, sales, 60)
    assert r["l_lag_14"] == pytest.approx(np.log1p(sales[46]))
    assert r["l_lag_21"] == pytest.approx(np.log1p(sales[39]))
    assert r["l_roll_mean_7"] == pytest.approx(np.log1p(np.mean(sales[40:47])))     # days d-20..d-14
    assert r["l_roll_mean_28"] == pytest.approx(np.log1p(np.mean(sales[19:47])))    # days d-41..d-14


def test_warmup_rows_are_dropped(spark):
    feats = build_features(_series(spark, [10] * 60))
    first = feats.agg(F.min("date")).first()[0]
    assert first == date(2024, 1, 1) + timedelta(days=41)           # needs 42 days of history


def test_folds_are_contiguous_non_overlapping_and_end_at_max_date():
    folds = make_folds(date(2025, 12, 31), horizon=14, n_folds=3)
    assert folds[-1][2] == date(2025, 12, 31)
    for (c, s, e), (c2, s2, e2) in zip(folds, folds[1:]):
        assert e + timedelta(days=1) == s2 and c2 == e            # next fold trains on the previous test window
    assert all((e - c).days == 14 and s == c + timedelta(days=1) for c, s, e in folds)


def test_metrics():
    y, p = [10, 0, 10], [8, 2, 12]
    assert M.wape(y, p) == pytest.approx(6 / 20)
    assert M.bias([10, 10], [12, 12]) == pytest.approx(0.2)
    assert M.bias([10, 10], [8, 8]) == pytest.approx(-0.2)
    assert M.wape([0, 0], [0, 0]) == 0.0                          # zero-sales safe
    assert M.interval_coverage([1, 5, 9], [0, 0, 0], [2, 4, 10]) == pytest.approx(2 / 3)
