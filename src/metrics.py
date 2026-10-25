"""Forecast accuracy metrics (pandas / numpy, unit-tested)."""
import numpy as np
import pandas as pd


def wape(y, p) -> float:
    """Weighted absolute percentage error = sum|y-p| / sum y. Robust to zero-sales days (unlike MAPE)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9))


def bias(y, p) -> float:
    """Signed: >0 means over-forecasting."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float((p - y).sum() / max(y.sum(), 1e-9))


def mae(y, p) -> float:
    return float(np.abs(np.asarray(y, float) - np.asarray(p, float)).mean())


def summarize(df: pd.DataFrame, methods, by=None) -> pd.DataFrame:
    groups = [(None, df)] if by is None else df.groupby(by)
    rows = []
    for key, g in groups:
        for m in methods:
            rows.append({**({} if by is None else {by: key}), "method": m, "WAPE": wape(g.sales, g[m]),
                         "MAE": mae(g.sales, g[m]), "bias": bias(g.sales, g[m]), "n": len(g)})
    return pd.DataFrame(rows)


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    return float(((y >= lo) & (y <= hi)).mean())
