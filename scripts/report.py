"""Charts + prediction intervals from the saved backtest predictions.

Intervals are *empirical*: for fold k we take the 10th/90th percentiles of log-ratio residuals from
fold k-1 (which is strictly in the past) and apply them to fold k's forecasts. Coverage is then
measured honestly on data the calibration never saw. Fold 1 has no earlier fold, so it is skipped.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.metrics import interval_coverage, summarize  # noqa: E402

p = pd.read_csv(C.PREDICTIONS_CSV, parse_dates=["date"])
METHODS = ["naive_14", "avg3_same_weekday", "ma28", "linear", "gbt"]
LABELS = {"naive_14": "Naive\n(same day -14)", "avg3_same_weekday": "Avg of last 3\nsame weekdays",
          "ma28": "28-day\nmoving avg", "linear": "Linear\n(log scale)", "gbt": "Gradient\nboosting"}
COLORS = ["#b0b7c3", "#8895a7", "#6b7a8f", "#4c8bf5", "#1a56c9"]

# ---------- prediction intervals (calibrated on the previous fold) ----------
p["resid_log"] = np.log1p(p.sales) - np.log1p(p.gbt)
lo_q, hi_q = {}, {}
for k in sorted(p.fold.unique()):
    prev = p[p.fold == k - 1]
    if len(prev):
        lo_q[k], hi_q[k] = prev.resid_log.quantile(0.10), prev.resid_log.quantile(0.90)
p["lo"] = np.nan; p["hi"] = np.nan
for k in lo_q:
    m = p.fold == k
    p.loc[m, "lo"] = np.expm1(np.log1p(p.loc[m, "gbt"]) + lo_q[k]).clip(lower=0)
    p.loc[m, "hi"] = np.expm1(np.log1p(p.loc[m, "gbt"]) + hi_q[k])
cal = p.dropna(subset=["lo"])
coverage = interval_coverage(cal.sales, cal.lo, cal.hi)
print(f"80% interval empirical coverage on folds {sorted(lo_q)}: {coverage:.1%}  (target 80%)")

# ---------- bias correction (log-trained models forecast closer to the median than the mean) ----------
# scale for fold k = (sum actual / sum forecast) on fold k-1, i.e. strictly past data. Fold 1 has none.
p["gbt_adj"] = np.nan
for k in sorted(p.fold.unique())[1:]:
    prev = p[p.fold == k - 1]
    p.loc[p.fold == k, "gbt_adj"] = p.loc[p.fold == k, "gbt"] * prev.sales.sum() / prev.gbt.sum()
sub = p[p.fold >= 2]
print("\n=== BIAS CORRECTION (folds 2-3 only, calibrated on the previous fold) ===")
print(summarize(sub, ["naive_14", "ma28", "gbt", "gbt_adj"]).round(4).to_string(index=False))
tot = sub.groupby("fold").agg(actual=("sales", "sum"), gbt=("gbt", "sum"), gbt_adj=("gbt_adj", "sum")).round(0)
print("\ntotal units per fold:\n", tot.to_string())

# ---------- figure 1: accuracy comparison ----------
overall = summarize(p, METHODS).set_index("method")
promo = summarize(p, METHODS, by="promo").pivot(index="method", columns="promo", values="WAPE")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].bar([LABELS[m] for m in METHODS], overall.loc[METHODS, "WAPE"], color=COLORS)
ax[0].set_title("WAPE overall (lower is better)")
for i, m in enumerate(METHODS):
    ax[0].text(i, overall.loc[m, "WAPE"], f"{overall.loc[m, 'WAPE']:.3f}", ha="center", va="bottom", fontsize=9)
x = np.arange(len(METHODS)); w = 0.38
ax[1].bar(x - w / 2, promo.loc[METHODS, 0], w, label="normal days", color="#4c8bf5")
ax[1].bar(x + w / 2, promo.loc[METHODS, 1], w, label="promo days", color="#d9534f")
ax[1].set_xticks(x); ax[1].set_xticklabels([LABELS[m] for m in METHODS]); ax[1].legend(); ax[1].set_title("WAPE: promo vs normal days")
by_h = p.groupby("horizon_day").apply(lambda g: pd.Series({m: np.abs(g.sales - g[m]).sum() / g.sales.sum()
                                                          for m in ["naive_14", "gbt"]}), include_groups=False)
ax[2].plot(by_h.index, by_h.naive_14, "o-", color="#8895a7", label="Naive"); ax[2].plot(by_h.index, by_h.gbt, "o-", color="#1a56c9", label="GBT")
ax[2].set_xlabel("days ahead of forecast origin"); ax[2].set_title("WAPE by forecast horizon"); ax[2].legend()
for a in ax[:2]: a.tick_params(axis="x", labelsize=7.5)
plt.tight_layout(); plt.savefig(C.ROOT / "reports" / "accuracy.png", dpi=130); plt.close()

# ---------- figure 2: example forecasts with intervals (last fold) ----------
last = cal[cal.fold == cal.fold.max()]
picks = last.groupby("series_id").sales.sum().sort_values().index
chosen = [picks[len(picks) // 5], picks[len(picks) // 2], picks[-len(picks) // 10]]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for a, sid in zip(axes, chosen):
    d = last[last.series_id == sid].sort_values("date")
    a.fill_between(d.date, d.lo, d.hi, color="#4c8bf5", alpha=0.2, label="80% interval")
    a.plot(d.date, d.gbt, color="#1a56c9", label="forecast"); a.plot(d.date, d.sales, "k.-", lw=0.8, label="actual")
    a.scatter(d.date[d.promo == 1], d.sales[d.promo == 1], color="#d9534f", zorder=3, s=30, label="promo day")
    a.set_title(f"series {sid} ({d.category.iloc[0]})", fontsize=10); a.tick_params(axis="x", rotation=45, labelsize=7)
axes[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(C.ROOT / "reports" / "forecasts.png", dpi=130)
print("charts saved")
