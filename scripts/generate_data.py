"""Simulate 2 years of daily unit sales for 20 stores x 20 products (400 series).

Demand = base level x weekly pattern x yearly seasonality x trend x holiday lift x promo lift,
then drawn from a negative binomial (over-dispersed counts, like real retail).
Promotions are *planned*, so promo/price/holiday are known in advance (legitimate future features).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

rng = np.random.default_rng(C.SEED)
N_STORES, N_PRODUCTS, N_DAYS = 20, 20, 730
CATEGORIES = ["Grocery", "Beverages", "Household", "Toys"]
dates = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")
t = np.arange(N_DAYS)
doy = dates.dayofyear.values

HOLIDAYS = [(1, 1), (2, 14), (5, 27), (7, 4), (9, 2), (10, 31), (11, 28), (12, 24), (12, 25), (12, 31)]
is_hol = np.array([(d.month, d.day) in HOLIDAYS for d in dates])

cat_weekly = {c: 1 + rng.uniform(-0.35, 0.45, 7) for c in CATEGORIES}     # Mon..Sun multipliers
cat_amp = {c: rng.uniform(0.05, 0.30) for c in CATEGORIES}
cat_phase = {c: rng.uniform(0, 365) for c in CATEGORIES}
cat_hol = {"Grocery": 1.35, "Beverages": 1.25, "Household": 0.9, "Toys": 1.9}

rows = []
for s in range(1, N_STORES + 1):
    store_scale = rng.lognormal(0, 0.35)
    for p in range(1, N_PRODUCTS + 1):
        cat = CATEGORIES[(p - 1) % len(CATEGORIES)]
        base = rng.lognormal(np.log(7), 0.7) * store_scale
        slope = rng.uniform(-0.15, 0.30)
        base_price = float(np.round(rng.lognormal(2.5, 0.5), 2))

        promo = np.zeros(N_DAYS, dtype=int)
        lift = np.ones(N_DAYS)
        day = int(rng.integers(5, 40))
        while day < N_DAYS:
            length = int(rng.integers(3, 8)); strength = rng.uniform(1.3, 2.0)
            promo[day:day + length] = 1; lift[day:day + length] = strength
            day += length + int(rng.integers(20, 55))
        price = np.where(promo == 1, base_price * (1 - 0.35 * (lift - 1)), base_price).round(2)

        mu = (base * cat_weekly[cat][dates.dayofweek.values]
              * (1 + cat_amp[cat] * np.sin(2 * np.pi * (doy - cat_phase[cat]) / 365.25))
              * (1 + slope * t / N_DAYS) * np.where(is_hol, cat_hol[cat], 1.0) * lift)
        r = 8.0
        sales = rng.negative_binomial(r, r / (r + mu))
        rows.append(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "store_id": s, "product_id": p,
                                  "category": cat, "sales": sales, "promo": promo, "price": price,
                                  "base_price": base_price, "is_holiday": is_hol.astype(int)}))

df = pd.concat(rows, ignore_index=True)
C.DATA.mkdir(exist_ok=True)
df.to_csv(C.SALES_CSV, index=False)
print(f"series={N_STORES * N_PRODUCTS}  days={N_DAYS}  rows={len(df):,}  mean sales/day={df.sales.mean():.1f}  "
      f"promo share={df.promo.mean():.1%}  zero-sales share={(df.sales == 0).mean():.1%}")
