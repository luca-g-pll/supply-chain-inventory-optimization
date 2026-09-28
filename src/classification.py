"""ABC (value) and XYZ (variability) classification of items.

* ABC: items sorted by revenue (descending, ties by item ID); an item is **A** if
  its cumulative revenue share (itself included) is <= 80% — the top item is
  always A — **B** if <= 95%, otherwise **C**.
* XYZ: coefficient of variation (sample std / mean) of the item's demand summed
  over all stores in complete 7-day buckets starting at the reference-period
  start: X < 0.5 <= Y < 1.0 <= Z.

Both use the reference period only (the validation period), so no information
from the test period leaks into the classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.series import natural_key


def _ranked(values: pd.Series) -> pd.Series:
    """Sort descending by value, ties by natural item order."""
    df = values.rename("v").reset_index()
    df.columns = ["item", "v"]
    df["_k"] = df["item"].map(natural_key)
    df = df.sort_values(["v", "_k"], ascending=[False, True], kind="mergesort")
    return df.set_index("item")["v"]


def abc_classify(revenue: pd.Series, thresholds: tuple[float, float] = (0.80, 0.95)) -> pd.DataFrame:
    """``revenue`` indexed by item -> ``[revenue, revenue_share, cum_share, abc]`` (ranked)."""
    ranked = _ranked(revenue.astype("float64"))
    total = ranked.sum()
    share = ranked / total if total > 0 else ranked * 0
    cum = share.cumsum()
    a, b = thresholds
    cls = np.where(cum <= a + 1e-12, "A", np.where(cum <= b + 1e-12, "B", "C"))
    cls[0] = "A"
    return pd.DataFrame({"revenue": ranked, "revenue_share": share, "cum_share": cum, "abc": cls})


def weekly_buckets(sales: pd.DataFrame, start) -> pd.DataFrame:
    """Item demand (all stores) per complete 7-day bucket from ``start``: ``[item, bucket, units]``.

    A bucket counts only if the item has all 7 days in it (partial first/last weeks
    and items that start mid-week are left out).
    """
    start = pd.Timestamp(start)
    df = sales.assign(bucket=(sales["date"] - start).dt.days // 7)
    g = df.groupby(["item", "bucket"], as_index=False).agg(units=("sales", "sum"), n_days=("date", "nunique"))
    return g[g["n_days"] == 7][["item", "bucket", "units"]].reset_index(drop=True)


def xyz_classify(weekly: pd.DataFrame, thresholds: tuple[float, float] = (0.5, 1.0)) -> pd.DataFrame:
    """``[item, bucket, units]`` -> ``[weekly_mean, weekly_std, weekly_cv, xyz]`` indexed by item."""
    g = weekly.groupby("item")["units"]
    out = pd.DataFrame({"weekly_mean": g.mean(), "weekly_std": g.std(ddof=1)})
    with np.errstate(divide="ignore", invalid="ignore"):
        out["weekly_cv"] = np.where(out["weekly_mean"] > 0, out["weekly_std"] / out["weekly_mean"], np.inf)
    x, y = thresholds
    out["xyz"] = np.where(out["weekly_cv"] < x, "X", np.where(out["weekly_cv"] < y, "Y", "Z"))
    return out


def classify_items(sales: pd.DataFrame, item_master: pd.DataFrame, ref_start, ref_end,
                   abc_thresholds=(0.80, 0.95), xyz_thresholds=(0.5, 1.0)) -> pd.DataFrame:
    """Full classification table, one row per item, on the reference period."""
    ref = sales[(sales["date"] >= pd.Timestamp(ref_start)) & (sales["date"] <= pd.Timestamp(ref_end))]
    price = item_master.set_index("item")["unit_price"]
    units = ref.groupby("item")["sales"].sum()
    items = pd.Index(sorted(set(sales["item"]), key=natural_key), name="item")
    units = units.reindex(items, fill_value=0.0)
    revenue = units * price.reindex(items).fillna(0.0)

    abc = abc_classify(revenue, abc_thresholds)
    abc_volume = abc_classify(units, abc_thresholds)["abc"].rename("abc_volume")
    xyz = xyz_classify(weekly_buckets(ref, ref_start), xyz_thresholds).reindex(items)
    xyz["xyz"] = xyz["xyz"].fillna("Z")

    out = abc.join(units.rename("units")).join(abc_volume).join(xyz)
    out.index.name = "item"
    out = out.reset_index()
    out["abc_xyz"] = out["abc"] + out["xyz"]
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["item", "rank", "units", "revenue", "revenue_share", "cum_share", "abc", "abc_volume",
                "weekly_mean", "weekly_std", "weekly_cv", "xyz", "abc_xyz"]]


def abc_xyz_matrix(classification: pd.DataFrame) -> pd.DataFrame:
    """3x3 table: item count and revenue share per ABC x XYZ cell."""
    g = classification.groupby(["abc", "xyz"])
    m = pd.DataFrame({"items": g.size(), "revenue_share": g["revenue_share"].sum()})
    full = pd.MultiIndex.from_product([list("ABC"), list("XYZ")], names=["abc", "xyz"])
    return m.reindex(full, fill_value=0).reset_index()
