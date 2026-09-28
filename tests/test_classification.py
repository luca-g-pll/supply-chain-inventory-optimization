import numpy as np
import pandas as pd

from src.classification import abc_classify, abc_xyz_matrix, weekly_buckets, xyz_classify


def test_abc_thresholds_and_top_item_always_a():
    rev = pd.Series({"a": 70.0, "b": 15.0, "c": 10.0, "d": 5.0})
    out = abc_classify(rev)
    assert out["abc"].to_dict() == {"a": "A", "b": "B", "c": "B", "d": "C"}
    alone = abc_classify(pd.Series({"x": 90.0, "y": 10.0}))
    assert alone.loc["x", "abc"] == "A"


def test_ties_are_broken_in_natural_item_order():
    out = abc_classify(pd.Series({"10": 5.0, "2": 5.0, "1": 5.0}))
    assert list(out.index) == ["1", "2", "10"]


def test_xyz_uses_complete_weeks_only():
    dates = pd.date_range("2024-01-01", periods=17)
    sales = pd.DataFrame({"date": dates, "item": "A", "sales": np.r_[np.full(7, 10.0), np.full(7, 20.0), [99, 99, 99]]})
    w = weekly_buckets(sales, "2024-01-01")
    assert w["units"].tolist() == [70.0, 140.0]
    x = xyz_classify(w)
    assert x.loc["A", "weekly_cv"] == np.std([70, 140], ddof=1) / 105


def test_matrix_has_nine_cells():
    c = pd.DataFrame({"abc": ["A", "B"], "xyz": ["X", "Z"], "revenue_share": [0.8, 0.2]})
    m = abc_xyz_matrix(c)
    assert len(m) == 9 and m["items"].sum() == 2
