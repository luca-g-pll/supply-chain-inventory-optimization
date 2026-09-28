import numpy as np
import pandas as pd
import pytest

from src.metrics import bias, error_sums, evaluate, improvement, mape, metrics_from_sums, rmse, wape


def test_wape_hand_computed():  # T-M1
    assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)


def test_bias_zero_when_errors_cancel():  # T-M2
    assert bias([10, 20], [12, 18]) == pytest.approx(0.0)


def test_mape_ignores_zero_actuals():  # T-M3
    assert mape([0, 10], [5, 12]) == pytest.approx(0.2)
    assert np.isnan(mape([0, 0], [1, 1]))


def test_perfect_forecast_is_zero_everywhere():  # T-M4
    a = np.array([3.0, 5.0, 7.0])
    assert wape(a, a) == 0 and mape(a, a) == 0 and bias(a, a) == 0 and rmse(a, a) == 0


def test_nan_pairs_are_ignored():
    assert wape([10, np.nan, 20], [12, 5, 18]) == pytest.approx(4 / 30)


def test_sums_give_same_metrics_as_direct_formulas():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 50, 200).astype(float)
    f = a + rng.normal(0, 5, 200)
    s = error_sums(a, f)
    m = metrics_from_sums(pd.DataFrame({k: [v] for k, v in s.items()})).iloc[0]
    assert m["wape"] == pytest.approx(wape(a, f))
    assert m["mape"] == pytest.approx(mape(a, f))
    assert m["bias"] == pytest.approx(bias(a, f))
    assert m["rmse"] == pytest.approx(rmse(a, f))


def test_evaluate_and_improvement():
    errors = pd.DataFrame({"model": ["base", "base", "good", "good"], "abs_err": [10, 10, 5, 5],
                           "err": [1, -1, 0, 0], "actual": [100, 100, 100, 100], "sq_err": [100] * 4,
                           "n": [10] * 4, "ape_sum": [1] * 4, "n_pos": [10] * 4})
    lb = evaluate(errors, ["model"])
    imp = improvement(lb, baseline="base")
    assert imp["good"] == pytest.approx(0.5)
    assert imp["base"] == pytest.approx(0.0)
