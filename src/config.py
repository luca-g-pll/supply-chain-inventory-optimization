"""Load and validate ``config/params.yaml`` into frozen dataclasses.

Every business assumption of the project lives in the YAML file; this module is the
only place that reads it. Other modules receive the typed objects defined here.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "params.yaml"

KNOWN_MODELS = ("seasonal_naive", "moving_average", "holt_winters", "lightgbm")
KNOWN_POLICIES = ("P0_naive", "P1_static_sq", "P2_forecast_sq")
KNOWN_CHANNELS = ("physical", "online")


# --------------------------------------------------------------------------- blocks
@dataclass(frozen=True)
class PathsConfig:
    raw_train: Path
    duckdb: Path
    prices: Path
    work: Path
    artefacts: Path
    figures: Path
    sql: Path
    validation_report: Path


@dataclass(frozen=True)
class LimitsConfig:
    max_file_mb: float = 50.0
    max_rows: int = 2_000_000
    max_series: int = 1_000


@dataclass(frozen=True)
class InputConfig:
    source: Literal["demo", "file"] = "demo"
    path: Path | None = None
    sheet: str | int | None = None
    item_master: Path | None = None
    column_map: dict[str, str] = field(default_factory=dict)
    date_format: Literal["auto", "ISO", "dayfirst", "monthfirst"] = "auto"
    missing_days_as_zero: bool = True
    negative_quantity: Literal["clip", "reject"] = "clip"
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    min_history_days: int = 56


@dataclass(frozen=True)
class PeriodsConfig:
    test_days: int | None = None
    valid_days: int | None = None
    test_start: date | None = None
    test_end: date | None = None
    valid_start: date | None = None


@dataclass(frozen=True)
class PricesConfig:
    min: float = 5.0
    max: float = 50.0
    round_to: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class ClassificationConfig:
    abc_thresholds: tuple[float, float] = (0.80, 0.95)
    xyz_thresholds: tuple[float, float] = (0.5, 1.0)


@dataclass(frozen=True)
class HoltWintersConfig:
    seasonal_periods: int = 7
    refit_every: int = 13


@dataclass(frozen=True)
class LightGBMConfig:
    refit_every: int = 13
    lags: tuple[int, ...] = (14, 21, 28, 35, 364)
    rolling_windows: tuple[int, ...] = (7, 28, 91)
    params: dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 600


@dataclass(frozen=True)
class ForecastConfig:
    horizon: int = 14
    origin_step: int = 7
    baseline: str = "seasonal_naive"
    models: tuple[str, ...] = KNOWN_MODELS
    ma_window: int = 28
    holt_winters: HoltWintersConfig = field(default_factory=HoltWintersConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)


@dataclass(frozen=True)
class FastModeConfig:
    refit_every: int = 1000
    lightgbm_num_boost_round: int = 300
    holt_winters_max_series: int = 300


@dataclass(frozen=True)
class InventoryParams:
    lead_time_days: int = 7
    order_cost: float = 50.0
    holding_rate: float = 0.25
    cost_to_price_ratio: float = 0.60
    service_level: float = 0.95
    service_type: Literal["cycle", "fill_rate"] = "cycle"
    days_per_year: int = 365
    max_cover_days: int | None = 28    # cap the EOQ lot at this many days of demand (None = no cap)


@dataclass(frozen=True)
class ChannelParams:
    name: Literal["physical", "online"]
    inventory: InventoryParams
    backorder_share: float = 0.0
    backorder_cost: float = 0.0


@dataclass(frozen=True)
class ScenarioConfig:
    n_stores: int | None = None
    n_items: int | None = None
    item_selection: Literal["top_revenue", "random", "manual"] = "top_revenue"
    random_seed: int = 42
    manual_items: tuple[str, ...] = ()
    online_stores: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimulationConfig:
    policies: tuple[str, ...] = KNOWN_POLICIES
    channels: tuple[str, ...] = KNOWN_CHANNELS
    naive_order_weekday: int = 0
    calibrate_to_fill_rate: bool = True
    target_fill_rate: float = 0.95
    calibration_tol: float = 0.002
    detail_stores: tuple[str, ...] = ("1",)


@dataclass(frozen=True)
class SensitivityConfig:
    lead_times: tuple[int, ...] = (3, 7, 14)
    service_levels: tuple[float, ...] = (0.90, 0.95, 0.98, 0.99)


@dataclass(frozen=True)
class Config:
    root: Path
    paths: PathsConfig
    input: InputConfig
    periods: PeriodsConfig
    prices: PricesConfig
    classification: ClassificationConfig
    forecast: ForecastConfig
    fast_mode: FastModeConfig
    inventory: InventoryParams
    channels: dict[str, ChannelParams]
    scenario: ScenarioConfig
    simulation: SimulationConfig
    sensitivity: SensitivityConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def replace(self, **changes: Any) -> Config:
        """Return a copy with some top-level blocks replaced."""
        return dataclasses.replace(self, **changes)

    def channel(self, name: str) -> ChannelParams:
        try:
            return self.channels[name]
        except KeyError as exc:
            raise ValueError(f"Unknown channel {name!r}; expected one of {sorted(self.channels)}") from exc


# --------------------------------------------------------------------------- loading
def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Read the YAML file, apply optional nested ``overrides`` and validate.

    Raises ``ValueError`` if a value is out of range and ``FileNotFoundError`` if
    the file does not exist.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = deep_merge(raw, overrides or {})
    root = cfg_path.resolve().parent.parent
    return _build(raw, root)


def _path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def _as_date(value: Any, key: str) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{key}: expected a date YYYY-MM-DD, got {value!r}") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _inventory(raw: dict[str, Any], base: InventoryParams | None = None) -> InventoryParams:
    base = base or InventoryParams()
    inv = InventoryParams(
        lead_time_days=int(raw.get("lead_time_days", base.lead_time_days)),
        order_cost=float(raw.get("order_cost", base.order_cost)),
        holding_rate=float(raw.get("holding_rate", base.holding_rate)),
        cost_to_price_ratio=float(raw.get("cost_to_price_ratio", base.cost_to_price_ratio)),
        service_level=float(raw.get("service_level", base.service_level)),
        service_type=raw.get("service_type", base.service_type),
        days_per_year=int(raw.get("days_per_year", base.days_per_year)),
        max_cover_days=(None if raw.get("max_cover_days", base.max_cover_days) is None
                        else int(raw.get("max_cover_days", base.max_cover_days))),
    )
    validate_inventory(inv)
    return inv


def validate_inventory(inv: InventoryParams) -> None:
    """Range checks shared by the config loader and the dashboard what-if."""
    _check(inv.lead_time_days >= 1, "inventory.lead_time_days must be >= 1")
    _check(inv.order_cost > 0, "inventory.order_cost must be > 0")
    _check(0 < inv.holding_rate <= 1, "inventory.holding_rate must be in (0, 1]")
    _check(0 < inv.cost_to_price_ratio <= 1, "inventory.cost_to_price_ratio must be in (0, 1]")
    _check(0.5 < inv.service_level < 1, "inventory.service_level must be in (0.5, 1)")
    _check(inv.service_type in ("cycle", "fill_rate"), "inventory.service_type must be cycle or fill_rate")
    _check(inv.days_per_year > 0, "inventory.days_per_year must be > 0")
    _check(inv.max_cover_days is None or inv.max_cover_days >= 1, "inventory.max_cover_days must be >= 1 or null")


def _thresholds(values: Any, key: str) -> tuple[float, float]:
    t = tuple(float(v) for v in values)
    _check(len(t) == 2 and 0 < t[0] < t[1], f"{key}: expected two increasing positive numbers")
    return t  # type: ignore[return-value]


def _build(raw: dict[str, Any], root: Path) -> Config:
    p = raw.get("paths", {})
    paths = PathsConfig(
        raw_train=_path(root, p.get("raw_train", "data/raw/train.csv")),
        duckdb=_path(root, p.get("duckdb", "data/processed/inventory.duckdb")),
        prices=_path(root, p.get("prices", "data/reference/item_prices.csv")),
        work=_path(root, p.get("work", "data/processed/work")),
        artefacts=_path(root, p.get("artefacts", "data/processed/artefacts")),
        figures=_path(root, p.get("figures", "reports/figures")),
        sql=_path(root, p.get("sql", "sql/kpi_queries.sql")),
        validation_report=_path(root, p.get("validation_report", "reports/validation.md")),
    )

    i = raw.get("input", {})
    limits = LimitsConfig(**{k: v for k, v in (i.get("limits") or {}).items()})
    inp = InputConfig(
        source=i.get("source", "demo"),
        path=_path(root, i.get("path")),
        sheet=i.get("sheet"),
        item_master=_path(root, i.get("item_master")),
        column_map={str(k): str(v) for k, v in (i.get("column_map") or {}).items()},
        date_format=i.get("date_format", "auto"),
        missing_days_as_zero=bool(i.get("missing_days_as_zero", True)),
        negative_quantity=i.get("negative_quantity", "clip"),
        limits=limits,
        min_history_days=int(i.get("min_history_days", 56)),
    )
    _check(inp.source in ("demo", "file"), "input.source must be demo or file")
    _check(inp.source == "demo" or inp.path is not None, "input.path is required when input.source = file")
    _check(inp.date_format in ("auto", "ISO", "dayfirst", "monthfirst"), "input.date_format is invalid")
    _check(inp.negative_quantity in ("clip", "reject"), "input.negative_quantity must be clip or reject")
    _check(inp.min_history_days >= 14, "input.min_history_days must be >= 14")

    pr = raw.get("periods", {})
    periods = PeriodsConfig(
        test_days=None if pr.get("test_days") in (None, "auto") else int(pr["test_days"]),
        valid_days=None if pr.get("valid_days") in (None, "auto") else int(pr["valid_days"]),
        test_start=_as_date(pr.get("test_start"), "periods.test_start"),
        test_end=_as_date(pr.get("test_end"), "periods.test_end"),
        valid_start=_as_date(pr.get("valid_start"), "periods.valid_start"),
    )
    _check(periods.test_days is None or periods.test_days >= 14, "periods.test_days must be >= 14")
    _check(periods.valid_days is None or periods.valid_days >= 14, "periods.valid_days must be >= 14")

    pc = raw.get("prices", {})
    prices = PricesConfig(
        min=float(pc.get("min", 5.0)),
        max=float(pc.get("max", 50.0)),
        round_to=float(pc.get("round_to", 0.05)),
        seed=int(pc.get("seed", 42)),
    )
    _check(0 < prices.min < prices.max, "prices: need 0 < min < max")

    c = raw.get("classification", {})
    classification = ClassificationConfig(
        abc_thresholds=_thresholds(c.get("abc_thresholds", (0.8, 0.95)), "classification.abc_thresholds"),
        xyz_thresholds=_thresholds(c.get("xyz_thresholds", (0.5, 1.0)), "classification.xyz_thresholds"),
    )
    _check(classification.abc_thresholds[1] < 1, "classification.abc_thresholds must be < 1")

    f = raw.get("forecast", {})
    hw = f.get("holt_winters", {})
    lg = f.get("lightgbm", {})
    forecast = ForecastConfig(
        horizon=int(f.get("horizon", 14)),
        origin_step=int(f.get("origin_step", 7)),
        baseline=f.get("baseline", "seasonal_naive"),
        models=tuple(f.get("models", KNOWN_MODELS)),
        ma_window=int(f.get("ma_window", 28)),
        holt_winters=HoltWintersConfig(
            seasonal_periods=int(hw.get("seasonal_periods", 7)),
            refit_every=int(hw.get("refit_every", 13)),
        ),
        lightgbm=LightGBMConfig(
            refit_every=int(lg.get("refit_every", 13)),
            lags=tuple(int(x) for x in lg.get("lags", (14, 21, 28, 35, 364))),
            rolling_windows=tuple(int(x) for x in lg.get("rolling_windows", (7, 28, 91))),
            params=dict(lg.get("params") or {}),
            num_boost_round=int(lg.get("num_boost_round", 600)),
        ),
    )
    _check(forecast.horizon >= 1, "forecast.horizon must be >= 1")
    _check(forecast.origin_step >= 1, "forecast.origin_step must be >= 1")
    _check(set(forecast.models) <= set(KNOWN_MODELS), f"forecast.models must be a subset of {KNOWN_MODELS}")
    _check(forecast.baseline in forecast.models, "forecast.baseline must be one of forecast.models")
    _check(forecast.origin_step <= forecast.horizon, "forecast.origin_step must be <= forecast.horizon")
    _check(min(forecast.lightgbm.lags) >= forecast.horizon,
           "forecast.lightgbm.lags must all be >= forecast.horizon (leakage guard)")

    fm = raw.get("fast_mode", {})
    fast_mode = FastModeConfig(
        refit_every=int(fm.get("refit_every", 1000)),
        lightgbm_num_boost_round=int(fm.get("lightgbm_num_boost_round", 300)),
        holt_winters_max_series=int(fm.get("holt_winters_max_series", 300)),
    )

    inventory = _inventory(raw.get("inventory", {}))

    channels: dict[str, ChannelParams] = {}
    for name, ch_raw in (raw.get("channels") or {}).items():
        _check(name in KNOWN_CHANNELS, f"Unknown channel {name!r}")
        ch_raw = dict(ch_raw or {})
        share = float(ch_raw.pop("backorder_share", 0.0))
        bo_cost = float(ch_raw.pop("backorder_cost", 0.0))
        _check(0 <= share <= 1, f"channels.{name}.backorder_share must be in [0, 1]")
        _check(bo_cost >= 0, f"channels.{name}.backorder_cost must be >= 0")
        channels[name] = ChannelParams(name, _inventory(ch_raw, inventory), share, bo_cost)
    for name in KNOWN_CHANNELS:
        channels.setdefault(name, ChannelParams(name, inventory))

    s = raw.get("scenario", {})
    scenario = ScenarioConfig(
        n_stores=None if s.get("n_stores") is None else int(s["n_stores"]),
        n_items=None if s.get("n_items") is None else int(s["n_items"]),
        item_selection=s.get("item_selection", "top_revenue"),
        random_seed=int(s.get("random_seed", 42)),
        manual_items=tuple(str(x) for x in (s.get("manual_items") or ())),
        online_stores=tuple(str(x) for x in (s.get("online_stores") or ())),
    )
    _check(scenario.item_selection in ("top_revenue", "random", "manual"), "scenario.item_selection is invalid")

    sm = raw.get("simulation", {})
    simulation = SimulationConfig(
        policies=tuple(sm.get("policies", KNOWN_POLICIES)),
        channels=tuple(sm.get("channels", KNOWN_CHANNELS)),
        naive_order_weekday=int(sm.get("naive_order_weekday", 0)),
        calibrate_to_fill_rate=bool(sm.get("calibrate_to_fill_rate", True)),
        target_fill_rate=float(sm.get("target_fill_rate", 0.95)),
        calibration_tol=float(sm.get("calibration_tol", 0.002)),
        detail_stores=tuple(str(x) for x in (sm.get("detail_stores") or ())),
    )
    _check(set(simulation.policies) <= set(KNOWN_POLICIES), f"simulation.policies must be a subset of {KNOWN_POLICIES}")
    _check(set(simulation.channels) <= set(KNOWN_CHANNELS), f"simulation.channels must be a subset of {KNOWN_CHANNELS}")
    _check(0 <= simulation.naive_order_weekday <= 6, "simulation.naive_order_weekday must be 0..6")
    _check(0 < simulation.target_fill_rate < 1, "simulation.target_fill_rate must be in (0, 1)")

    se = raw.get("sensitivity", {})
    sensitivity = SensitivityConfig(
        lead_times=tuple(int(x) for x in se.get("lead_times", (3, 7, 14))),
        service_levels=tuple(float(x) for x in se.get("service_levels", (0.9, 0.95, 0.98, 0.99))),
    )
    _check(all(0.5 < x < 1 for x in sensitivity.service_levels), "sensitivity.service_levels must be in (0.5, 1)")
    _check(all(x >= 1 for x in sensitivity.lead_times), "sensitivity.lead_times must be >= 1")

    return Config(
        root=root,
        paths=paths,
        input=inp,
        periods=periods,
        prices=prices,
        classification=classification,
        forecast=forecast,
        fast_mode=fast_mode,
        inventory=inventory,
        channels=channels,
        scenario=scenario,
        simulation=simulation,
        sensitivity=sensitivity,
        raw=raw,
    )
