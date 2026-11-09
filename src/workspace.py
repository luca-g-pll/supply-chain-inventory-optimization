"""Company folder: the files the company drops in, its settings and the last analysis, kept on its own computer.

The hosted demo keeps everything in the browser session. Installed in a company (started with
the launcher, which sets ``PLANNER_WORKSPACE``), the software works on a folder instead::

    Demand Planner/
        data/            sales, purchases, item list, stock, supplier terms, promotions - any supported format
        results/         the last analysis (reopened at the next start) and the files written by the command line
        settings.yaml    the company's costs and options (they replace config/params.yaml)

Every file in ``data/`` is recognised by its columns (and, in doubt, by its name): nothing has
to be renamed or converted. Several sales files (one per year, per store, per export) are joined.

Command line (for a scheduled refresh): ``python -m src.workspace --analyse``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.config import Config, load_config
from src.constraints import _TERM_COLUMNS, read_supplier_terms
from src.ingest import (
    FIELDS,
    SYNONYMS,
    ColumnMapping,
    StandardisedData,
    guess_mapping,
    merge_item_master,
    normalise_name,
    parse_dates,
    parse_numbers,
    read_any,
    standardise,
)
from src.promotions import _CALENDAR, read_promo_calendar
from src.purchases import PURCHASE_COLUMNS, PurchaseSummary, apply_purchases, purchase_columns, read_purchases
from src.readers import SUPPORTED, InputError
from src.replenishment import STOCK_SYNONYMS, read_stock_file

ENV_VAR = "PLANNER_WORKSPACE"
DATA, RESULTS, SETTINGS = "data", "results", "settings.yaml"
ROLES = ("sales", "purchases", "item_master", "stock", "supplier_terms", "promo_calendar")
NUMERIC = ("quantity", "unit_price", "unit_cost", "lead_time_days", "pack_size", "min_order_qty")
SETTING_KEYS = {"inventory": ("lead_time_days", "order_cost", "holding_rate", "cost_to_price_ratio"),
                "simulation": ("target_fill_rate",),
                "input": ("date_format", "negative_quantity", "granularity")}
# words in a file name that settle the role when the columns alone fit more than one
NAME_HINTS = {
    "purchases": ("acquist", "purchas", "ordini_fornitor", "ordinifornitor", "carichi", "ddt", "bolle", "fatture_passive",
                  "supplier_order", "po_"),
    "sales": ("vendit", "sales", "scontrin", "venduto", "ordini_clienti", "order_lines", "corrispettiv", "sell"),
    "stock": ("giacenz", "stock", "inventario", "inventory", "rimanenz"),
    "item_master": ("anagraf", "articoli", "items", "listino", "prodotti", "products", "master", "catalog"),
    "supplier_terms": ("fornitori", "supplier_terms", "suppliers", "condizioni"),
    "promo_calendar": ("promo", "volantin", "offerte"),
}
README = """Demand & Inventory Planner - company folder / cartella aziendale

ITALIANO
1. Metti i tuoi file nella cartella "data": vendite, acquisti, anagrafica articoli, giacenze,
   condizioni dei fornitori, calendario promozioni. Vanno bene Excel (.xlsx, .xls), CSV, testo,
   OpenDocument (.ods), JSON, Markdown, HTML, XML e Parquet. Non serve rinominare le colonne.
2. Apri il software e premi "Analizza i dati della cartella" nella pagina Dati.
3. L'ultima analisi resta salvata in "results" e si riapre da sola al prossimo avvio.
Le tue impostazioni (costi, tempi di consegna) sono in "settings.yaml".
Niente lascia questo computer.

ENGLISH
1. Put your files in the "data" folder: sales, purchases, item list, stock, supplier terms,
   promotion calendar. Excel (.xlsx, .xls), CSV, text, OpenDocument (.ods), JSON, Markdown, HTML,
   XML and Parquet all work. Columns do not need to be renamed.
2. Open the software and press "Analyse the folder" on the Data page.
3. The last analysis stays in "results" and reopens by itself at the next start.
Your settings (costs, lead times) are in "settings.yaml".
Nothing leaves this computer.
"""


# --------------------------------------------------------------------------- folder
def workspace_dir() -> Path | None:
    """The company folder, or ``None`` when the software runs as the hosted demo (no folder on disk)."""
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value).expanduser() if value else None


def default_dir(cfg: Config | None = None) -> Path:
    name = (cfg or load_config()).workspace.folder_name
    documents = Path.home() / "Documents"
    return (documents if documents.is_dir() else Path.home()) / name


def ensure(folder: Path) -> Path:
    """Create the folder structure (idempotent) and return the folder."""
    folder = Path(folder)
    (folder / DATA).mkdir(parents=True, exist_ok=True)
    (folder / RESULTS).mkdir(exist_ok=True)
    readme = folder / "LEGGIMI - README.txt"
    if not readme.exists():
        readme.write_text(README, encoding="utf-8")
    if not (folder / SETTINGS).exists():
        save_settings(folder, {})
    return folder


# --------------------------------------------------------------------------- settings
def load_settings(folder: Path) -> dict:
    """``{"company": {...}, "overrides": {...}}``; ``overrides`` goes to ``load_config``."""
    path = Path(folder) / SETTINGS
    raw = {}
    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise InputError(f"Could not read {SETTINGS}: {exc}") from exc
    overrides = {block: {k: raw[block][k] for k in keys if k in (raw.get(block) or {})}
                 for block, keys in SETTING_KEYS.items()}
    return {"company": dict(raw.get("company") or {}), "overrides": {b: v for b, v in overrides.items() if v}}


def save_settings(folder: Path, overrides: dict, company: dict | None = None) -> None:
    clean = {block: {k: overrides[block][k] for k in keys if k in (overrides.get(block) or {})}
             for block, keys in SETTING_KEYS.items()}
    body = {"company": company or {"name": ""}, **{b: v for b, v in clean.items() if v}}
    header = ("# Company settings. They replace the defaults of the software (config/params.yaml).\n"
              "# Impostazioni dell'azienda: sostituiscono i valori predefiniti del software.\n"
              "# inventory: lead_time_days, order_cost (EUR per order), holding_rate (0.25 = 25% a year),\n"
              "#            cost_to_price_ratio (unit cost as a share of the price, used only where no cost is known)\n"
              "# simulation: target_fill_rate (0.95 = 95% of the demand served from stock)\n")
    (Path(folder) / SETTINGS).write_text(header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")


def company_config(folder: Path) -> Config:
    return load_config(overrides=load_settings(folder)["overrides"])


# --------------------------------------------------------------------------- files
@dataclass
class FileInfo:
    name: str
    role: str                    # one of ROLES, "unknown", "error" or "skipped"
    rows: int = 0
    columns: int = 0
    size: int = 0
    modified: float = 0.0
    message: str = ""
    notes: list[str] = field(default_factory=list)
    raw: pd.DataFrame | None = field(default=None, repr=False, compare=False)


def classify(columns, filename: str = "") -> str:
    """Which kind of file this is, from its columns and (in doubt) its name."""
    norm = {normalise_name(c) for c in columns}
    name = normalise_name(Path(filename).stem)
    hinted = [role for role, words in NAME_HINTS.items() if any(w in name for w in words)]

    def has(names) -> bool:
        return bool(norm & set(names))

    has_item = has(SYNONYMS["item"])
    sales = guess_mapping(columns)
    is_sales = all(sales[f] for f in ("date", "item", "quantity"))
    bought = purchase_columns(columns)
    is_purchases = all(bought[f] for f in ("order_date", "item", "quantity"))
    if has_item and has(_CALENDAR["start"]) and has(_CALENDAR["end"]):
        return "promo_calendar"
    if has_item and has(STOCK_SYNONYMS["on_hand"]) and not (is_sales and "sales" in hinted):
        return "stock"
    if not has_item and has(_TERM_COLUMNS["supplier"]) and any(
            has(v) for k, v in _TERM_COLUMNS.items() if k != "supplier"):
        return "supplier_terms"
    if is_purchases and ("purchases" in hinted or (bought["receipt_date"] and "sales" not in hinted)
                         or (bought["supplier"] and not sales["unit_price"] and not is_sales)):
        return "purchases"
    if is_sales:
        return "sales"
    if is_purchases and has(PURCHASE_COLUMNS["supplier"]):
        return "purchases"
    described = ("unit_price", "unit_cost", "lead_time_days", "supplier", "item_name", "pack_size", "min_order_qty",
                 "category")
    if has_item and any(has(SYNONYMS[f]) for f in described):
        return "item_master"
    return "unknown"


def data_files(folder: Path) -> list[Path]:
    root = Path(folder) / DATA
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith((".", "~$"))
                  and p.suffix.lower() in SUPPORTED)


def fingerprint(folder: Path) -> str:
    """Changes whenever a data file or the settings change: tells whether the saved analysis is current."""
    parts = [f"{p.relative_to(folder)}|{p.stat().st_size}|{p.stat().st_mtime_ns}" for p in data_files(folder)]
    settings = Path(folder) / SETTINGS
    if settings.exists():
        parts.append(hashlib.sha256(settings.read_bytes()).hexdigest())
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def scan(folder: Path, cfg: Config | None = None) -> list[FileInfo]:
    """Read and classify every file of ``data/``. Never raises: a file that cannot be read is listed with its error."""
    cfg = cfg or load_config()
    out = []
    for path in data_files(folder):
        stat = path.stat()
        info = FileInfo(str(path.relative_to(Path(folder) / DATA)), "unknown", size=stat.st_size, modified=stat.st_mtime)
        if stat.st_size > cfg.workspace.max_scan_mb * 1024 * 1024:
            info.role, info.message = "skipped", "File too large for the dashboard: use the batch mode."
        else:
            try:
                raw = read_any(path)
                info.raw, info.rows, info.columns = raw, len(raw), len(raw.columns)
                info.notes = list(raw.attrs.get("import_notes", []))
                info.role = classify(raw.columns, path.name)
                if info.role == "unknown":
                    info.message = "Columns not recognised: " + ", ".join(map(str, raw.columns[:8]))
            except InputError as exc:
                info.role, info.message = "error", str(exc)
        out.append(info)
    return out


# --------------------------------------------------------------------------- loading
@dataclass
class CompanyData:
    data: StandardisedData
    files: list[FileInfo]
    purchases: PurchaseSummary | None = None
    stock: pd.DataFrame | None = None            # [store, item, on_hand, on_order, arrival]
    terms: pd.DataFrame | None = None            # read_supplier_terms output
    promo_calendar: pd.DataFrame | None = None   # read_promo_calendar output
    notes: list[str] = field(default_factory=list)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Extra tables saved with the analysis and read by the Orders and Promotions pages."""
        out = {"ws_files": pd.DataFrame([{"name": f.name, "role": f.role, "rows": f.rows, "message": f.message}
                                         for f in self.files])}
        for name, df in (("ws_stock", self.stock), ("ws_terms", self.terms), ("ws_promo_calendar", self.promo_calendar)):
            if df is not None and len(df):
                out[name] = df
        return out


def _canonical(info: FileInfo, column_map: dict[str, str]) -> pd.DataFrame:
    """One sales file with the standard column names, dates and numbers already parsed (files differ in format)."""
    raw = info.raw
    guess = guess_mapping(raw.columns)
    guess.update({f: c for f, c in column_map.items() if f in FIELDS and c in raw.columns})
    out = pd.DataFrame(index=raw.index)
    for fld in FIELDS:
        col = guess.get(fld)
        if col is None:
            continue
        out[fld] = parse_dates(raw[col])[0] if fld == "date" else (
            parse_numbers(raw[col]) if fld in NUMERIC else raw[col])
    return out


def _latest(files: list[FileInfo], role: str) -> FileInfo | None:
    found = [f for f in files if f.role == role]
    return max(found, key=lambda f: f.modified) if found else None


def load_company(folder: Path, cfg: Config, files: list[FileInfo] | None = None) -> CompanyData:
    """Join the files of the company folder into the standard input. Raises ``InputError`` without a sales file."""
    files = scan(folder, cfg) if files is None else files
    sales_files = [f for f in files if f.role == "sales"]
    if not sales_files:
        raise InputError("No sales file found in the data folder.")
    notes: list[str] = []
    frames = [_canonical(f, cfg.input.column_map) for f in sales_files]
    raw = pd.concat(frames, ignore_index=True)
    mapping = ColumnMapping.from_dict({f: (f if f in raw.columns else None) for f in FIELDS})
    label = sales_files[0].name if len(sales_files) == 1 else f"{len(sales_files)} sales files"
    data = standardise(raw, mapping, cfg.input, cfg.prices, cfg.inventory, source="file", file_name=label,
                       file_hash_value=fingerprint(folder), promotions=cfg.promotions)
    if len(sales_files) > 1:
        notes.append(f"{len(sales_files)} sales files were joined: " + ", ".join(f.name for f in sales_files) + ".")

    summary = None
    bought = [f for f in files if f.role == "purchases"]
    if bought:
        parts = []
        for f in bought:
            cols = purchase_columns(f.raw.columns)
            part = pd.DataFrame({k: f.raw[c] for k, c in cols.items() if c is not None})
            for k in ("order_date", "receipt_date"):
                if k in part:
                    part[k] = parse_dates(part[k])[0]
            for k in ("quantity", "unit_cost", "amount"):
                if k in part:
                    part[k] = parse_numbers(part[k])
            parts.append(part)
        try:
            summary = read_purchases(pd.concat(parts, ignore_index=True), cfg.purchases, cfg.inventory.lead_time_days)
            data = apply_purchases(data, summary, cfg.prices, cfg.inventory)
        except InputError as exc:
            notes.append(f"Purchases not used: {exc}")
    for f in sorted((f for f in files if f.role == "item_master"), key=lambda f: f.modified):
        data = merge_item_master(data, f.raw, cfg.prices, cfg.inventory)

    stores = data.stores["store"].tolist()
    single = stores[0] if len(stores) == 1 else None
    stock = terms = calendar = None
    for role, reader in (("stock", lambda r: read_stock_file(r, single)), ("supplier_terms", read_supplier_terms),
                         ("promo_calendar", lambda r: read_promo_calendar(r, cfg.promotions.default_discount))):
        info = _latest(files, role)
        if info is None:
            continue
        try:
            table = reader(info.raw)
        except InputError as exc:
            notes.append(f"{info.name} not used: {exc}")
            continue
        if role == "stock":
            stock = table
        elif role == "supplier_terms":
            terms = table
        else:
            calendar = table
    if summary is not None and len(summary.open_orders):
        stock = _with_open_orders(stock, summary.open_orders, single, notes)
    return CompanyData(data, files, summary, stock, terms, calendar, notes)


def _with_open_orders(stock: pd.DataFrame | None, open_orders: pd.DataFrame, single_store: str | None,
                      notes: list[str]) -> pd.DataFrame | None:
    """Add the open purchase orders to a stock file that does not say what is on its way."""
    if stock is None or (stock["on_order"] > 0).any():
        return stock                                 # no stock on hand to start from, or the file already says it
    orders = open_orders.copy()
    if single_store:
        orders["store"] = single_store
    if (orders["store"] == "").any():
        notes.append("Open purchase orders have no store: they are not added to the stock on its way.")
        return stock
    merged = stock.drop(columns=["on_order", "arrival"]).merge(orders, on=["store", "item"], how="left")
    merged["on_order"] = merged["on_order"].fillna(0.0)
    return merged


# --------------------------------------------------------------------------- saved analysis
def save_results(folder: Path, tables: dict[str, pd.DataFrame], meta: dict) -> Path:
    """Write the analysis to ``results/`` (replacing the previous one) so the next start reopens it."""
    target = Path(folder) / RESULTS / "analysis"
    tmp = target.with_name("analysis.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name, df in tables.items():
        df.to_parquet(tmp / f"{name}.parquet", index=False)
    (tmp / "meta.json").write_text(json.dumps(meta, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    return target


def load_results(folder: Path) -> tuple[dict[str, pd.DataFrame], dict] | None:
    target = Path(folder) / RESULTS / "analysis"
    if not (target / "meta.json").exists():
        return None
    try:
        meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
        tables = {p.stem: pd.read_parquet(p) for p in sorted(target.glob("*.parquet"))}
    except (OSError, ValueError):
        return None
    return tables, meta


def analyse(folder: Path, cfg: Config | None = None, progress=None, fast: bool = True):
    """Read the folder, run the analysis and save it. Returns ``(tables, meta, company)``."""
    from src.pipeline import run_in_memory

    cfg = cfg or company_config(folder)
    company = load_company(folder, cfg)
    res = run_in_memory(company.data, cfg, fast=fast, progress=progress)
    tables = res.tables() | company.tables()
    meta = res.meta() | {"file_name": company.data.file_name, "sample": False, "workspace": {
        "fingerprint": fingerprint(folder), "analysed_at": datetime.now().isoformat(timespec="seconds"),
        "notes": company.notes, "company": load_settings(folder)["company"]}}
    save_results(folder, tables, meta)
    return tables, meta, company


def reorder_parameters(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reorder point, lot and safety stock per store and item (the forecast-driven policy of each store's channel)."""
    s = tables["policy_summary"]
    channel = tables["stores"].set_index("store")["channel"]
    plan = s[(s["variant"] == "calibrated") & (s["policy"] == "P2_forecast_sq")]
    plan = plan[plan["channel"] == plan["store"].map(channel)]
    return plan[["store", "item", "lead_time", "mean_daily", "reorder_point", "order_qty", "safety_stock"]]


# --------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Company folder: create it, list its files or analyse it.")
    parser.add_argument("--folder", type=Path, default=None, help="company folder (default: Documents/Demand Planner)")
    parser.add_argument("--init", action="store_true", help="create the folder structure")
    parser.add_argument("--status", action="store_true", help="list the files found and how they are read")
    parser.add_argument("--analyse", "--analyze", action="store_true", help="run the analysis and save it")
    args = parser.parse_args(argv)
    folder = ensure(args.folder or workspace_dir() or default_dir())
    print(f"Company folder: {folder}")
    if args.status or not (args.init or args.analyse):
        for f in scan(folder):
            print(f"  {f.role:<15} {f.rows:>9,} rows  {f.name}  {f.message}")
    if args.analyse:
        try:
            tables, meta, _ = analyse(folder, progress=lambda m, x: print(f"  {x:4.0%} {m}"))
        except (InputError, ValueError) as exc:
            print(f"Analysis stopped: {exc}", file=sys.stderr)
            return 1
        out = folder / RESULTS / "reorder_parameters.csv"
        reorder_parameters(tables).to_csv(out, index=False)
        print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
