# 13 · Product mode: installed in a company

The project started as a hosted demo: one browser session, files uploaded by hand, nothing kept. This document
describes what turns it into a program a small company installs and uses every week.

## 1. What changes between the two modes

| | Hosted demo | Installed (launcher or standalone program) |
|---|---|---|
| Start | a URL | double click |
| Data | one file uploaded in the browser | every file of the company folder, any supported format |
| Persistence | the browser session | the last analysis and the settings stay on the computer |
| Settings | typed at every visit | `settings.yaml` in the company folder |
| Switch | – | environment variable `PLANNER_WORKSPACE` = the company folder |

`src/workspace.workspace_dir()` returns the folder when `PLANNER_WORKSPACE` is set and `None` otherwise; every
difference in the dashboard hangs on that one test, so the hosted demo is unchanged and still writes nothing to disk.

## 2. Starting the software

- `launcher.py` (standard library only): creates `.venv` and installs `requirements.txt` when they are missing or
  when the requirements changed (their hash is stored in `.venv/planner_launcher.json`), creates the company folder,
  picks a free port from 8501, starts Streamlit bound to `localhost`, waits for `/_stcore/health` and opens the
  browser. Started twice, it finds the running dashboard and only opens the browser.
- `Demand Planner.command` / `Demand Planner.bat`: find a Python ≥ 3.11 and run the launcher.
- `scripts/build_standalone.py` (PyInstaller, `requirements-build.txt`): a folder with one executable and Python
  inside. `packaging/planner_app.py` is its entry point; it turns off the process pool of joblib
  (`JOBLIB_MULTIPROCESSING=0`), because a frozen program would start itself again instead of a worker. The build is
  per operating system and unsigned.

## 3. Reading the company's files (`src/readers.py`)

| Format | Notes |
|---|---|
| `.csv .tsv .txt .dat` | separator = the one that splits most lines into the same number of fields; title lines above the header skipped; UTF-8, UTF-16, cp1252 |
| `.xlsx .xlsm .xls .ods` | first sheet that holds a table; header row found among the first 30 rows; a `.xls` that is really HTML or text is read as such |
| `.json .jsonl .ndjson` | list of records, also wrapped in an object or column-oriented; nested order lines are opened into one row per line |
| `.md .markdown` | the largest pipe table |
| `.html .htm` | the largest `<table>` |
| `.xml` | repeated record elements |
| `.parquet` | as is |

A table with one column per month or day (`gen 24`, `2024-02`, `03/2024`…) is turned into rows. Weekly or monthly
totals are detected from the gaps between dates and spread evenly over their days (`input.granularity`).

## 4. Recognising what a file is (`src/workspace.classify`)

By columns first, by file name in doubt: promotion calendar (item + start + end), stock (item + stock on hand),
supplier terms (supplier + order days / minimum / discount, no item), purchases (order date + item + quantity with a
receipt date, or a supplier and no selling price, or a name like *acquisti*), sales (date + item + quantity), item
list (item + price / cost / lead time / supplier). Anything else is listed as *not recognised* with its columns.

Several sales files are joined after each one is read with its own column names, date format and decimal mark.

## 5. Purchase history (`src/purchases.py`)

| Value | Rule | Setting |
|---|---|---|
| Unit cost | median cost of the item's purchases in the last 90 days (line amount ÷ quantity when there is no unit cost) | `purchases.cost_window_days` |
| Supplier | supplier of the most recent purchase | |
| Lead time | median days from order to receipt; the item's own with ≥ 2 receipts, otherwise its supplier's | `purchases.min_lead_time_orders` |
| Pack size | greatest common divisor of the quantities bought, with ≥ 3 purchases and a divisor ≥ 2 | `purchases.min_orders_for_pack` |
| Open orders | lines without a receipt date in the last 60 days of the file; arrival = order date + lead time | `purchases.open_order_max_days` |

Order of precedence item by item: sales file < purchase history < item list.

## 6. What the Orders page gained

- stock and supplier terms come from the company folder when the files are there;
- today's orders as an Excel workbook, one sheet per supplier (`src/exports.orders_workbook`);
- **excess stock** (`src/replenishment.stock_health`): stock covering more than `stock_health.excess_cover_days`
  (90) of the sales expected in the next `horizon_days` (28), and stock with no sale expected.

## 7. Compared with the tools on the market

| Capability | Typical inventory-planning SaaS | This software |
|---|---|---|
| Data in | connectors to ERP / e-commerce platforms | the files the company already exports, any format, dropped in a folder |
| Where the data lives | the vendor's cloud | the company's computer only |
| Forecast, reorder points, supplier constraints, promotions, new products | yes | yes |
| Overstock / dead-stock report | yes | yes (Orders → Excess stock) |
| Purchase orders | created and sent, often synced back to the ERP | Excel / CSV per supplier, sent by the user |
| Proof on the company's own past | rarely shown | policy comparison on the test period, with the assumptions listed |
| Users | accounts, roles, shared workspace | one user per installation |
| Updates | continuous | a new copy of the program |

Still missing for a full commercial release: ERP / e-commerce connectors, sending the orders, user accounts, signed
installers with automatic updates, transfers between stores, a support channel and a commercial licence (the
repository is public under the MIT licence: anyone may reuse the code).
