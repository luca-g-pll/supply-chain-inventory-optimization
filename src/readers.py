"""File readers: every format a small company keeps its sales or purchases in -> one raw table.

``read_table`` is the single entry point (``src.ingest.read_any`` calls it). Supported:
delimited text (``.csv .tsv .txt .dat``), Excel (``.xlsx .xlsm .xls``), OpenDocument
(``.ods``), JSON and JSON lines (also nested order lines), Markdown tables, HTML tables,
XML and Parquet. Three things spreadsheets exported by hand usually need are handled
here too: title rows above the header, a file whose extension lies about its content
(an HTML or text export saved as ``.xls``) and *wide* tables with one column per period.

Notes about what was done are returned in ``df.attrs["import_notes"]``.
"""

from __future__ import annotations

import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import BinaryIO

import pandas as pd

TEXT = (".csv", ".tsv", ".txt", ".dat", ".tab")
EXCEL = (".xlsx", ".xlsm", ".xls", ".ods")
JSON = (".json", ".jsonl", ".ndjson")
MARKDOWN = (".md", ".markdown")
HTML = (".html", ".htm")
SUPPORTED = TEXT + EXCEL + JSON + MARKDOWN + HTML + (".xml", ".parquet", ".pq")
HEADER_SCAN_ROWS = 30
MIN_WIDE_COLUMNS = 4

_ZIP, _OLE = b"PK\x03\x04", b"\xd0\xcf\x11\xe0"


class InputError(ValueError):
    """A problem with the user's file, with a message meant for the user."""


def as_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    pos = source.tell() if hasattr(source, "tell") else None
    data = source.read()
    if pos is not None and hasattr(source, "seek"):
        source.seek(pos)
    return data


def extension(source, filename: str | None) -> str:
    name = filename or getattr(source, "name", None) or (str(source) if isinstance(source, (str, Path)) else "")
    ext = Path(str(name)).suffix.lower()
    if not ext:
        raise InputError("Cannot tell the file type: use a .csv, .xlsx or .parquet file.")
    return ext


def decode(data: bytes, limit: int | None = None) -> tuple[str, str]:
    chunk = data if limit is None else data[:limit]
    if chunk[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return chunk.decode("utf-16", errors="replace"), "utf-16"
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return chunk.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise InputError("The file is not readable text (unknown encoding).")


def sniff_delimiter(sample: str) -> str:
    """The separator that splits most lines into the same number of fields (title lines do not count)."""
    lines = [ln for ln in sample.splitlines()[:200] if ln.strip()]
    best, best_score = ",", (0, 0)
    for sep in ",;\t|":
        counts = [ln.count(sep) for ln in lines]
        used = [c for c in counts if c > 0]
        if not used:
            continue
        typical = max(set(used), key=used.count)
        score = (used.count(typical), typical)
        if score > best_score:
            best, best_score = sep, score
    return best


# --------------------------------------------------------------------------- header row
def text_header_row(sample: str, sep: str) -> int:
    """Index of the header line of a delimited text: the first line with as many fields as the body.

    Title lines above the table (``Report vendite 2024``) have fewer separators than the rows below.
    """
    lines = [ln for ln in sample.splitlines()[:HEADER_SCAN_ROWS + 20]]
    counts = [ln.count(sep) for ln in lines]
    body = [c for c in counts if c > 0]
    if not body:
        return 0
    typical = max(set(body), key=body.count)
    return next((k for k, c in enumerate(counts) if c >= typical), 0)


def grid_header_row(head: pd.DataFrame) -> int:
    """Index of the header row in the first rows of a sheet read without a header.

    The header is the first row that is about as wide as the widest of the first rows and is
    made of text labels; rows above it are titles, logos or blank lines.
    """
    if head.empty:
        return 0
    filled = head.notna() & (head.astype(str).apply(lambda s: s.str.strip()) != "")
    width = filled.sum(axis=1)
    widest = int(width.max())
    if widest == 0:
        return 0
    for k in range(len(head)):
        if width.iloc[k] < max(2, 0.6 * widest):
            continue
        cells = head.iloc[k][filled.iloc[k]]
        labels = sum(isinstance(v, str) and not _is_number(v) for v in cells)
        if labels >= 0.7 * len(cells):
            return k
    return 0


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?[\d.,\s€$%]+", text.strip()))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows and columns, give every column a unique text name."""
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    names, seen = [], {}
    for k, col in enumerate(df.columns):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed:") or name.lower() == "nan":
            name = f"column_{k + 1}"
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    df.columns = names
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- formats
def read_text(data: bytes, ext: str = ".csv") -> pd.DataFrame:
    sample, encoding = decode(data, 65536)
    sep = "\t" if ext in (".tsv", ".tab") else sniff_delimiter(sample)
    skip = text_header_row(sample, sep)
    df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, encoding=encoding, skipinitialspace=True,
                     skiprows=skip, skip_blank_lines=True, on_bad_lines="skip")
    if skip:
        df.attrs["import_notes"] = [f"{skip} title line(s) above the header were skipped."]
    return df


def sheet_names(data: bytes, ext: str = ".xlsx") -> list[str]:
    try:
        with pd.ExcelFile(io.BytesIO(data), engine=_excel_engine(data, ext)) as book:
            return [str(s) for s in book.sheet_names]
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"Could not open the workbook: {exc}") from exc


def _excel_engine(data: bytes, ext: str) -> str:
    if ext == ".ods":
        return "odf"
    if data[:4] == _OLE:
        return "xlrd"
    if data[:4] == _ZIP:
        return "openpyxl"
    raise InputError("The file is not an Excel workbook.")


def read_excel(data: bytes, ext: str, sheet: str | int | None = None) -> pd.DataFrame:
    """Excel / OpenDocument sheet. With no ``sheet`` the first sheet that has a table is read.

    A file named ``.xls`` that is really an HTML page or delimited text (common in exports of
    older management software) is read as what it is.
    """
    if ext == ".xls" and data[:4] not in (_OLE, _ZIP):
        head = data[:2048].lstrip().lower()
        notes = ["The file is named .xls but is not an Excel workbook: it was read as "]
        if head.startswith(b"<") or b"<table" in head:
            df = read_html(data)
            df.attrs["import_notes"] = [notes[0] + "an HTML table."]
        else:
            df = read_text(data)
            df.attrs["import_notes"] = [notes[0] + "delimited text."] + df.attrs.get("import_notes", [])
        return df
    engine = _excel_engine(data, ext)
    with pd.ExcelFile(io.BytesIO(data), engine=engine) as book:
        names = list(book.sheet_names)
        wanted = [names[sheet] if isinstance(sheet, int) else sheet] if sheet is not None else names
        for name in wanted:
            head = book.parse(name, header=None, nrows=HEADER_SCAN_ROWS)
            if head.dropna(how="all").shape[0] < 2:
                continue
            row = grid_header_row(head)
            df = book.parse(name, header=row)
            notes = []
            if row:
                notes.append(f"{row} title line(s) above the header were skipped.")
            if sheet is None and name != names[0]:
                notes.append(f"Sheet {name!r} was read (the sheets before it are empty).")
            df.attrs["import_notes"] = notes
            return df
    raise InputError("The file has no rows.")


def _records(obj, depth: int = 0) -> list[dict] | None:
    """The largest list of objects inside a JSON document (``{"data": {"rows": [...]}}``)."""
    if isinstance(obj, list):
        return obj if obj and all(isinstance(r, dict) for r in obj[:50]) else None
    if isinstance(obj, dict) and depth < 4:
        found = [r for r in (_records(v, depth + 1) for v in obj.values()) if r]
        return max(found, key=len) if found else None
    return None


def read_json(data: bytes) -> pd.DataFrame:
    """JSON array of records, an object wrapping one, a column-oriented object or JSON lines.

    Records that hold their lines in a nested list (orders with ``line_items`` / ``righe``) are
    opened into one row per line, with the order's own fields repeated on every line.
    """
    text, _ = decode(data)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            obj = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        except json.JSONDecodeError as exc:
            raise InputError(f"Could not read the file: not valid JSON ({exc.msg}, line {exc.lineno}).") from exc
    rows = _records(obj)
    if rows is None:
        if isinstance(obj, dict) and obj and all(isinstance(v, (list, dict)) for v in obj.values()):
            try:
                return pd.DataFrame(obj)                       # column-oriented: {"date": [...], "qty": [...]}
            except ValueError as exc:
                raise InputError(f"Could not read the file: {exc}") from exc
        raise InputError("The JSON file has no list of records.")
    sample = rows[:200]
    nested = [k for k in sample[0] if isinstance(sample[0][k], list)
              and sum(isinstance(r.get(k), list) and all(isinstance(x, dict) for x in r[k]) for r in sample)
              >= 0.8 * len(sample) and any(r.get(k) for r in sample)]
    if nested:
        key = nested[0]
        meta = [k for k in sample[0] if k != key and not isinstance(sample[0][k], (list, dict))]
        df = pd.json_normalize(rows, record_path=key, meta=meta, errors="ignore", sep="_",
                                       record_prefix=f"{key}_" if set(meta) & _line_keys(sample, key) else None)
        df.attrs["import_notes"] = [f"Nested list {key!r} was opened: one row per line."]
        return df
    return pd.json_normalize(rows, sep="_")


def _line_keys(sample: list[dict], key: str) -> set[str]:
    return {k for r in sample for line in (r.get(key) or [])[:5] for k in line}


def read_markdown(data: bytes) -> pd.DataFrame:
    """The largest pipe table of a Markdown document."""
    text, _ = decode(data)
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines() + [""]:
        s = line.strip()
        if s.startswith("|") or (s.count("|") >= 1 and current):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{1,}:?", c) for c in cells if c) or not any(cells):
                current.append(cells)
        else:
            if len(current) >= 2:
                tables.append(current)
            current = []
    if not tables:
        raise InputError("The Markdown file has no table (rows written as | a | b | c |).")
    best = max(tables, key=len)
    width = len(best[0])
    body = [(r + [""] * width)[:width] for r in best[1:]]
    df = pd.DataFrame(body, columns=best[0]).replace("", pd.NA)
    if len(tables) > 1:
        df.attrs["import_notes"] = [f"The file has {len(tables)} tables: the largest ({len(body):,} rows) was read."]
    return df


class _Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self.tables:
            if any(self._row):
                self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def read_html(data: bytes) -> pd.DataFrame:
    """The largest ``<table>`` of an HTML page."""
    text, _ = decode(data)
    parser = _Tables()
    parser.feed(text)
    tables = [t for t in parser.tables if len(t) >= 2]
    if not tables:
        raise InputError("The HTML file has no table.")
    best = max(tables, key=len)
    grid = pd.DataFrame(best).replace("", pd.NA)
    row = grid_header_row(grid.head(HEADER_SCAN_ROWS))
    df = grid.iloc[row + 1:].reset_index(drop=True)
    df.columns = [str(c) if pd.notna(c) else "" for c in grid.iloc[row]]
    return df


def read_xml(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_xml(io.BytesIO(data), parser="etree", dtype=str)
    except Exception as exc:
        raise InputError(f"Could not read the file: the XML has no repeated record element ({exc}).") from exc


# --------------------------------------------------------------------------- wide tables
_MONTHS = {m: k + 1 for k, group in enumerate([
    ("gen", "jan"), ("feb",), ("mar",), ("apr",), ("mag", "may"), ("giu", "jun"), ("lug", "jul"), ("ago", "aug"),
    ("set", "sep"), ("ott", "oct"), ("nov",), ("dic", "dec")]) for m in group}
_MONTH_NAME = re.compile(r"^([a-z]{3})[a-z]*[\s._/\-']*(\d{2}|\d{4})$")
_YEAR_MONTH = re.compile(r"^(\d{4})[\-/._m](\d{1,2})$")
_MONTH_YEAR = re.compile(r"^(\d{1,2})[\-/.](\d{4})$")
_FULL_DATE = re.compile(r"^\d{1,4}[\-/.]\d{1,2}[\-/.]\d{1,4}$")


def period_of(label) -> pd.Timestamp | None:
    """A column label that names a day or a month (``2024-03``, ``mar 24``, ``03/2024``, ``15/03/2024``)."""
    if isinstance(label, (pd.Timestamp,)) or hasattr(label, "year") and hasattr(label, "month"):
        return pd.Timestamp(label).normalize()
    text = str(label).strip().lower()
    if " 00:00:00" in text:
        text = text.split(" ")[0]
    m = _MONTH_NAME.match(text)
    if m and m.group(1) in _MONTHS:
        year = int(m.group(2))
        return pd.Timestamp(year + 2000 if year < 100 else year, _MONTHS[m.group(1)], 1)
    m = _YEAR_MONTH.match(text)
    if m and 1 <= int(m.group(2)) <= 12:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    m = _MONTH_YEAR.match(text)
    if m and 1 <= int(m.group(1)) <= 12:
        return pd.Timestamp(int(m.group(2)), int(m.group(1)), 1)
    if _FULL_DATE.match(text):
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=not text[:4].isdigit())
        return None if pd.isna(parsed) else parsed.normalize()
    return None


def unpivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a table with one column per day or month into rows ``[..., date, quantity]``.

    Applied only when at least ``MIN_WIDE_COLUMNS`` column names are periods and no column is
    already a date column, so a normal long file is returned unchanged.
    """
    periods = {c: period_of(c) for c in df.columns}
    wide = [c for c, p in periods.items() if p is not None]
    keys = [c for c in df.columns if c not in wide]
    if len(wide) < MIN_WIDE_COLUMNS or not keys or len(wide) < len(keys):
        return df
    long = df.melt(id_vars=keys, value_vars=wide, var_name="date", value_name="quantity")
    long["date"] = long["date"].map(periods)
    long = long.dropna(subset=["quantity"]).reset_index(drop=True)
    long.attrs["import_notes"] = df.attrs.get("import_notes", []) + [
        f"The table had one column per period ({len(wide)} columns): it was turned into one row per period."]
    return long


# --------------------------------------------------------------------------- entry point
def read_table(source: str | Path | bytes | BinaryIO, filename: str | None = None,
               sheet: str | int | None = None) -> pd.DataFrame:
    """Read any supported file into a raw table (text columns are converted later by ``standardise``)."""
    ext = extension(source, filename)
    if ext not in SUPPORTED:
        raise InputError(f"Unsupported file type {ext!r}: use .csv, .xlsx or .parquet.")
    data = as_bytes(source)
    if not data.strip():
        raise InputError("The file has no rows.")
    try:
        if ext in TEXT:
            df = read_text(data, ext)
        elif ext in EXCEL:
            df = read_excel(data, ext, sheet)
        elif ext in JSON:
            df = read_json(data)
        elif ext in MARKDOWN:
            df = read_markdown(data)
        elif ext in HTML:
            df = read_html(data)
        elif ext == ".xml":
            df = read_xml(data)
        else:
            df = pd.read_parquet(io.BytesIO(data))
    except InputError:
        raise
    except ImportError as exc:
        raise InputError(f"Could not read the file: a component is missing ({exc.name}). "
                         "Run the installation again.") from exc
    except Exception as exc:  # pandas / openpyxl / pyarrow errors
        raise InputError(f"Could not read the file: {exc}") from exc
    notes = df.attrs.get("import_notes", [])
    df = _clean(df)
    if df.empty or len(df.columns) == 0:
        raise InputError("The file has no rows.")
    df.attrs["import_notes"] = notes
    return unpivot_wide(df)
