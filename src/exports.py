"""Files handed to people outside the software: the purchase orders as an Excel workbook."""

from __future__ import annotations

import io
import re

import pandas as pd

SHEET_NAME_MAX = 31


def sheet_name(label: str, used: set[str]) -> str:
    """A valid, unique Excel sheet name (no ``[]:*?/\\``, at most 31 characters)."""
    base = re.sub(r"[\[\]:*?/\\]", " ", str(label)).strip()[:SHEET_NAME_MAX] or "-"
    name, k = base, 2
    while name.lower() in used:
        suffix = f" ({k})"
        name, k = base[:SHEET_NAME_MAX - len(suffix)] + suffix, k + 1
    used.add(name.lower())
    return name


def orders_workbook(orders: pd.DataFrame, labels: dict[str, str] | None = None,
                    summary_sheet: str = "Summary") -> bytes:
    """Order lines ``[supplier, store, item, ..., qty, value]`` -> ``.xlsx`` with a summary and one sheet per supplier.

    ``labels`` renames the columns for the reader (the dashboard passes them in its language).
    """
    labels = labels or {}
    by_supplier = (orders.groupby("supplier", sort=False)
                   .agg(lines=("qty", "size"), units=("qty", "sum"), value=("value", "sum"))
                   .sort_values("value", ascending=False).reset_index())
    buf = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as book:
        _write(book, by_supplier.rename(columns=labels), sheet_name(summary_sheet, used))
        for supplier in by_supplier["supplier"]:
            lines = orders[orders["supplier"] == supplier].drop(columns="supplier")
            _write(book, lines.rename(columns=labels), sheet_name(supplier, used))
    return buf.getvalue()


def _write(book: pd.ExcelWriter, df: pd.DataFrame, name: str) -> None:
    df.to_excel(book, sheet_name=name, index=False)
    sheet = book.sheets[name]
    for k, col in enumerate(df.columns):
        width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)])
        sheet.column_dimensions[sheet.cell(row=1, column=k + 1).column_letter].width = min(max(10, width + 2), 48)
    sheet.freeze_panes = "A2"
