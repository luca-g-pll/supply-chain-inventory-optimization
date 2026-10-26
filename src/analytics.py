"""First-party usage analytics: anonymous, cookie-free, one JSON line per event.

What is recorded (privacy policy, page "Privacy"): time (to the minute, UTC), a random
session id that lives only in the browser tab, the event (``page_view``,
``analysis_run``, ``download`` …), the page, the language and the kind of data in use
(demo / sample / upload). **Never** recorded: IP address, browser, cookies, file names,
file contents or any value typed in the app. Browsers that send *Do Not Track* or
*Global Privacy Control* are not recorded, and a visitor can opt out on the Privacy page.

``python -m src.analytics`` prints the usage report.
"""

from __future__ import annotations

import argparse
import json
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

EVENTS = ("page_view", "analysis_run", "sample_run", "download", "cta_click", "language", "not_found")
_ALLOWED_DETAIL = ("rows", "series", "seconds", "kind", "target", "fast")


def new_session_id() -> str:
    return secrets.token_hex(6)


def tracking_allowed(headers: Mapping[str, str] | None, opted_out: bool = False) -> bool:
    """False when the visitor opted out or the browser asks not to be tracked (DNT / GPC)."""
    if opted_out:
        return False
    h = {str(k).lower(): str(v).strip() for k, v in (headers or {}).items()}
    return h.get("dnt") != "1" and h.get("sec-gpc") != "1"


def log_event(path: str | Path, event: str, session: str, page: str = "", lang: str = "", source: str = "",
              **detail) -> bool:
    """Append one event; never raises (analytics must not break the app). Returns True if written."""
    if event not in EVENTS:
        return False
    row = {"ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"), "session": session, "event": event,
           "page": str(page)[:40], "lang": str(lang)[:5], "source": str(source)[:12]}
    row.update({k: v for k, v in detail.items() if k in _ALLOWED_DETAIL and isinstance(v, (int, float, str, bool))})
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except (OSError, ValueError):
        return False


def read_events(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["ts", "session", "event", "page", "lang", "source"])
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(rows)
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def usage_report(events: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    """Sessions, page views, funnel from visit to analysis to download, pages, languages, days."""
    if events.empty:
        return {"totals": {"sessions": 0, "page_views": 0}}
    pv = events[events["event"] == "page_view"]
    sessions = events["session"].nunique()
    ran = events.loc[events["event"].isin(["analysis_run", "sample_run"]), "session"].nunique()
    downloaded = events.loc[events["event"] == "download", "session"].nunique()
    totals = {
        "sessions": int(sessions), "page_views": int(len(pv)),
        "pages_per_session": round(len(pv) / sessions, 1) if sessions else 0.0,
        "sessions_with_analysis": int(ran), "sessions_with_download": int(downloaded),
        "analysis_rate": round(ran / sessions, 3) if sessions else 0.0,
        "download_rate": round(downloaded / sessions, 3) if sessions else 0.0,
        "not_found": int((events["event"] == "not_found").sum()),
    }
    by_page = (pv.groupby("page").agg(views=("session", "size"), sessions=("session", "nunique"))
               .sort_values("views", ascending=False).reset_index())
    by_day = (events.assign(day=events["ts"].dt.date).groupby("day")
              .agg(sessions=("session", "nunique"), events=("event", "size")).reset_index())
    by_lang = pv.groupby("lang")["session"].nunique().rename("sessions").reset_index()
    by_event = events.groupby("event").size().rename("count").reset_index()
    return {"totals": totals, "by_page": by_page, "by_day": by_day, "by_lang": by_lang, "by_event": by_event}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.analytics", description="Usage report of the dashboard.")
    p.add_argument("--path", default="data/analytics/events.jsonl")
    args = p.parse_args(argv)
    rep = usage_report(read_events(args.path))
    print("Totals:", json.dumps(rep["totals"], indent=2))
    for key in ("by_page", "by_event", "by_lang", "by_day"):
        if key in rep:
            print(f"\n{key}\n{rep[key].to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
