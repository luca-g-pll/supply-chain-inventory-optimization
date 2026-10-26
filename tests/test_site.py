"""Website essentials: robots.txt, sitemap.xml, anonymous analytics (T-W1 .. T-W5)."""

import json
from datetime import date

import pytest

from src.analytics import log_event, read_events, tracking_allowed, usage_report
from src.site_files import PUBLIC_PAGES, robots_txt, sitemap_xml, write_site_files


def test_robots_txt():  # T-W1
    txt = robots_txt("https://planner.example.com/")
    assert "Allow: /" in txt and "Disallow: /_stcore/" in txt
    assert "Sitemap: https://planner.example.com/sitemap.xml" in txt
    assert robots_txt("https://x.io", indexable=False) == "User-agent: *\nDisallow: /\n"
    assert "Sitemap" not in robots_txt("")


def test_sitemap_lists_public_pages_only(tmp_path):  # T-W2
    xml = sitemap_xml("https://planner.example.com", lastmod=date(2026, 10, 26))
    assert xml.count("<url>") == len(PUBLIC_PAGES)
    assert "<loc>https://planner.example.com/</loc>" in xml and "/orders</loc>" in xml
    assert "404" not in xml and "<lastmod>2026-10-26</lastmod>" in xml
    with pytest.raises(ValueError):
        sitemap_xml("")
    files = write_site_files("https://planner.example.com", folder=tmp_path)
    assert {f.name for f in files} == {"robots.txt", "sitemap.xml"}


def test_tracking_respects_privacy_signals():  # T-W3
    assert tracking_allowed({}) and tracking_allowed({"User-Agent": "x"})
    assert not tracking_allowed({"DNT": "1"}) and not tracking_allowed({"Sec-GPC": "1"})
    assert not tracking_allowed({}, opted_out=True)


def test_events_hold_no_personal_data(tmp_path):  # T-W4
    path = tmp_path / "events.jsonl"
    assert log_event(path, "page_view", "abc123", page="orders", lang="it", source="sample",
                     ip="1.2.3.4", file_name="clienti.csv", series=256)
    assert not log_event(path, "unknown_event", "abc123")
    row = json.loads(path.read_text().splitlines()[0])
    assert set(row) == {"ts", "session", "event", "page", "lang", "source", "series"}
    assert row["ts"].endswith("Z") and len(row["ts"]) == len("2026-10-26T10:42Z")    # minute precision
    assert not log_event(tmp_path / "missing" / "\0bad", "page_view", "x")              # never raises


def test_usage_report_funnel(tmp_path):  # T-W5
    path = tmp_path / "events.jsonl"
    for s, events in (("a", ["page_view", "page_view", "sample_run", "download"]), ("b", ["page_view"]),
                      ("c", ["page_view", "analysis_run"])):
        for e in events:
            log_event(path, e, s, page="orders")
    rep = usage_report(read_events(path))
    t = rep["totals"]
    assert t["sessions"] == 3 and t["page_views"] == 4
    assert t["sessions_with_analysis"] == 2 and t["download_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert rep["by_page"].iloc[0]["page"] == "orders"
    assert usage_report(read_events(tmp_path / "none.jsonl"))["totals"]["sessions"] == 0
