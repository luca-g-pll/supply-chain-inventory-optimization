"""``robots.txt`` and ``sitemap.xml`` for the published dashboard.

``python -m src.site_files --base-url https://planner.example.com`` writes both files to
``app/static/`` (Streamlit serves them at ``/app/static/…`` with static serving on).
Search engines read them only at the site root: the reverse proxy in front of the app
maps ``/robots.txt`` and ``/sitemap.xml`` to those files (docs/12_deployment.md).

Only the public pages are listed; the 404 page is never indexed.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

# url path -> (priority, change frequency); "" is the home page (Overview)
PUBLIC_PAGES: dict[str, tuple[float, str]] = {
    "": (1.0, "weekly"), "data": (0.9, "monthly"), "faq": (0.8, "monthly"), "summary": (0.7, "monthly"),
    "demand": (0.6, "monthly"), "forecast": (0.6, "monthly"), "promotions": (0.6, "monthly"),
    "inventory": (0.6, "monthly"), "orders": (0.6, "monthly"), "privacy": (0.3, "yearly"),
    "terms": (0.3, "yearly"),
}
STATIC_DIR = Path(__file__).resolve().parents[1] / "app" / "static"


def robots_txt(base_url: str, indexable: bool = True) -> str:
    base = base_url.rstrip("/")
    if not indexable:
        return "User-agent: *\nDisallow: /\n"
    lines = ["User-agent: *", "Allow: /", "Disallow: /_stcore/"]     # /_stcore = Streamlit internals
    if base:
        lines += ["", f"Sitemap: {base}/sitemap.xml"]
    return "\n".join(lines) + "\n"


def sitemap_xml(base_url: str, pages: dict[str, tuple[float, str]] | None = None,
                lastmod: date | None = None) -> str:
    if not base_url:
        raise ValueError("A sitemap needs the public address of the app (site.base_url / --base-url).")
    base = base_url.rstrip("/")
    lastmod = (lastmod or date.today()).isoformat()
    urls = []
    for path, (priority, freq) in (pages or PUBLIC_PAGES).items():
        loc = escape(f"{base}/{path}" if path else f"{base}/")
        urls.append(f"  <url>\n    <loc>{loc}</loc>\n    <lastmod>{lastmod}</lastmod>\n"
                    f"    <changefreq>{freq}</changefreq>\n    <priority>{priority:.1f}</priority>\n  </url>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            + "\n".join(urls) + "\n</urlset>\n")


def write_site_files(base_url: str, indexable: bool = True, folder: Path = STATIC_DIR) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = [folder / "robots.txt"]
    out[0].write_text(robots_txt(base_url, indexable), encoding="utf-8")
    if base_url:
        out.append(folder / "sitemap.xml")
        out[1].write_text(sitemap_xml(base_url), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    from src.config import load_config

    cfg = load_config()
    p = argparse.ArgumentParser(prog="python -m src.site_files", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=cfg.site.base_url, help="public address of the app (default: site.base_url)")
    p.add_argument("--noindex", action="store_true", help="ask search engines not to index the app")
    args = p.parse_args(argv)
    for f in write_site_files(args.base_url, indexable=cfg.site.indexable and not args.noindex):
        print(f"Wrote {f}")
    if not args.base_url:
        print("No base URL: sitemap.xml skipped (set site.base_url in config/params.yaml or pass --base-url).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
