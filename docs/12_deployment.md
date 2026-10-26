# 12 — Publishing the app

What to set before putting the dashboard on a public address, and what Streamlit can and cannot do as a website.

## 1. Checklist

| # | Item | Where | Done by |
|---|---|---|---|
| 1 | Owner and contact shown in the privacy policy and terms | `site.owner`, `site.contact_email` in `config/params.yaml` | you |
| 2 | Public address (sitemap, canonical links, Open Graph image) | `site.base_url` | you |
| 3 | Search engines allowed or not | `site.indexable` | you |
| 4 | `robots.txt` and `sitemap.xml` | `python -m src.site_files` → `app/static/` | command |
| 5 | Root paths `/robots.txt`, `/sitemap.xml` | reverse proxy (§3) | server |
| 6 | Upload limits that fit the host's memory | `input.limits` + `.streamlit/config.toml` `maxUploadSize` | you |
| 7 | Usage statistics on / off, optional Plausible | `analytics.*` | you |
| 8 | Privacy policy and terms reviewed | pages *Privacy* and *Terms* (templates, not legal advice) | lawyer |
| 9 | HTTPS | host or reverse proxy | server |
| 10 | Licence of the code | `LICENSE` file in the repository | you |

## 2. What the app already does

| Website essential | Implementation |
|---|---|
| Privacy policy, terms | pages `/privacy`, `/terms` (linked in the sidebar footer, hidden from the menu) |
| Clear call to action | home without results: "Try it now with sample data" (one click, ≈ 20 s), "Upload your sales history"; home with results: "Next: what to order today" |
| FAQ | page `/faq` (13 questions) |
| Custom 404 | an unknown address shows a "Page not found" page with links (recorded as `not_found`) |
| Page titles | `<page> · <site name>` in the browser tab for every page |
| Meta description, Open Graph, canonical, robots, `html lang` | written into `<head>` by a script on every page (`app/website.py`) |
| Alt text | logo (set by script), every chart (screen-reader description with its title and series), README images |
| Analytics | first-party anonymous events (§4), optional cookie-free Plausible |

## 3. robots.txt and sitemap.xml at the site root

Streamlit serves extra files only under `/app/static/` (static serving is on in `.streamlit/config.toml`).
Crawlers look for `/robots.txt` and `/sitemap.xml` at the root, so the reverse proxy in front of Streamlit maps them:

```bash
python -m src.site_files --base-url https://planner.example.com
```

**Caddy**

```
planner.example.com {
    rewrite /robots.txt /app/static/robots.txt
    rewrite /sitemap.xml /app/static/sitemap.xml
    reverse_proxy localhost:8501
}
```

**nginx**

```
location = /robots.txt  { proxy_pass http://127.0.0.1:8501/app/static/robots.txt; }
location = /sitemap.xml { proxy_pass http://127.0.0.1:8501/app/static/sitemap.xml; }
location / {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;      # Streamlit uses a WebSocket
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

On Streamlit Community Cloud there is no proxy: the files stay at `/app/static/…`, and the sitemap can be submitted
by hand in Google Search Console.

## 4. Usage analytics

Every page change and a few actions are appended to `data/analytics/events.jsonl` (git-ignored), one JSON line:

```json
{"ts": "2026-10-26T10:42Z", "session": "3f9c1a2b7d04", "event": "page_view", "page": "orders", "lang": "it", "source": "sample"}
```

| Event | When | Extra fields |
|---|---|---|
| `page_view` | the visitor opens a page (not on every click inside it) | — |
| `cta_click` | home call to action | `target` |
| `sample_run`, `analysis_run` | an analysis finishes | `series`, `seconds`, `kind` |
| `download` | any CSV download | `target` (file name without extension) |
| `not_found` | an unknown address | — |

Never recorded: IP address, user agent, cookies, file names or contents, typed values. *Do Not Track* and
*Global Privacy Control* browsers are skipped; visitors can opt out on the Privacy page. Report:

```bash
python -m src.analytics            # sessions, page views, visit -> analysis -> download funnel, pages, languages, days
```

Keep the file at most 13 months (as the privacy policy says): delete or trim it periodically.

## 5. Limits of Streamlit as a website

- The pages are drawn in the browser by JavaScript from one HTML shell: the `<head>` tags are written by a script,
  so search engines that run JavaScript (Google, Bing) read them, simple link previews (some chat apps) may only see
  the default title. For marketing pages that must rank, a static landing page (any site builder) linking to the app
  is the usual setup.
- A 404 is shown as a page, but the HTTP status is still 200 (Streamlit answers every path with the same shell).
- The free hosts have about 1 GB of memory: keep the upload limits low there (≈ 50 MB, 1,000 series).
