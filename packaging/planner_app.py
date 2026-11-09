"""Entry point of the standalone build (``scripts/build_standalone.py``): one program, nothing to install.

It does what ``launcher.py`` does for a source checkout, without the setup step: creates the company
folder, starts the dashboard on this computer only and opens the browser. Python and every
component are inside the program folder.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


def bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def _open_when_ready(port: int) -> None:
    import launcher

    for _ in range(300):
        if launcher.is_running(port):
            webbrowser.open(f"http://localhost:{port}")
            return
        time.sleep(0.4)


def main() -> int:
    root = bundle_root()
    sys.path.insert(0, str(root))
    os.chdir(root)
    import launcher

    # the pool processes of joblib would start this program again instead of a Python worker
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    folder = launcher.company_folder(None)
    os.environ[launcher.ENV_VAR] = str(folder)
    browser = not os.environ.get("PLANNER_NO_BROWSER")
    last = launcher.last_port(folder)
    if last and launcher.is_running(last):
        launcher.say(f"Already running: http://localhost:{last}")
        if browser:
            webbrowser.open(f"http://localhost:{last}")
        return 0
    port = launcher.free_port(int(os.environ.get("PLANNER_PORT", launcher.FIRST_PORT)))
    launcher.remember_port(folder, port)
    launcher.say(f"Dashboard: http://localhost:{port}")
    launcher.say(f"Company folder: {folder}")
    launcher.say("Keep this window open while you work. Close it to stop the software.")
    if browser:
        threading.Thread(target=_open_when_ready, args=(port,), daemon=True).start()

    from streamlit.web import bootstrap

    flags = {"server.headless": True, "server.address": "localhost", "server.port": port,
             "server.fileWatcherType": "none", "server.enableStaticServing": True, "server.maxUploadSize": 300,
             "browser.gatherUsageStats": False, "client.toolbarMode": "minimal", "global.developmentMode": False}
    bootstrap.load_config_options(flag_options=flags)
    bootstrap.run(str(root / "app" / "streamlit_app.py"), False, [], flags)
    return 0


def _components() -> None:
    """Never called: tells the packaging tool which modules the dashboard pages load from disk at run time."""
    import duckdb  # noqa: F401
    import lightgbm  # noqa: F401
    import matplotlib.pyplot  # noqa: F401
    import odf.opendocument  # noqa: F401
    import openpyxl  # noqa: F401
    import plotly.express  # noqa: F401
    import plotly.graph_objects  # noqa: F401
    import pyarrow.parquet  # noqa: F401
    import statsmodels.tsa.holtwinters  # noqa: F401
    import xlrd  # noqa: F401
    import yaml  # noqa: F401

    import src.analytics  # noqa: F401
    import src.cold_start  # noqa: F401
    import src.exports  # noqa: F401
    import src.partition  # noqa: F401
    import src.pipeline  # noqa: F401
    import src.site_files  # noqa: F401
    import src.workspace  # noqa: F401


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
