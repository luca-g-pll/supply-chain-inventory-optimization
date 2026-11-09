"""Build the standalone program: a folder with one executable, Python and every component inside.

    pip install -r requirements-build.txt
    python scripts/build_standalone.py            ->  dist/DemandPlanner/DemandPlanner(.exe)

The result runs on computers with the same operating system and processor as the one that built it
(build on Windows for Windows, on a Mac for macOS). Copy the whole ``dist/DemandPlanner`` folder,
or zip it: the user double-clicks ``DemandPlanner`` and nothing has to be installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "DemandPlanner"
# folders and files the dashboard reads from disk at run time: source -> place inside the program folder
DATA = {"app": "app", "config": "config", "sql": "sql", "data/samples": "data/samples",
        "data/templates": "data/templates", ".streamlit/config.toml": ".streamlit", "launcher.py": ".",
        "requirements.txt": "."}
COLLECT_ALL = ("streamlit", "lightgbm")          # packages with data files or native libraries of their own
METADATA = ("streamlit",)
EXCLUDE = ("jupyterlab", "notebook", "IPython", "pytest", "hypothesis", "ruff", "tkinter")


def command() -> list[str]:
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console", "--name", NAME,
            "--paths", str(ROOT), "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / "pyinstaller"),
            "--specpath", str(ROOT / "build")]
    for source, target in DATA.items():
        if (ROOT / source).exists():
            args += ["--add-data", f"{ROOT / source}{os.pathsep}{target}"]
    for package in COLLECT_ALL:
        args += ["--collect-all", package]
    for package in METADATA:
        args += ["--copy-metadata", package]
    for module in EXCLUDE:
        args += ["--exclude-module", module]
    icon = ROOT / "app" / "assets" / ("favicon.ico" if os.name == "nt" else "favicon.icns")
    if icon.exists():
        args += ["--icon", str(icon)]
    return [*args, str(ROOT / "packaging" / "planner_app.py")]


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed: pip install -r requirements-build.txt", file=sys.stderr)
        return 1
    subprocess.check_call(command(), cwd=ROOT)
    out = ROOT / "dist" / NAME
    for junk in out.rglob("__pycache__"):
        shutil.rmtree(junk, ignore_errors=True)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"\nBuilt {out} ({size:,.0f} MB). Start it with the '{NAME}' file inside.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
