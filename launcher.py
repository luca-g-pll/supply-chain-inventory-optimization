"""Start Demand & Inventory Planner with a double click.

No command to remember: this script prepares what is missing and opens the dashboard.

* first start only: creates the private Python environment (``.venv``) and installs the
  components (a few minutes, needs the internet once);
* creates the company folder (``Documents/Demand Planner``) where the data files go;
* starts the dashboard on this computer only and opens it in the browser;
* if the dashboard is already running, it just opens the browser again.

Double-click ``Demand Planner.command`` (macOS), ``Demand Planner.bat`` (Windows) or run
``python launcher.py``. Options: ``--no-browser``, ``--port N``, ``--folder PATH``, ``--setup-only``.
Only the Python standard library is used here: the script must run before anything is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STATE = VENV / "planner_launcher.json"          # which requirements are installed
RUNNING = "results/running.json"                # in the company folder: port of the dashboard started last
ENV_VAR = "PLANNER_WORKSPACE"
FOLDER_NAME = "Demand Planner"
FIRST_PORT, PORT_TRIES = 8501, 20
MIN_PYTHON = (3, 11)
START_TIMEOUT = 90


def say(text: str) -> None:
    print(f"[Demand Planner] {text}", flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def read_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_state(**changes) -> None:
    state = read_state() | changes
    try:
        STATE.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def setup() -> None:
    """Create ``.venv`` and install the components when they are missing or the requirements changed."""
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed (found {sys.version.split()[0]}). "
                         "Install it from https://www.python.org/downloads/ and start again.")
    wanted = requirements_hash()
    if venv_python().exists() and read_state().get("requirements") == wanted:
        return
    if not venv_python().exists():
        say("First start: preparing the software (a few minutes, only this time)…")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    else:
        say("Updating the components…")
    pip = [str(venv_python()), "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    subprocess.check_call([*pip, "-r", str(REQUIREMENTS)])
    fix = ROOT / "scripts" / "fix_lightgbm_macos.py"
    if sys.platform == "darwin" and fix.exists():
        subprocess.call([str(venv_python()), str(fix)])
    write_state(requirements=wanted)
    say("Ready.")


def company_folder(custom: str | None) -> Path:
    if custom:
        folder = Path(custom).expanduser()
    elif os.environ.get(ENV_VAR):
        folder = Path(os.environ[ENV_VAR]).expanduser()
    else:
        documents = Path.home() / "Documents"
        folder = (documents if documents.is_dir() else Path.home()) / FOLDER_NAME
    (folder / "data").mkdir(parents=True, exist_ok=True)
    (folder / "results").mkdir(exist_ok=True)
    return folder


def last_port(folder: Path) -> int | None:
    try:
        return int(json.loads((folder / RUNNING).read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def remember_port(folder: Path, port: int) -> None:
    try:
        (folder / RUNNING).write_text(json.dumps({"port": port}), encoding="utf-8")
    except OSError:
        pass


def is_running(port: int) -> bool:
    """True when our dashboard already answers on this port."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1.5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def free_port(first: int) -> int:
    for port in range(first, first + PORT_TRIES):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit(f"No free port between {first} and {first + PORT_TRIES - 1}: close other programs and retry.")


def start(port: int, folder: Path) -> subprocess.Popen:
    env = os.environ | {ENV_VAR: str(folder), "PYTHONUNBUFFERED": "1"}
    command = [str(venv_python()), "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
               "--server.headless", "true", "--server.address", "localhost", "--server.port", str(port),
               "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal"]
    return subprocess.Popen(command, cwd=ROOT, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start Demand & Inventory Planner on this computer.")
    parser.add_argument("--port", type=int, default=None, help=f"port of the dashboard (default: {FIRST_PORT})")
    parser.add_argument("--folder", default=None, help=f"company folder (default: Documents/{FOLDER_NAME})")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--setup-only", action="store_true", help="prepare the environment and stop")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    try:
        setup()
    except subprocess.CalledProcessError as exc:
        say(f"The installation did not finish (step failed with code {exc.returncode}). Check the internet "
            "connection and start again.")
        return 1
    if args.setup_only:
        return 0
    folder = company_folder(args.folder)
    last = last_port(folder)
    if args.port is None and last and is_running(last):
        url = f"http://localhost:{last}"
        say(f"Already running: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    port = free_port(args.port or FIRST_PORT)
    url = f"http://localhost:{port}"
    process = start(port, folder)
    remember_port(folder, port)
    deadline = time.time() + START_TIMEOUT
    while not is_running(port):
        if process.poll() is not None:
            say("The dashboard stopped while starting: read the messages above.")
            return process.returncode or 1
        if time.time() > deadline:
            say("The dashboard is taking too long to start.")
            process.terminate()
            return 1
        time.sleep(0.4)
    say(f"Dashboard: {url}")
    say(f"Company folder: {folder}")
    say("Keep this window open while you work. Close it (or press Ctrl+C) to stop the software.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        say("Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
