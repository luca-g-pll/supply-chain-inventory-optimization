"""Make the LightGBM wheel importable on macOS machines without Homebrew's libomp.

The macOS LightGBM wheel links to ``@rpath/libomp.dylib`` and only searches the
Homebrew / MacPorts folders. scikit-learn's macOS wheel ships its own copy of
``libomp.dylib``. This script adds that folder to LightGBM's rpath list and
re-signs the library (ad hoc), so both packages share one OpenMP runtime.

Only the virtual environment is modified. The usual alternative is
``brew install libomp``. On Linux and Windows this script does nothing.

Usage:  python scripts/fix_lightgbm_macos.py
"""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from pathlib import Path


def _lightgbm_imports() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import lightgbm"], capture_output=True, text=True
    )
    return result.returncode == 0


def main() -> int:
    if platform.system() != "Darwin":
        print("Not macOS: nothing to do.")
        return 0
    if _lightgbm_imports():
        print("lightgbm already imports correctly: nothing to do.")
        return 0

    lgb_spec = importlib.util.find_spec("lightgbm")
    skl_spec = importlib.util.find_spec("sklearn")
    if lgb_spec is None or skl_spec is None:
        print("lightgbm and scikit-learn must both be installed.", file=sys.stderr)
        return 1

    lib = Path(lgb_spec.origin).parent / "lib" / "lib_lightgbm.dylib"
    omp_dir = Path(skl_spec.origin).parent / ".dylibs"
    if not (omp_dir / "libomp.dylib").exists():
        print(f"No vendored libomp.dylib in {omp_dir}; run `brew install libomp`.", file=sys.stderr)
        return 1

    rpath = "@loader_path/" + str(omp_dir.relative_to(lib.parent, walk_up=True))
    print(f"Adding rpath {rpath} to {lib}")
    subprocess.run(["install_name_tool", "-add_rpath", rpath, str(lib)], check=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(lib)], check=True)

    if _lightgbm_imports():
        print("lightgbm now imports correctly.")
        return 0
    print("lightgbm still fails to import; install libomp with Homebrew.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
