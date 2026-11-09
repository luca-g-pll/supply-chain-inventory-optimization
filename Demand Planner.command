#!/bin/bash
# macOS: double-click to start Demand & Inventory Planner (first start installs what is needed).
cd "$(dirname "$0")" || exit 1
PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11 or newer is needed. Install it from https://www.python.org/downloads/ and double-click again."
  read -r -p "Press Enter to close."
  exit 1
fi
"$PY" launcher.py "$@"
status=$?
if [ $status -ne 0 ]; then read -r -p "Press Enter to close."; fi
exit $status
