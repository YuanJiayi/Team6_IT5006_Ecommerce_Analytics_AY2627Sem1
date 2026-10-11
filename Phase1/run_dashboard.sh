#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [[ ! -x it5006-proj/bin/python3 ]]; then
  # A Python upgrade can leave an existing venv with a broken interpreter link.
  python3 -m venv --clear it5006-proj
fi

# Some Python installs create a venv without working pip/streamlit launcher
# scripts (bin/pip, bin/streamlit missing) even though the packages installed
# fine. Invoking via "python3 -m <tool>" sidesteps that entirely.
it5006-proj/bin/python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
it5006-proj/bin/python3 -m pip install -q -r requirements.txt
exec it5006-proj/bin/python3 -m streamlit run app.py
