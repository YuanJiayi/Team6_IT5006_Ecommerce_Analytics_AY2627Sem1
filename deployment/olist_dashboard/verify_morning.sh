#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
python3 verify_morning.py
python3 verify_analytics.py
python3 verify_ask.py
node verify_morning.js
cd "$HERE/../.."
git diff --check
