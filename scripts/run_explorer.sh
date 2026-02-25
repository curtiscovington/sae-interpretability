#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
streamlit run src/explorer_app.py
