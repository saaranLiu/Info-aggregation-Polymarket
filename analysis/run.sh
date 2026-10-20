#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-outputs}"
if [[ "$MODE" == "check" ]]; then
  "$PYTHON_BIN" analysis/build_samples.py --check
  "$PYTHON_BIN" analysis/sample_demo.py
  "$PYTHON_BIN" analysis/validate.py
  exit 0
fi
if [[ "$MODE" == "presentation" ]]; then
  for stage in 11_outputs 18_r2_outputs 28_v3_tables 35_external_tables 10_figures 19_r2_figures 29_editorial_outputs 36_cross_platform; do
    "$PYTHON_BIN" "analysis/$stage.py"
  done
  if [[ -f paper/build.sh ]]; then
    bash paper/build.sh
    "$PYTHON_BIN" analysis/validate.py --layout
  else
    "$PYTHON_BIN" analysis/validate.py
  fi
  exit 0
fi
if [[ "$MODE" == "raw" ]]; then
  if [[ -z "${DATA_ROOT:-}" || ! -d "$DATA_ROOT/database/daily_aligned" ]]; then
    echo 'Set DATA_ROOT to the archive root containing database/daily_aligned, OrderFilled, and CTF.' >&2
    exit 2
  fi
  for stage in 00_markets 01_trades 02_events; do
    "$PYTHON_BIN" "analysis/base/$stage.py"
  done
  "$PYTHON_BIN" analysis/base/05_quote_channel.py
  cp work/base/results/quote_YesNo_pairs_mid.csv results/descriptive/
  cp work/base/results/quote_YesNo_driftshare_mid.csv results/descriptive/
  for stage in 00_build 01_audit 02_proxy 03_within_between 04_simulation 05_inference 05b_horizon_samples 06_market_level 07_returns 08_anchor 09_accounts; do
    "$PYTHON_BIN" "analysis/$stage.py"
  done
fi
if [[ "$MODE" == "raw" || "$MODE" == "events" ]]; then
  # Event-level analyses: need work/events/mid_*.parquet; 14 also needs DATA_ROOT for fill-level quantities.
  # Observed quotes: 13a downloads selected row groups of the pmxt archive (network, several hours); skip with SKIP_QUOTES=1.
  if [[ "${SKIP_QUOTES:-0}" != "1" ]]; then
    "$PYTHON_BIN" analysis/13a_fetch_quotes.py
  fi
  if compgen -G "work/quotes/raw/date=*.parquet" > /dev/null; then
    "$PYTHON_BIN" analysis/13b_quote_validation.py
  fi
  for stage in 14_fill_prices 15_clock_horizon 16_wb_size 17_repro_materials 20_residual_controls \
               21_v3_direction 22_v3_contrasts 23_v3_retention_concentration 24_v3_spline; do
    "$PYTHON_BIN" "analysis/$stage.py"
  done
elif [[ "$MODE" != "outputs" ]]; then
  echo 'Usage: bash analysis/run.sh [check|presentation|outputs|events|raw]' >&2
  exit 2
fi
"$PYTHON_BIN" analysis/11_outputs.py
"$PYTHON_BIN" analysis/12_economic_extensions.py
"$PYTHON_BIN" analysis/10_figures.py
"$PYTHON_BIN" analysis/18_r2_outputs.py
"$PYTHON_BIN" analysis/19_r2_figures.py
"$PYTHON_BIN" analysis/28_v3_tables.py
"$PYTHON_BIN" analysis/35_external_tables.py
"$PYTHON_BIN" analysis/29_editorial_outputs.py
"$PYTHON_BIN" analysis/36_cross_platform.py
if [[ -f paper/build.sh ]]; then
  bash paper/build.sh
  "$PYTHON_BIN" analysis/validate.py --layout
else
  "$PYTHON_BIN" analysis/validate.py
fi
