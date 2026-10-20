# Information aggregation in prediction markets

Analysis scripts and data for studying order size, settlement margins, and conditional pricing in online prediction markets. The Polymarket analysis is accompanied by a separate Kalshi sample.

This repository contains the analysis code, stratified data samples, and the saved result tables those checks read. It does not include a manuscript, and it does not include the full trade or quote archives.

## Contents

| Path | Contents |
|---|---|
| [`analysis/`](analysis/) | Sample construction, decomposition, clustered inference, direction and clock checks, prediction, and cross-platform comparison |
| [`data/samples/`](data/samples/) | Six sample files: 1,483 Yes/No markets, 200 sports moneyline markets and their windows, plus saved predictions, settlement estimates, and Kalshi summaries |
| [`data/README.md`](data/README.md) | Sample selection, fields, units, sources, and limits of use |
| [`results/`](results/) | Saved numerical summaries, market-level windows, and common-sample model predictions. Field notes are in [`results/README.md`](results/README.md) |

## Check

Python 3.10 or newer is required. The commands below use only the standard library and have been run with Python 3.12.

```sh
git clone <repository-url>
cd Info-aggregation-Polymarket
bash analysis/run.sh check
```

The check:

1. Matches the six sample files to their source tables and hashes.
2. Checks the accounting identities for settlement margins and subsequent residuals.
3. Checks event folds in the saved predictions and computes demonstration Brier gains.
4. Checks the saved estimates, tests, and common-market models.

Demonstration output is written to the ignored path `work/sample_demo/summary.json`. Markets are drawn by topic and final outcome. **Scores on this sample are not population estimates.** The Kalshi files are saved inference summaries, not raw trades.

Rebuild the samples from the saved result tables:

```sh
python3 analysis/build_samples.py
```

## Refit the joint models from saved market windows

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
bash analysis/run.sh outputs
```

`analysis/12_economic_extensions.py` reestimates four common-sample models and the event-grouped cross-validation from the market-level windows in this repository. The other stages read the saved summaries they need. This mode does not reestimate every trade-level result.

## Reestimate from the raw trade archive

```sh
DATA_ROOT=/path/to/archive bash analysis/run.sh raw
```

`DATA_ROOT` must contain `database/daily_aligned/`, `database/OrderFilled/`, and `database/CTF/`. Intermediate files are written under `work/base/` and `work/events/`. `ANALYSIS_THREADS` and `ANALYSIS_MEMORY` can be set for the full archive.

Place external inputs in `external/d1`, `external/d4`, `external/d5`, and `external/d6`, then run scripts 31–34. `MARKETS_PARQUET` overrides the market-metadata path; the default is `work/base/results/markets.parquet`. `DA_DIR` overrides the daily trade directory; otherwise it is derived from `DATA_ROOT`. `HF_ENDPOINT` sets the quote-download endpoint.

## Measurement

- Polymarket main sample: 1–24 hours before on-chain resolution, an interior price, at least 100 trades in the market, and a defined pre-trade proxy midpoint.
- Outcome-token share weights and dollar weights are kept separate. A settlement margin is not an account profit.
- Clock horizons use resolution, the scheduled close, one hour before that close, and a retrospective convergence cutoff.
- Kalshi uses the margin between the execution price and the final value during the 1–24 hours before exchange close, with both absolute cash bins and within-market cash ranks.
- Both venues show small-order participation and a positive small-order margin. Size gradients and forecast gains depend on the size definition and the included markets. The venues are not matched event by event, so a difference is not a platform effect.

Units and sample differences are documented in [`results/README.md`](results/README.md). Sources and third-party terms are documented in [`data/README.md`](data/README.md). This repository does not grant a new license for third-party data.
