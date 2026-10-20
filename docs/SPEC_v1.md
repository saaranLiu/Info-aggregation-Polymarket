# Experiment Specification v1 — corrected definitions and full-population protocol

This document fixes the measurement framework (replacing the proposal's IIE / OutcomeAlignedResponse) and defines exactly what the pipeline in this folder computes. Everything below is implemented in `common.py` and the numbered stage scripts.

## 1. Why the proposal's metric is invalid

Proposal: `OAR_h = (2Y−1)·D_i·(P_{i+h} − P_i)`, claimed `>0 ⇔ trade pushed price toward the outcome`.

Counter-example: Y=0 (No wins), D=−1 (taker sells Yes — correct direction), price falls ΔP<0 (toward outcome) ⇒ OAR = (−1)(−1)(ΔP) = ΔP < 0.

Algebra: let `I = (2Y−1)·D` (+1 iff the taker bought the eventual winner). Then `OAR = I·ΔP`, i.e. "ex-post-correct indicator × raw change on the arbitrary Yes/No axis". Its sign flips if the market's reference leg is relabelled.

More fundamentally, in the triple (D, Y, ΔP), conditioning on Y makes I and D the same variable. Hence for any stratum there are exactly **three identifiable primitives** per horizon h; every "conditional" or "interaction" quantity is a rearrangement of them:

| primitive | definition | meaning | invariances |
|---|---|---|---|
| **informed share** `s` | `P(I = +1)` (count- and USD-weighted); equivalently **taker edge** `e = E[D·(Y − p)]` | does aggressive flow point toward the outcome? | unchanged under relabelling; → 0.5 / 0 under Y-shuffle |
| **signed response** `R_h` | `E[D·(P_{i+h} − P_i)] = E[I·Δq_h]` | does price follow the trade? | Y-free; → 0 under D-shuffle |
| **drift** `μ_h` | `E[Δq_h]`, `Δq_h = (2Y−1)(P_{i+h} − P_i)` | movement toward the outcome irrespective of trades | D-free; → 0 under Y-shuffle |

where `q = Y·p + (1−Y)(1−p)` is the price of the eventual winner.

Derived (reported but not independent): `E[Δq_h | I=+1] = μ_h + (1−s)·(E[R_h|+] + E[R_h|−])`, etc. The proposal's "genuine aggregation vs mechanical convergence" distinction maps onto: `R_h` and `s` (flow-driven) vs `μ_h` (flow-independent). There is no fourth quantity called "information incorporation efficiency"; the paper's construct is the **triple (s, R_h, μ_h)** and its decomposition by trade size, time-to-resolution, market state, and participant type.

## 2. Units and sign conventions

* Trade = all fills of one `(condition_id, block_timestamp, taker, D)`; `p` = USD-VWAP of `p_event`; `usd` = Σ `usdc_amount`.
* All price changes are on the **winner axis** `q` unless the quantity is Y-free (R_h is identical on either axis).
* Horizons: trade-count `h ∈ {1,2,5,10,20,50,100}`; clock `H` (group-specific, see `common.py`); pre-trade lags `{1,5,20,100}` (`pre_h = I·(q_i − q_{i−h})`).
* Log-odds variant: `R^LO_h = D·(logit P_{i+h} − logit P_i)` with P clipped to [0.005, 0.995].
* Half-spread-normalised variant: `R_h / (Roll_m / 2)`, `Roll_m = 2·sqrt(max(−Cov(ΔP_1, ΔP_1 lag), 0))` per market.
* Time-to-resolution `τ = T_res − t` from on-chain `ConditionResolution` (block → timestamp via OrderFilled block map, 2.1 s/block interpolation, capped at 1e5 blocks). Normalised lifecycle `θ = τ / (T_res − t_first)`.
* Strata: `|p − 0.5|` layers `[0,.1) [.1,.2) [.2,.3) [.3,.4) [.4,.48) [.48,.5]`; main results at `|p−0.5| < 0.3` and all prices. τ coarse `{<1h, 1–24h, 1–7d, 7–30d, >30d}`; τ fine adds `{<10m, 10m–1h, 1–6h, 6–24h, 1–3d, 3–7d}`. Size `{<$10, $10–100, $100–1k, $1k–10k, >$10k}`.

## 3. Population

All resolved markets in `daily_aligned/` whose metadata winner matches the CTF payout vector, grouped by mechanism: `YesNo`, `SportsML` (team-vs-team moneyline), `OverUnder`, `UpDown` (crypto). No sampling. Main analysis filter `n_trades ≥ 100`; robustness `≥ 50`, `≥ 500`. UpDown uses clock horizons only (89 % of trades share a block).

## 4. Inference

* Trade-level cells: pooled point estimate; **market-cluster bootstrap** (B = 1000, seed 0) for SE and 95 % CI; also market-equal-weight mean.
* Permutation tests report `observed − shuffled` with the shuffled distribution from 100 seeded replicates (D-shuffle within market; Y-shuffle across markets; pseudo-resolution times).
* Market-level regressions: HC1, category × year fixed effects where stated.

## 5. Stage map

| stage | script | outputs |
|---|---|---|
| 00 | `00_markets.py` | `results/markets.parquet` (all markets, Y, res_ts, mtype) |
| 01 | `01_trades.py` | `work/trades_{grp}.parquet`, `results/maker_market_stats_{grp}.parquet` |
| 02 | `02_events.py` | `work/ev_{grp}.parquet`, `results/market_features_{grp}.parquet` |
| 03 | `03_core.py` | E1–E3: response paths, informed share, edge, size × τ, layers, categories, years, liquidity, normalised variants (`results/core_*.csv`) |
| 04 | `04_falsification.py` | E10: D-shuffle, Y-shuffle, pseudo-resolution, block-level, trimming, thresholds, log-odds, fee/self-trade exclusions |
| 05 | `05_quote_channel.py` | E4: same-side vs switch decomposition, midpoint proxy, flow → midpoint regression |
| 06 | `06_maker_side.py` | E7: maker edge, concentration, maker requote direction |
| 07 | `07_large_late.py` | E8: large-late taker accounts, recurrence, account-level informedness |
| 08 | `08_market_level.py` | E5/E6: flow imbalance → Y, Brier ~ market state, participation dynamics |
| 09 | `09_updown.py` | E9: UpDown observed vs shuffled, market-level net flow → Y |
| 10 | `10_figures.py` | figures from `results/` |
