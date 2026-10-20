# Result tables and definitions

The CSV files are saved numerical summaries for the current paper. Raw estimates are not silently replaced by values from different samples. Rebuilding paper tables and tests from summaries does not rerun the underlying database analysis.

## Canonical paper outputs

- `paper_estimates.csv`: five metrics, aligned sample labels, estimates in probability bp, market/event SEs, observation counts and horizon policy. `main` is the full main sample; `observed_h100` requires 100 subsequent trades.
- `paper_equivalence.csv`: TOST at fixed 50/25 probability-bp margins, observed-horizon sample only.
- `paper_headline_tests.csv`: one family of 21 explicitly enumerated main and observed-horizon tests, with BH q-values. This is an analysis choice, not a documented preregistration.
- `paper_cross_topic_differences.csv`: exploratory Politics-minus-Crypto comparisons, assuming independent category-by-day clusters. Within-topic size covariance is unavailable from summaries, so no within-topic difference significance is inferred.
- `VALIDATION.json`: reproducible arithmetic checks and selected source values.
- `SOURCES.json`: source-table hashes and sample conventions.

## Source families

| Prefix | Contents | Units/sample notes |
|---|---|---|
| `audit_A` | sample flow | counts, USD, clusters |
| `audit_B` | trade merge and fill diagnostics | source-specific populations; `est` is unscaled |
| `audit_C_weights` | weighting comparisons | common observed-horizon sample; `est/lo/hi` in probability units |
| `audit_D` | horizon coverage, selection and metrics | `h100` rows requiring a future trade differ from capped post-proxy rows; consult the generating expressions |
| `audit_E/F/G` | terminal proxy, clocks, filters | specifications vary sample; post-proxy metrics can use capped endpoints |
| `proxy_A/B/C/D` | internal proxy checks, constructions, momentum, bounds | main sample or selected subsets; `over_100` is capped when a future trade is missing |
| `wb_A/B/E` | decomposition and serial structure | main sample; within term is outcome-invariant |
| `wb_D_oos` | event-grouped cross-validation | main sample; post-price specifications use capped endpoints, not uniform observed horizons |
| `inf_A_clusters` | main-sample clustering diagnostics | probability bp; post-proxy metrics use capped endpoints |
| `inf_A_horizon_samples` | main versus observed-horizon summary | source for canonical estimates; `all` uses capped post-proxy metrics, `horizon` is observed |
| `inf_B` | outcome and conditional run-block permutations | probability bp; 200 permutations; fixed proxy paths for run-block tests |
| `sim_` | simulated quote processes | most price fields already in bp; classifier held-out-seed and held-out-config rows are distinct |
| `mkt_A/B` | window-level USD imbalance and outcome regressions | windows end 24 h before resolution; coefficients are logit coefficients, not probability bp |
| `ret_A/B/C/D` | margins, fees, size and competition | `shares=usd/p` is constructed; relative size uses the market mean; HHI uses maker USD shares |
| `anchor_A/B` | deterministic candidate-reference classes and regressions | inferred from slugs, not verified quote availability; capped post-proxy residuals; `coef_bp` and `se_bp` in probability bp |
| `acct_A/B` | account persistence | `mean_B`, `population_mean_B`, `se_event` are probability units: multiply by 10,000 for bp; account slopes and trade-weighted means differ |
| `robustness_matrix` | specification map | probability bp; weight rows are observed-horizon, most other rows capped; intervals are individually unadjusted |
| `descriptive/` | run/switch empirical comparison | retained inputs used by the current simulation figure; not channel identification |

`RS=D(p-Y)` is a gross settlement price margin computed from grouped prices. An `usd`-weighted mean of RS is not a portfolio return. In `ret_A_dollars`, reconstructed cash margins use constructed quantity `usd/p`, not an observed maker's total inventory or actual fill quantities. The `tau>=1h` source restriction excludes the last hour, so sums across its time buckets must not be labelled the whole lifetime.

The saved examples in `anchor_A_examples` are the three highest-fill slugs per topic/label. No independent random annotation-validation record is present. A1/A2/A3 represent candidate external references inferred from event text. A4 is a release/attention-count class, not verified consensus availability.

## Economic extensions

`joint_flow_models.csv` and `joint_flow_predictions.csv` contain newly computed common-market nested logits and held-out predictions from saved market-window observations. `joint_flow_coefficients.csv` reports a separate supplementary family of 24 coefficients with BH adjustment. All four model variants use the same observations and event folds within each segment/window. Brier intervals resample paired event-cluster loss differences.

`weight_covariance.csv` and `weight_activity_shifts.csv` recover normalized covariance terms from the common horizon summaries. `size_response_gradients.csv` reports descriptive large-minus-small component means. `account_persistence_summary.csv` joins the time and disjoint-market persistence summaries, preserving the distinction between account-weighted slopes and trade-weighted means. Details and settings are in `ECONOMIC_EXTENSIONS.md` and `ECONOMIC_EXTENSIONS.json`.

## Cross-platform presentation (2026-10-20)

`k_D_participation_comparison.csv` derives cash-bin trade shares from the Polymarket main Yes/No counts in `wb_F_size.csv` and Kalshi counts in `k_B_margin.csv`. The shares are descriptive; event samples and clocks differ.

`k_E_margin_comparison.csv` brings together main-sample Polymarket Politics/Crypto margins and Kalshi cash-bin margins, preserving their source SEs and cluster definitions. No pooled venue effect, matched-event comparison, or new midpoint-residual estimate is inferred.

Script `36_cross_platform.py` produces these displays and the two detailed appendix tables. The Kalshi small-flow and both-flow prediction results retain separate cohorts and price-only baselines from `k_C_prediction.csv`; they are not a nested comparison on common markets. All earlier source CSV values remain unchanged.
