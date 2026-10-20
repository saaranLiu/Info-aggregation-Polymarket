# Economic extensions

The analysis uses the retained market-window observations and summary estimates.

## Joint-flow comparison

Four nested outcome-logit models use exactly the same markets in each topic/window: price, price plus small flow, price plus large flow, and price plus both. The pooled Yes/No baseline includes topic indicators. All models use five folds grouped by topic and resolution date. Paired cluster bootstrap intervals resample held-out loss differences. Coefficient tests form a separate, exploratory 24-test family.

| Segment | Window | Markets | Small coefficient (SE) | Large coefficient (SE) | Joint Brier gain [95% CI] |
|---|---|---:|---:|---:|---:|
| YesNo/pooled | W_paper (life before 24h) | 5755 | -0.6611 (0.2076) | 0.1294 (0.0696) | 0.000459 [-0.000156, 0.001080] |
| YesNo/pooled | W_48h_24h | 4009 | -0.4580 (0.1833) | 0.0964 (0.0744) | -0.000158 [-0.000822, 0.000496] |
| YesNo/pooled | W_7d_24h | 5239 | -0.6201 (0.1983) | 0.1507 (0.0734) | 0.000354 [-0.000321, 0.000985] |
| YesNo/Politics | W_paper (life before 24h) | 1559 | -1.3318 (0.3965) | 0.2226 (0.1331) | 0.001870 [-0.000559, 0.004149] |
| YesNo/Politics | W_48h_24h | 977 | -0.8045 (0.3688) | 0.1708 (0.1437) | 0.000955 [-0.001620, 0.003157] |
| YesNo/Politics | W_7d_24h | 1353 | -1.2583 (0.3717) | 0.1850 (0.1282) | 0.002142 [-0.000533, 0.004965] |
| YesNo/Crypto | W_paper (life before 24h) | 2825 | -0.0308 (0.2965) | 0.0641 (0.1185) | -0.000426 [-0.000796, -0.000062] |
| YesNo/Crypto | W_48h_24h | 2344 | -0.0857 (0.2653) | -0.0463 (0.1028) | -0.000597 [-0.001038, -0.000160] |
| YesNo/Crypto | W_7d_24h | 2792 | -0.0317 (0.2942) | 0.0729 (0.1211) | -0.000620 [-0.001024, -0.000230] |
| SportsML/pooled | W_paper (life before 24h) | 5376 | -0.0383 (0.0878) | -0.0124 (0.0405) | -0.000139 [-0.000285, -0.000003] |
| SportsML/pooled | W_48h_24h | 4694 | -0.0653 (0.0897) | -0.0158 (0.0397) | -0.000153 [-0.000381, 0.000071] |
| SportsML/pooled | W_7d_24h | 5328 | -0.0343 (0.0870) | -0.0192 (0.0406) | -0.000246 [-0.000414, -0.000096] |

## Weighting identities

For any positive weight w, E_w[x]-E[x]=Cov(w,x)/E[w]. The USD weighting shift of RS equals the shift of execution cost minus the shift of information. Across markets, the per-trade minus equal-market mean equals Cov_market(n_market, mean_x_market)/E_market[n_market]. The source tables identify these normalized covariances exactly on the common horizon sample.

## Account and size summaries

The account summary retains separate equal-account slopes and trade-weighted held-out means, under within-account time and disjoint-market hash splits. Size gradients are differences between the large and small bin means on the observed horizon. They are descriptive; within-topic covariance is required for inference on those differences.
