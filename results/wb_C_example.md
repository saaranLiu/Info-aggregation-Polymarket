# Numerical example: within-market direction-price covariance does not depend on Y

One market, four trades, proxy midpoints m and taker directions D (buys at low m, sells at high m):

| trade | D | m |
|---|---|---|
| 1 | +1 | 0.40 |
| 2 | -1 | 0.50 |
| 3 | +1 | 0.42 |
| 4 | -1 | 0.52 |

Y = 1: info per trade = [0.6, -0.5, 0.58, -0.48]; mean info = 0.0500 = between 0.0000 + within 0.0500
Y = 0: info per trade = [-0.4, 0.5, -0.42, 0.52]; mean info = 0.0500 = between -0.0000 + within 0.0500

The within term -Cov_g(D, m) = +0.05 is identical under both outcomes: it measures that buys occur at lower midpoints than sells.
It is informative about Y only if 'buying low inside a market' is itself predictive of the resolution, which the decomposition does not test;
stage D (out-of-sample prediction) is the test that does.