"""can run/switch statistics identify the information channel?  

Four data-generating processes, each with a known true pre-trade midpoint m_true, pushed through the *same* proxy construction
and the same statistics as the empirical pipeline (switch-pair midpoint proxy, info/eff/RS/PI_100/over_100, drift share at
direction switches, within/between split):

  S1  order-flow learning   private signal only in takers (prob alpha, higher for large orders); maker updates quotes by Bayes on D
                            with an assumed informed share alpha_hat (= alpha: correct; 2 alpha: over-reaction; alpha/2: under-reaction)
  S2  public information    maker's midpoint follows public signals toward Y; takers uninformed (persistent Markov directions)
  S3  inventory             no outcome information at all; maker shifts the midpoint with inventory pressure (mean-reverting)
  S4  mixed                 public signals + informed takers + inventory

Common features: exogenous half-spread per market, direction persistence rho, lognormal order sizes, exponential inter-arrival
times with market-specific intensity, trades execute at the quotes (price = m_true + D * half-spread).

Outputs: results/sim_runs.csv (one row per scenario x parameter set x seed), results/sim_summary.csv (scenario means),
         results/sim_classifier.csv (held-out misclassification when using the statistics to label the mechanism)
"""
from study_common import *
import duckdb, itertools

N_MKT = 150 if SMOKE else 2000
SEEDS = [0] if SMOKE else [0, 1, 2]
sig = lambda x: 1 / (1 + np.exp(-x)); lgt = lambda p: np.log(p / (1 - p))

def simulate(scn, P, N, seed):
    rng = np.random.default_rng(seed)
    T = np.clip(np.exp(rng.normal(np.log(400), 0.5, N)).astype(int), 150, 1500); Tmax = T.max()
    v0 = rng.uniform(0.25, 0.75, N); Y = (rng.random(N) < v0).astype(int)
    s = rng.uniform(0.005, 0.03, N)                                     # half-spread 50-300 bp
    lam = np.exp(rng.normal(np.log(30), 0.8, N))                        # mean inter-arrival seconds
    rho = P.get("rho", 0.6); a0 = P.get("alpha0", 0.0); a1 = P.get("alpha1", 0.0); ah = P.get("alpha_hat_ratio", 1.0)
    mu = P.get("mu", 0.0); pi = P.get("pi", 0.0); gam = P.get("gamma", 0.0); eta = P.get("eta", 0.0)
    ell = lgt(v0).copy(); inv = np.zeros(N); Dprev = rng.choice([-1, 1], N); t_clock = np.zeros(N)
    out = dict(g=[], rn=[], ts=[], D=[], p=[], usd=[], Y=[], m_true=[])
    for t in range(Tmax):
        act = t < T
        size = np.exp(rng.normal(np.log(30), 1.5, N))
        alpha = a0 + a1 * (size >= 1000)
        informed = rng.random(N) < alpha
        Dunf = np.where(rng.random(N) < rho, Dprev, -Dprev)
        D = np.where(informed, 2 * Y - 1, Dunf)
        m = np.clip(sig(ell + gam * inv), 0.01, 0.99)
        p = np.clip(m + D * s, 0.005, 0.995)
        t_clock = t_clock + rng.exponential(lam)
        for k, v in (("g", np.arange(N)), ("rn", np.full(N, t + 1)), ("ts", t_clock.copy()), ("D", D), ("p", p), ("usd", size), ("Y", Y), ("m_true", m)):
            out[k].append(v[act])
        # --- quote updates after the trade
        if a0 > 0 or a1 > 0:                                            # Bayes on direction (and size, through alpha_hat(size))
            ahat = np.clip(ah * alpha, 0, 0.99)
            lr_buy = np.log((ahat + (1 - ahat) * 0.5) / ((1 - ahat) * 0.5))
            ell = ell + D * lr_buy
        if pi > 0:                                                      # public signal with probability pi per trade
            hit = rng.random(N) < pi
            x = rng.normal((2 * Y - 1) * mu, 1.0, N)
            ell = ell + hit * 2 * mu * x
        if gam > 0:                                                     # inventory: maker short after a buy -> raises the midpoint
            inv = (1 - eta) * inv + D * size / 100.0
        Dprev = D
    df = pd.DataFrame({k: np.concatenate(v) for k, v in out.items()})
    df["condition_id"] = "sim" + df.g.astype(str); return df

PIPE = """
CREATE OR REPLACE TABLE ev AS SELECT *, lead(p,1) OVER w p_lead_1, lead(D,1) OVER w D_next, lead(m_true,100) OVER w m_true_after_100, lead(p,100) OVER w p_lead_100
    FROM sim WINDOW w AS (PARTITION BY condition_id ORDER BY rn);
ALTER TABLE ev ADD COLUMN kind VARCHAR; UPDATE ev SET kind = CASE WHEN D_next IS NULL THEN NULL WHEN D = D_next THEN 'same' ELSE 'switch' END;
CREATE OR REPLACE TABLE mids AS SELECT a.condition_id, a.rn + 1 rn_mid, b.ts ts_mid, b.ts - a.ts dt_pair, (a.p + b.p)/2.0 mid,
    CASE WHEN a.D = 1 THEN a.p ELSE b.p END ask_pair, CASE WHEN a.D = 1 THEN b.p ELSE a.p END bid_pair
    FROM ev a JOIN ev b ON a.condition_id = b.condition_id AND b.rn = a.rn + 1 WHERE a.kind = 'switch';
CREATE OR REPLACE TABLE e1 AS SELECT e.*, m.mid mid_before, e.rn - m.rn_mid stale, (m.ask_pair < m.bid_pair) crossed_before
    FROM ev e ASOF LEFT JOIN mids m ON m.condition_id = e.condition_id AND m.rn_mid <= e.rn - 1;
CREATE OR REPLACE TABLE e AS SELECT e.*, m.mid mid_after_100 FROM e1 e ASOF LEFT JOIN mids m ON m.condition_id = e.condition_id AND m.rn_mid <= e.rn + 100;
"""
STATS = f"""
WITH b AS (SELECT * FROM e WHERE {MID} AND mid_before IS NOT NULL AND mid_after_100 IS NOT NULL AND m_true_after_100 IS NOT NULL),
     c AS (SELECT *, avg(D) OVER (PARTITION BY condition_id) Dbar FROM b),
     pairs AS (SELECT kind, count(*) n, sum((2*Y-1)*(p_lead_1-p)) sum_dq, avg(D*(p_lead_1-p)) R1 FROM e WHERE {MID} AND kind IS NOT NULL GROUP BY 1)
SELECT (SELECT count(*) FROM b) n,
  (SELECT n FROM pairs WHERE kind='switch')*1.0/(SELECT sum(n) FROM pairs) switch_share,
  (SELECT sum_dq FROM pairs WHERE kind='switch')/(SELECT sum(sum_dq) FROM pairs) switch_drift_share,
  (SELECT R1 FROM pairs WHERE kind='same')*1e4 R1_same_bp, (SELECT R1 FROM pairs WHERE kind='switch')*1e4 R1_switch_bp,
  avg(D*(Y-mid_before))*1e4 info_proxy, avg(D*(p-mid_before))*1e4 eff_proxy, avg(D*(p-Y))*1e4 RS,
  avg(D*(mid_after_100-mid_before))*1e4 PI_proxy, avg(D*(mid_after_100-Y))*1e4 over_proxy,
  avg(D*(Y-m_true))*1e4 info_true, avg(D*(p-m_true))*1e4 eff_true, avg(D*(m_true_after_100-m_true))*1e4 PI_true, avg(D*(m_true_after_100-Y))*1e4 over_true,
  avg(D*(mid_before-m_true))*1e4 De_bp, avg(D*(mid_after_100-m_true_after_100))*1e4 De_after_bp,
  avg(abs(mid_before-m_true))*1e4 mae_bp, sqrt(avg(power(mid_before-m_true,2)))*1e4 rmse_bp,
  avg((D*(p-mid_before) < 0)::int) sign_violation, avg(crossed_before::int) crossed, avg((stale<=1)::int) fresh_share,
  avg(D*(Y-mid_before)) FILTER (stale<=1)*1e4 info_fresh, avg(D*(mid_after_100-Y)) FILTER (stale<=1)*1e4 over_fresh,
  avg(D*(Y-m_true)) FILTER (stale<=1)*1e4 info_true_fresh,
  avg(Dbar*(Y-mid_before))*1e4 info_between_proxy, avg((D-Dbar)*(Y-mid_before))*1e4 info_within_proxy,
  avg(Dbar*(Y-m_true))*1e4 info_between_true, avg((D-Dbar)*(Y-m_true))*1e4 info_within_true,
  avg(((2*Y-1)*D = 1)::int) s_informed_share,
  avg(D*(Y-mid_before)) FILTER (usd < 100)*1e4 info_proxy_small, avg(D*(Y-mid_before)) FILTER (usd >= 1000)*1e4 info_proxy_large,
  avg(D*(mid_after_100-Y)) FILTER (usd < 100)*1e4 over_proxy_small, avg(D*(mid_after_100-Y)) FILTER (usd >= 1000)*1e4 over_proxy_large,
  avg(D*(m_true_after_100-Y)) FILTER (usd < 100)*1e4 over_true_small, avg(D*(m_true_after_100-Y)) FILTER (usd >= 1000)*1e4 over_true_large
FROM c"""

GRID = []
for a0, ah in itertools.product([0.01, 0.03, 0.06], [0.5, 1.0, 2.0]): GRID.append(("S1 order-flow learning", dict(alpha0=a0, alpha1=0.0, alpha_hat_ratio=ah, rho=0.6)))
for rho in (0.5, 0.75): GRID.append(("S1 order-flow learning", dict(alpha0=0.03, alpha1=0.0, alpha_hat_ratio=1.0, rho=rho)))
GRID.append(("S1 order-flow learning", dict(alpha0=0.01, alpha1=0.15, alpha_hat_ratio=1.0, rho=0.6)))   # size-dependent information (Kyle flavour)
for mu, pi in itertools.product([0.05, 0.1, 0.2], [0.2, 1.0]): GRID.append(("S2 public information", dict(mu=mu, pi=pi, rho=0.6)))
for rho in (0.5, 0.75): GRID.append(("S2 public information", dict(mu=0.1, pi=0.5, rho=rho)))
for gam, eta in itertools.product([0.002, 0.01, 0.03], [0.0, 0.05]): GRID.append(("S3 inventory", dict(gamma=gam, eta=eta, rho=0.6)))
for a0, mu, ah in itertools.product([0.01, 0.03], [0.05, 0.1], [1.0, 2.0]): GRID.append(("S4 mixed", dict(alpha0=a0, alpha1=0.05, alpha_hat_ratio=ah, mu=mu, pi=0.5, gamma=0.005, eta=0.02, rho=0.6)))
if SMOKE: GRID = GRID[::4]

con = duckdb.connect(); con.execute("SET threads=16")
rows = []
for i, (scn, P) in enumerate(GRID):
    for seed in SEEDS:
        df = simulate(scn, P, N_MKT, seed=1000 * i + seed)
        con.register("sim", df); con.execute(PIPE)
        st = con.sql(STATS).df().iloc[0].to_dict()
        st.update(scenario=scn, config=i, seed=seed, n_trades_total=len(df), **{f"p_{k}": v for k, v in P.items()})
        rows.append(st)
        log(f"{scn:24s} cfg{i:02d} seed{seed} switch_share={st['switch_share']:.3f} switch_drift={st['switch_drift_share']:.3f} "
            f"info_true={st['info_true']:.0f} info_proxy={st['info_proxy']:.0f} over_true={st['over_true']:.0f} over_proxy={st['over_proxy']:.0f} De={st['De_bp']:.0f}")
runs = pd.DataFrame(rows); save(runs, "sim_runs.csv")
keys = ["switch_share", "switch_drift_share", "R1_same_bp", "R1_switch_bp", "info_true", "info_proxy", "info_fresh", "eff_true", "eff_proxy", "RS",
        "PI_true", "PI_proxy", "over_true", "over_proxy", "over_fresh", "De_bp", "De_after_bp", "mae_bp", "rmse_bp", "sign_violation", "crossed", "fresh_share",
        "info_between_true", "info_within_true", "info_between_proxy", "info_within_proxy", "s_informed_share",
        "over_proxy_small", "over_proxy_large", "over_true_small", "over_true_large"]
summ = runs.groupby("scenario")[keys].agg(["mean", "min", "max"]); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
save(summ.reset_index(), "sim_summary.csv")

# ---- classifier check: can the empirical statistics label the mechanism?  Train on seeds 0-1, test on seed 2 (same configs) and on
#      held-out configs (every third config) -- reports misclassification, not a claim of identification.
if not SMOKE:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feats = ["switch_share", "switch_drift_share", "R1_same_bp", "R1_switch_bp", "info_proxy", "eff_proxy", "RS", "over_proxy", "fresh_share", "sign_violation"]
    X = runs[feats].fillna(0).to_numpy(); y = runs.scenario.to_numpy()
    res = []
    tr = runs.seed < 2; te = ~tr
    sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te])); res.append(dict(split="held-out seed, same configs", n_test=int(te.sum()), accuracy=float((pred == y[te]).mean())))
    hold = runs.config % 3 == 0; tr = ~hold; te = hold
    sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te])); res.append(dict(split="held-out parameter configs", n_test=int(te.sum()), accuracy=float((pred == y[te]).mean())))
    cm = pd.crosstab(pd.Series(y[te], name="true"), pd.Series(pred, name="pred")); save(cm.reset_index(), "sim_classifier_confusion_heldout_configs.csv")
    # switch-drift-share alone as a rule: does S1 overlap with S2?
    for scn, g in runs.groupby("scenario"):
        res.append(dict(split=f"range of switch_drift_share in {scn}", n_test=len(g), lo=g.switch_drift_share.min(), hi=g.switch_drift_share.max(), mean=g.switch_drift_share.mean()))
    save(pd.DataFrame(res), "sim_classifier.csv")
manifest("04_simulation", n_runs=len(runs))
log("done")
