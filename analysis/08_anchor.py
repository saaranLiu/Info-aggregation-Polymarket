"""Deterministic candidate external-reference classes from slugs and observational contrasts.
A1/A2/A3 do not verify contemporaneous quotes; A4 does not verify a consensus.
The example output contains the top three slugs by fill count, not a random annotation audit.
Residuals here retain capped endpoints on the main sample.
"""
from study_common import *
import re, statsmodels.api as sm
FP = C; del C
con = connect()
MENTION = r"-say-|-says-|-said-|-mention|-tweet|number-of-tweets|-post-|-views|-wear-"          # speech/mention/social-count props: never anchored
NOT_PRICE = r"subscribers|followers|views|users|downloads|streams|volume|tvl|mindshare|market-cap|fdv|holders|committed|sales|visitors|signups|members|points|goals|yards|kills|rating|score|cases|deaths|votes|seats|percent|-pct|degrees|inches|million-|billion-|-tweets"
RULES = [
    ("A1", r"-btts$|-spread-|-total-|-over-under|-o/u-|^(nba|nfl|mlb|nhl|ncaa|ucl|uel|epl|lal|bun|sea|lig|elc|mls|wnba|atp|wta|ufc|pga|f1|lol|cs2|csgo|dota|val|cod|r6)-"),
    ("A1", r"^[a-z]{2,4}-[a-z0-9]{2,5}-[a-z0-9]{2,5}-\d{4}-\d{2}-\d{2}"),
    ("A1", r"win-the-.*(cup|series|finals|championship|super-bowl|title|mvp|stanley|masters|open|grand-prix|tournament|conference|division|pennant|bowl|league|derby|slam|medal)|make-the-(nba-|nfl-|nhl-|mlb-)?playoffs|-to-win-|game[0-9]|map[0-9]|-vs-|-match-|first-blood|quadra|pentakill|-ko-|fight-night|^ufc|boxing|-finish-in-the-.*top|qualify-for"),
    ("A2", r"(bitcoin|btc|ethereum|eth|solana|sol|xrp|ripple|dogecoin|doge|bnb|cardano|ada|hyperliquid|hype|avax|link|chainlink|ton|sui|pepe|shib|ltc|litecoin|trump-coin|usdc|usdt|tether)-(above|below|dip|reach|hit|close|price|all-time-high|ath|under|over|up-or-down|higher|lower)"),
    ("A2", r"-(above|below)-\d|-dip-to-\d|-dip-below-\d|-reach-\$?\d|-hit-(low|high)-\d|-hit-\$?\d|-close-above-|-close-below-|all-time-high|-ath-"),
    ("A2", r"^(nvda|googl|goog|tsla|aapl|msft|amzn|meta|nflx|amd|intc|spx|spy|qqq|nasdaq|s-p-500|sp-500|dow|gold|silver|oil|wti|brent|ng|natural-gas|eur|usd|jpy|gbp|dxy|us-10y|10-year|treasury|vix)-"),
    ("A2", r"will-(gold|silver|oil|wti|crude|the-s-p|spx|nasdaq|nvidia|tesla|apple|microsoft|amazon|alphabet|google|meta|nvda|tsla|aapl|msft|amzn|googl)-(reach|dip|hit|close|be-above|be-below|trade)"),
    ("A3", r"presidential-election|presidential-nominee|-election-winner|win-the-.*election|popular-vote|electoral-college|-primary-|-primaries|-nominee|-nomination|-caucus|governor-(race|election)|for-governor|senate-(race|election|seat|control)|house-(race|election|control|seat)|mayor(al)?-(race|election)|-runoff|-referendum|next-chancellor|next-prime-minister|next-pope|papal|-win-(michigan|pennsylvania|wisconsin|georgia|arizona|nevada|north-carolina|florida|texas|ohio|virginia|minnesota|new-hampshire|iowa)|balance-of-power|(trump|harris|biden|kamala|vance|walz|desantis|haley|newsom|rubio)-win|win-the-(2024|2026|2028)|-votes-|electoral"),
    ("A4", r"quarterly-earnings|-earnings-|beat-earnings|-eps-|-cpi-|inflation|fed-rate|fomc|rate-cut|rate-hike|jobs-report|nonfarm|unemployment|gdp-|box-office|opening-weekend|gross-|billboard|spotify|rotten-tomatoes|-chart-|-rank-|most-searched|subscribers|followers|-views|day-one|day-1"),
]
def annotate(slug):
    if not isinstance(slug, str): return "A0"
    if re.search(MENTION, slug): return "A4" if re.search(r"-views|number-of-tweets|tweets-\d|-\d+-times", slug) else "A0"
    for lab, pat in RULES:
        if re.search(pat, slug):
            if lab == "A2" and re.search(NOT_PRICE, slug): continue
            return lab
    return "A0"
LABEL = {"A0": "A0 none identified", "A1": "A1 same-event odds (sportsbook)", "A2": "A2 related asset price", "A3": "A3 same-event odds (other venues)", "A4": "A4 scheduled release, consensus only"}

mk = con.sql(f"SELECT condition_id, cat, slug, vol, n_fills, Y, res_ts, t_first FROM '{FRES}/markets.parquet' WHERE eligible AND grp='YesNo'").df()
mk["anchor"] = mk.slug.map(annotate); mk["anchor_label"] = mk.anchor.map(LABEL)
save(mk[["condition_id", "cat", "slug", "anchor", "anchor_label", "n_fills", "vol"]], "anchor_A_annotation_markets.csv")
mid_table(con, "YesNo"); con.register("ann", mk[["condition_id", "anchor", "anchor_label"]])
con.execute(f"CREATE OR REPLACE TABLE ea AS SELECT e.*, a.anchor, a.anchor_label FROM e JOIN ann a USING(condition_id) WHERE {LATE} AND {BASE}")
cnt = con.sql("""SELECT cat, anchor_label, count(distinct condition_id) n_markets, count(*) n_trades_lateday, sum(usd) usd_lateday,
                 avg(D*(p-mid_before))*1e4 eff_bp, avg(D*(Y-mid_before))*1e4 info_bp, avg(-edge)*1e4 RS_bp, avg(D*(mid_after_100-Y))*1e4 over100_bp
                 FROM ea GROUP BY 1,2 ORDER BY 1,2""").df()
save(cnt, "anchor_A_counts.csv")
# a few examples per label for the audit appendix
ex = mk.groupby(["cat", "anchor"]).apply(lambda g: g.nlargest(3, "n_fills").slug.tolist(), include_groups=False).reset_index().rename(columns={0: "examples_top3_by_fills"})
save(ex, "anchor_A_examples.csv")

# ---- B regressions
d = con.sql("""SELECT condition_id, ev_cluster, cat, anchor, size3, D::DOUBLE D, Y::DOUBLE Y, p, mid_before m, mid_after_100 ma, edge, usd, mkt_vol, mn, (res_ts - t_first)/86400.0 life_d,
                      strftime(date_trunc('quarter', to_timestamp(ts)), '%Y-%m') qtr FROM ea""").df()
d["eff"] = d.D * (d.p - d.m); d["info"] = d.D * (d.Y - d.m); d["RS"] = -d.edge; d["over_100"] = d.D * (d.ma - d.Y)
d["mdec"] = pd.qcut(d.m.rank(method="first"), 10, labels=False); d["lvol"] = np.log(d.mkt_vol); d["ln"] = np.log(d.mn); d["llife"] = np.log(d.life_d.clip(lower=0.01))
def fit(dd, y, ref, extended, by=None):
    X = [pd.get_dummies(dd.anchor, prefix="anc", dtype=float).drop(columns=[f"anc_{ref}"], errors="ignore"), pd.get_dummies(dd.mdec, prefix="m", drop_first=True, dtype=float), pd.get_dummies(dd.qtr, prefix="q", drop_first=True, dtype=float)]
    if extended: X += [dd[["lvol", "ln", "llife"]], pd.get_dummies(dd.cat, prefix="cat", drop_first=True, dtype=float)]
    X = pd.concat(X, axis=1); X.insert(0, "const", 1.0); X = X.loc[:, (X != 0).any()]
    yv = dd[y].to_numpy(); ok = ~np.isnan(yv)
    r = sm.OLS(yv[ok], X.to_numpy()[ok]).fit(cov_type="cluster", cov_kwds={"groups": dd.ev_cluster.astype("category").cat.codes.to_numpy()[ok]})
    return {c: (r.params[i], r.bse[i]) for i, c in enumerate(X.columns) if c.startswith("anc_")}, int(ok.sum())
rows = []
for y in ["eff", "info", "RS", "over_100"]:
    for extended in (False, True):
        coefs, n = fit(d, y, "A0", extended)
        for c, (b, se) in coefs.items(): rows.append(dict(sample="all Yes/No", metric=y, spec="extended (+volume, trades, life, topic FE)" if extended else "basic (midpoint deciles, quarter)", contrast=f"{c[4:]} - A0", coef_bp=1e4 * b, se_bp=1e4 * se, n=n))
    for cat, alt in [("Politics", "A3"), ("Crypto", "A2"), ("Sports", "A1"), ("Finance", "A2")]:
        dd = d[(d.cat == cat) & d.anchor.isin([alt, "A0"])]
        if dd.anchor.nunique() < 2 or (dd.anchor == alt).sum() < 2000 or (dd.anchor == "A0").sum() < 2000: continue
        for extended in (False, True):
            coefs, n = fit(dd, y, "A0", extended)
            for c, (b, se) in coefs.items(): rows.append(dict(sample=f"within {cat}", metric=y, spec="extended" if extended else "basic", contrast=f"{c[4:]} - A0", coef_bp=1e4 * b, se_bp=1e4 * se, n=n,
                                                                 n_alt_markets=dd[dd.anchor == alt].condition_id.nunique(), n_ref_markets=dd[dd.anchor == "A0"].condition_id.nunique()))
save(pd.DataFrame(rows), "anchor_B_regressions.csv")
rows = []
for s in sorted(d.size3.unique()):
    dd = d[d.size3 == s]
    for extended in (False, True):
        coefs, n = fit(dd, "over_100", "A0", extended)
        for c, (b, se) in coefs.items(): rows.append(dict(size3=s, spec="extended" if extended else "basic", contrast=f"{c[4:]} - A0", coef_bp=1e4 * b, se_bp=1e4 * se, n=n))
save(pd.DataFrame(rows), "anchor_B_bysize.csv")
manifest("08_anchor", n_markets=len(mk))
log("done")
