"""Reproduction materials read directly from the source of 04_simulation.py and 08_anchor.py, so that the published
parameter grid and classification rules cannot drift from the code that produced the results.

Outputs: results/repro_sim_grid.csv (one row per configuration), results/repro_sim_common.csv (settings shared by all
configurations), results/repro_anchor_rules.csv (ordered rules with full patterns and market counts),
paper/tables/sim_grid.tex, paper/tables/anchor_rules.tex
"""
from study_common import *
import itertools, re
from pathlib import Path

HERE = Path(__file__).resolve().parent; PAPER = Path(PROJECT) / "paper"

# ---- simulation grid
ns = {"itertools": itertools}
txt = (HERE / "04_simulation.py").read_text()
exec(txt[txt.index("GRID = []"):txt.index("if SMOKE: GRID")], ns)
rows = []
for i, (scn, P) in enumerate(ns["GRID"]):
    r = dict(config=i, scenario=scn, rho=P.get("rho", 0.6), alpha0=P.get("alpha0", 0.0), alpha1=P.get("alpha1", 0.0),
             alpha_hat_ratio=P.get("alpha_hat_ratio", 1.0) if ("alpha0" in P) else np.nan, mu=P.get("mu", 0.0), pi=P.get("pi", 0.0),
             gamma=P.get("gamma", 0.0), eta=P.get("eta", 0.0))
    rows.append(r)
grid = pd.DataFrame(rows); save(grid, "repro_sim_grid.csv")
assert len(grid) == 34
common = pd.DataFrame([
    ("markets per configuration and seed", "2000"), ("seeds", "3 (seed = 1000 x config + {0,1,2})"),
    ("trades per market T", "lognormal(log 400, 0.5), clipped to [150, 1500]"),
    ("prior v0 and outcome", "v0 ~ U(0.25, 0.75); Y ~ Bernoulli(v0)"), ("half-spread s", "U(0.005, 0.03) per market"),
    ("order size (USD)", "lognormal(log 30, 1.5)"), ("inter-arrival (s)", "exponential, market mean lognormal(log 30, 0.8)"),
    ("execution", "p = clip(m_true + D s, 0.005, 0.995); m_true clipped to [0.01, 0.99]"),
    ("uninformed direction", "D_t = D_{t-1} with probability rho"),
    ("informed taker", "probability alpha0 + alpha1 1{size >= 1000}; D = 2Y - 1"),
    ("maker learning", "logit update D log((ah + (1-ah)/2)/((1-ah)/2)), ah = alpha_hat_ratio x alpha"),
    ("public signal", "with probability pi: x ~ N((2Y-1) mu, 1); logit += 2 mu x"),
    ("inventory", "inv = (1-eta) inv + D size/100; m = sigmoid(logit + gamma inv)"),
    ("statistics", "same switch-pair proxy, |p-0.5|<0.3 filter, 100-trade horizon as the empirical pipeline")],
    columns=["setting", "value"])
save(common, "repro_sim_common.csv")

def rng_str(v):
    u = sorted(set(round(x, 4) for x in v if pd.notna(x)))
    return ", ".join(f"{x:g}" for x in u)
tex = []
for scn, g in grid.groupby("scenario", sort=False):
    free = {"S1 order-flow learning": ["alpha0", "alpha_hat_ratio", "rho", "alpha1"], "S2 public information": ["mu", "pi", "rho"],
            "S3 inventory": ["gamma", "eta"], "S4 mixed": ["alpha0", "mu", "alpha_hat_ratio"]}[scn]
    sym = {"alpha0": r"$\alpha_0$", "alpha1": r"$\alpha_1$", "alpha_hat_ratio": r"$\hat\alpha/\alpha$", "rho": r"$\rho$", "mu": r"$\mu$",
           "pi": r"$\pi$", "gamma": r"$\gamma$", "eta": r"$\eta$"}
    vary = "; ".join(f"{sym[k]} $\\in\\{{{rng_str(g[k])}\\}}$" for k in free)
    fixed = {"S1 order-flow learning": r"$\rho{=}0.6$ unless varied", "S2 public information": r"$\rho{=}0.6$ unless varied",
             "S3 inventory": r"$\rho{=}0.6$", "S4 mixed": r"$\alpha_1{=}0.05$, $\pi{=}0.5$, $\gamma{=}0.005$, $\eta{=}0.02$, $\rho{=}0.6$"}[scn]
    tex.append(f"{scn.split()[0]} & {len(g)} & {vary} & {fixed}" + r" \\")
(PAPER / "tables/sim_grid.tex").write_text("\n".join(tex) + "\n")

# ---- external-reference classification rules
ans = {"re": re}
txt = (HERE / "08_anchor.py").read_text(); txt = txt[txt.index("MENTION = "):]
exec(txt[:txt.index("\n", txt.index("LABEL = "))], ans)
con = connect()
slugs = con.sql(f"""SELECT m.condition_id, m.slug, m.cat, m.anchor FROM '{RES}/anchor_A_annotation_markets.csv' m
    WHERE m.condition_id IN (SELECT DISTINCT condition_id FROM '{WORK}/mid_YesNo.parquet' WHERE {LATE} AND {BASE})""").df()
first = []
for s in slugs.slug:
    hit = "A0"
    if isinstance(s, str):
        if re.search(ans["MENTION"], s): hit = "mention" if not re.search(r"-views|number-of-tweets|tweets-\d|-\d+-times", s) else "mention->A4"
        else:
            for k, (lab, pat) in enumerate(ans["RULES"]):
                if re.search(pat, s) and not (lab == "A2" and re.search(ans["NOT_PRICE"], s)): hit = f"rule{k}"; break
    first.append(hit)
slugs["first_rule"] = first; slugs["cls"] = slugs.slug.map(ans["annotate"])
assert (slugs.cls == slugs.anchor).all()
cnt = slugs.groupby("first_rule").condition_id.nunique()
rules = [dict(order=0, cls="A0 / A4", rule="mention/speech/social-count props (A4 only for count props)", pattern=ans["MENTION"],
              exclusion="", n_markets=int(cnt.get("mention", 0) + cnt.get("mention->A4", 0)))]
for k, (lab, pat) in enumerate(ans["RULES"]):
    rules.append(dict(order=k + 1, cls=lab, rule=ans["LABEL"][lab], pattern=pat, exclusion=ans["NOT_PRICE"] if lab == "A2" else "",
                      n_markets=int(cnt.get(f"rule{k}", 0))))
rules.append(dict(order=len(rules), cls="A0", rule="no rule matched", pattern="", exclusion="", n_markets=int(cnt.get("A0", 0))))
rules = pd.DataFrame(rules); save(rules, "repro_anchor_rules.csv")
assert rules.n_markets.sum() == slugs.condition_id.nunique()
by = slugs.groupby(["cls", "cat"]).condition_id.nunique().unstack(fill_value=0)
save(by.reset_index(), "repro_anchor_class_by_topic.csv")
desc = {"A1": "sports leagues, fixture slugs, spread/total/BTTS props, title and playoff props",
        "A2": "crypto, equity, index, commodity and FX tickers with above/below/reach/dip/close thresholds; count-type nouns excluded",
        "A3": "elections, primaries, nominations, referenda, runoffs, leadership succession",
        "A4": "earnings, CPI, rates, jobs, GDP, box office, charts, rankings, view counts",
        "A0": "no rule matched; speech and mention props without a count"}
tex = []
for c in ["A1", "A2", "A3", "A4", "A0"]:
    nr = int((rules.cls == c).sum()); n = int(slugs[slugs.cls == c].condition_id.nunique())
    tex.append(f"{c} & {nr} & {desc[c]} & {n:,}" + r" \\")
(PAPER / "tables/anchor_rules.tex").write_text("\n".join(tex) + "\n")
log("done")
