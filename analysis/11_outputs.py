"""Build the canonical paper tables and inference from saved estimates.

Uses only the Python standard library. No raw-data estimates are fabricated or
rerun here. Horizon tests use observed 100-trade horizons; carry-forward estimates
are retained solely as explicitly labelled diagnostics in their source tables.
"""
from pathlib import Path
import csv
import json
import math
from statistics import NormalDist
from manuscript_format import probability

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
TABLES.mkdir(exist_ok=True)
NORM = NormalDist()


def read(name):
    with (RES / name).open(newline="") as f:
        return list(csv.DictReader(f))


def one(rows, **filters):
    selected = [r for r in rows if all(r[k] == str(v) for k, v in filters.items())]
    if len(selected) != 1:
        raise ValueError((filters, len(selected)))
    return selected[0]


def write_csv(name, rows):
    with (RES / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pvalue(est, se):
    return math.erfc(abs(est / se) / math.sqrt(2))


def bh(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    q = 1.0
    for rank in range(len(order), 0, -1):
        i = order[rank - 1]
        q = min(q, rows[i]["p"] * len(order) / rank)
        rows[i]["q_bh"] = q


def main():
    h = read("inf_A_horizon_samples.csv")
    canonical = []
    for r in h:
        canonical.append(dict(
            grp=r["grp"], cell=r["cell"],
            sample="main" if r["sample"] == "all" else "observed_h100",
            metric=r["metric"], est_bp=float(r["est_bp"]),
            se_market_bp=float(r["se_mkt"]), se_event_bp=float(r["se_ev"]),
            n=int(r["n"]), n_event_clusters=int(r["G_ev"]),
            horizon_policy=("carry forward last available proxy for PI/over only"
                            if r["sample"] == "all" else "100 subsequent trades observed")))
    write_csv("paper_estimates.csv", canonical)
    # Check identities before rounding, separately in each actual sample.
    keys = {(r["grp"], r["cell"], r["sample"]) for r in canonical}
    max_error = 0.0
    for grp, cell, sample in keys:
        d = {r["metric"]: r["est_bp"] for r in canonical
             if (r["grp"], r["cell"], r["sample"]) == (grp, cell, sample)}
        max_error = max(max_error, abs(d["eff"] - d["info"] - d["RS"]),
                        abs(d["PI_100"] - d["info"] - d["over_100"]))
    assert max_error < 1e-7, max_error

    eq, tests = [], []
    headline_cells = {("YesNo", "pooled"), ("YesNo", "Politics"),
                      ("YesNo", "Crypto"), ("SportsML", "pooled"),
                      ("OverUnder", "pooled")}
    for r in canonical:
        e, se = r["est_bp"], r["se_event_bp"]
        if r["sample"] == "observed_h100" and r["metric"] == "over_100":
            for margin in (50.0, 25.0):
                pt = max(NORM.cdf((e - margin) / se), NORM.cdf((-e - margin) / se))
                eq.append(dict(grp=r["grp"], cell=r["cell"], sample=r["sample"],
                               est_bp=e, se_event_bp=se, margin_bp=margin,
                               ci90_lo_bp=e-NORM.inv_cdf(.95)*se,
                               ci90_hi_bp=e+NORM.inv_cdf(.95)*se,
                               p_tost=pt, equivalent_at_5pct=pt < .05))
        include = ((r["sample"] == "main" and r["metric"] in ("info", "RS")
                    and (r["grp"], r["cell"]) in headline_cells) or
                   (r["sample"] == "observed_h100" and r["metric"] == "over_100"
                    and ((r["grp"], r["cell"]) in headline_cells or " | " in r["cell"])))
        if include:
            tests.append(dict(grp=r["grp"], cell=r["cell"], sample=r["sample"],
                              metric=r["metric"], est_bp=e, se_event_bp=se,
                              p=pvalue(e, se), q_bh=0.0))
    bh(tests)
    assert len(tests) == 21
    write_csv("paper_equivalence.csv", eq)
    write_csv("paper_headline_tests.csv", tests)

    # A contrast between sizes within Politics shares event clusters. Its
    # covariance cannot be recovered from these summaries; do not invent a test.
    differences = []
    for cell in ("", " | s1 <$100", " | s2 $100-1k", " | s3 >=$1k"):
        a = one(canonical, grp="YesNo", cell="Politics"+cell,
                sample="observed_h100", metric="over_100")
        b = one(canonical, grp="YesNo", cell="Crypto"+cell,
                sample="observed_h100", metric="over_100")
        e, se = a["est_bp"]-b["est_bp"], math.hypot(a["se_event_bp"], b["se_event_bp"])
        differences.append(dict(a=a["cell"], b=b["cell"], sample="observed_h100",
                                metric="over_100", difference_bp=e, se_bp=se,
                                p=pvalue(e,se), assumption="independent category-by-day clusters; exploratory"))
    write_csv("paper_cross_topic_differences.csv", differences)

    def est(grp, cell, sample, metric):
        return one(canonical, grp=grp, cell=cell, sample=sample, metric=metric)

    # Paper table: all five quantities refer to the same observed-horizon sample.
    tex = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Cell & Weight & $\eff$ & $\info$ & $\RS$ & $\PI_{100}$ & $\ovr_{100}$ \\", r"\midrule"]
    for grp, cell, label in [("YesNo","pooled","Yes/No"), ("YesNo","Politics","Politics"),
                              ("YesNo","Crypto","Crypto"),("SportsML","pooled","Sports moneyline"),
                              ("OverUnder","pooled","Over/Under")]:
        vals = [est(grp,cell,"observed_h100",m) for m in ("eff","info","RS","PI_100","over_100")]
        tex.append(label + " & per trade & " + " & ".join(
            f"{r['est_bp']:.1f} ({r['se_event_bp']:.1f})" if r["se_event_bp"] < 1 else f"{r['est_bp']:.1f} ({r['se_event_bp']:.1f})" for r in vals) + r" \\")
    weights = read("audit_C_weights_YesNo.csv")
    tex.append(r"\midrule")
    for weight, label in [("equal_market","per market"),("shares","per share"),("usd","per dollar")]:
        vals=[float(one(weights,weight=weight,cat="",metric=m)["est"])*1e4
              for m in ("eff","info","RS","PI_100","over_100")]
        tex.append("Yes/No & "+label+" & "+" & ".join(f"{x:.1f}".replace("-","$-$") for x in vals)+r" \\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    (TABLES/"decomposition.tex").write_text("\n".join(tex)+"\n")

    macros = {}
    for cell, prefix in [("Politics | s1 <$100","PolSmall"),("Politics | s2 $100-1k","PolMid"),
                         ("Politics | s3 >=$1k","PolLarge"),("Crypto | s1 <$100","CrySmall"),
                         ("Crypto | s2 $100-1k","CryMid"),("Crypto | s3 >=$1k","CryLarge")]:
        r=est("YesNo",cell,"observed_h100","over_100")
        macros[prefix]=f"{r['est_bp']:+.1f}"
        macros[prefix+"SE"]=f"{r['se_event_bp']:.1f}"
        q=one(tests,grp="YesNo",cell=cell,sample="observed_h100",metric="over_100")["q_bh"]
        macros[prefix+"Q"]=probability(q)
    account=one(read("acct_B_topdecile_YesNo.csv"),split="half_time",metric="info")
    macros.update(AccountTop=f"{float(account['mean_B'])*1e4:.1f}",
                  AccountPop=f"{float(account['population_mean_B'])*1e4:.1f}",
                  AccountSE=f"{float(account['se_event'])*1e4:.1f}",
                  HeadlineCount=str(len(tests)))
    (PAPER/"numbers.tex").write_text("% Generated from results by analysis/11_outputs.py.\n"+
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in macros.items())+"\n")

    wb=read("inf_B_perm_Y_YesNo.csv")
    classifier=one(read("sim_classifier.csv"),split="held-out parameter configs")
    sim=read("sim_runs.csv")
    max_sim_error=max(abs(float(r["De_after_bp"])) for r in sim)
    digest=dict(identity_max_error_bp=max_error, headline_tests=len(tests),
                account_top_bp=float(account["mean_B"])*1e4,
                account_comparison_bp=float(account["population_mean_B"])*1e4,
                politics_between_permutation_p=float(one(wb,cell="Politics",component="info_between")["p_two_sided"]),
                crypto_between_permutation_p=float(one(wb,cell="Crypto",component="info_between")["p_two_sided"]),
                classifier_heldout_configs=float(classifier["accuracy"]),
                simulation_max_absolute_post_horizon_bias_bp=max_sim_error)
    (RES/"VALIDATION.json").write_text(json.dumps(digest,indent=2)+"\n")
    lines=["# Current results and measurement definitions", "",
           "All probability quantities use 10,000 bp = 1 probability unit. Raw CSV estimates are preserved; derived paper tables are generated by `analysis/11_outputs.py`.", "",
           "The supplementary economic extensions are documented in `results/ECONOMIC_EXTENSIONS.md` and generated by `analysis/12_economic_extensions.py`.", "", "## Samples", "",
           "Main: 1 <= hours to on-chain resolution < 24, trade price in (0.2,0.8), proxy defined, markets with at least 100 trades. Observed horizon: additionally require 100 subsequent trades before resolution. The main-sample PI/over columns in diagnostic source files carry forward the last available proxy; they are not observed 100-trade outcomes.", "",
           "## Headline inference", "",
           "`paper_headline_tests.csv` contains one explicitly enumerated family of 21 tests: main-sample info and RS for five pooled/topic cells (10), observed-horizon over for those cells (5), and observed-horizon over for Politics/Crypto by three sizes (6). BH is applied across all 21. These are analysis choices, not a registered protocol.", "",
           "| Cell | Sample | Metric | Estimate, bp | Event SE | p | BH q |",
           "|---|---|---|---:|---:|---:|---:|"]
    for r in tests:
        lines.append(f"| {r['grp']} / {r['cell']} | {r['sample']} | {r['metric']} | {r['est_bp']:.1f} | {r['se_event_bp']:.1f} | {r['p']:.5g} | {r['q_bh']:.5g} |")
    lines += ["", "## Equivalence", "",
              "TOST uses event-cluster normal errors and fixed margins of 50 and 25 probability bp on the observed-horizon sample. Failure to establish equivalence is not evidence of mispricing. Passing cells:", ""]
    for r in eq:
        if r["equivalent_at_5pct"]:
            lines.append(f"- {r['grp']} / {r['cell']}: ±{r['margin_bp']:.1f} bp, p={r['p_tost']:.5g}.")
    lines += ["", "## Account stability", "",
              f"Top-decile selection uses equal-account first-half info. Second-half trade-weighted info is {digest['account_top_bp']:.1f} bp, compared with {digest['account_comparison_bp']:.1f} bp among eligible accounts; event SE {float(account['se_event'])*1e4:.1f} bp. The split-half slope is equal-account and is a different estimand. This comparison is exploratory and is not a causal estimate of trader skill.", "",
              "## Outcome permutations and simulation", "",
              f"Between-component permutation p values: Politics {digest['politics_between_permutation_p']:.3f}, Crypto {digest['crypto_between_permutation_p']:.3f}; 200 permutations. The within component is Y-invariant by identity. Held-out parameter-configuration classifier accuracy is {digest['classifier_heldout_configs']:.4f}. Maximum absolute post-horizon proxy bias across simulated runs: {max_sim_error:.1f} bp, not a bound for real data.", "",
              "## Interpretation and provenance", "",
              "- Trade p is a USD-weighted fill price. The share-weighted execution price differs by 2.45 bp on average (fp_A_price_gap.csv). The `shares` weight in audit_C_weights_YesNo.csv is the observed outcome-token quantity summed over fills (analysis/14_fill_prices.py); the earlier constructed u/p weight is retained in audit_C_weights_YesNo_constructed_u_over_p.csv. No table measures a wallet's net portfolio profit.",
              "- USD-weighted RS is a weighted probability-price difference, not a return per dollar invested. The cash reconstruction in ret_A_dollars uses u/p and differs from that statistic.",
              "- Relative order size in ret_C_risk uses the market mean USD notional, computed over the source table with tau >= 1 h. Maker HHI in ret_D uses squared maker USD shares, not fill-count shares.",
              "- Market-level imbalance is sum(D*u)/sum(u) within each size class; windows end 24 h before on-chain resolution. They do not end at resolution. Price p_post is the last switch midpoint before that boundary, which may predate a short window if it contains no switch.",
              "- The account analysis, proxy variants, OOS post-price regressions, anchor regressions and robustness matrix retain their original, labelled source samples. They must not be presented as observed-horizon re-estimates.",
              "- Anchor labels are deterministic slug-based candidate reference classes. Availability and price alignment at trade time are not verified. The retained examples are top-three-by-fill examples, not a documented random annotation audit.",
              "- Cross-topic difference SEs assume independent category-by-resolution-day clusters. Within-topic size differences share clusters and are not assigned a significance test from marginal summaries.",
              "- The retained empirical run/switch descriptors in results/descriptive are inputs to the current simulation comparison. They are descriptive and identify no information channel.",
              "- No raw database or event Parquet files are present locally. Saved-summary arithmetic and derived tests can be reproduced locally; raw-data reconstruction requires DATA_ROOT and the dependencies listed in analysis/requirements.txt."]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(digest,indent=2))


if __name__ == "__main__":
    main()
