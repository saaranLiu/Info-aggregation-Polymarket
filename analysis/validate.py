"""Validate paper/source consistency without raw data or third-party packages."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def read(name):
    with (RES / name).open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("outputs", ROOT / "analysis/11_outputs.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    canonical = read("paper_estimates.csv")
    source = read("inf_A_horizon_samples.csv")
    assert len(canonical) == len(source)
    for a,b in zip(canonical,source):
        assert a["grp"] == b["grp"] and a["cell"] == b["cell"] and a["metric"] == b["metric"]
        for dest,src in [("est_bp","est_bp"),("se_market_bp","se_mkt"),("se_event_bp","se_ev")]:
            assert float(a[dest]) == float(b[src]), (dest,a,b)
    tests = read("paper_headline_tests.csv")
    assert len(tests) == 21
    independent = [dict(r,p=float(r["p"]),q_bh=0.0) for r in tests]
    module.bh(independent)
    for a,b in zip(tests,independent):
        assert math.isclose(float(a["q_bh"]),b["q_bh"],abs_tol=1e-14)
        if a["metric"] == "over_100": assert a["sample"] == "observed_h100"
    eq=read("paper_equivalence.csv")
    assert all(r["sample"] == "observed_h100" for r in eq)
    assert not any(r["equivalent_at_5pct"] == "True" and float(r["margin_bp"]) == 25 for r in eq)
    assert not any(r["equivalent_at_5pct"] == "True" and r["cell"] == "Crypto | s3 >=$1k" for r in eq)
    tex=(ROOT/"paper/main.tex").read_text()
    assert not re.search(r"reviewer|earlier version|previous version|withdraw|no longer use|revised claim",tex,re.I)
    assert not re.search(r"XXXX|AAMAS|Dublin|acmConference",tex)
    rendered_prose = tex + "\n" + "\n".join(
        (ROOT/"paper"/(name+".tex")).read_text()
        for name in re.findall(r"\\input\{(tables/[^}]+)\}",tex))
    assert not re.search(r"rather\s+than|instead\s+of|we\s+(?:do|did|have)\s+not|we\s+cannot|不是|而非", rendered_prose, re.I)
    assert r"4.5\% (Politics), 3.0\% (Crypto)" in tex
    assert r"\AccountTop" in tex and r"\AccountPop" in tex and r"\AccountSE" in tex
    labels=set(re.findall(r"\\label\{([^}]+)\}",tex))
    refs=set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}",tex))
    assert refs <= labels, refs-labels
    cites={k.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}",tex) for k in group.split(",")}
    bib=set(re.findall(r"@\w+\s*\{\s*([^,]+)",(ROOT/"paper/ref.bib").read_text()))
    assert cites <= bib,cites-bib
    figs=re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",tex)
    expected_figures = {"measurement_timeline", "weights", "size_pricing", "clock_profile",
                        "forward_prediction", "quote_validation", "proxy_sensitivity", "cross_platform"}
    assert len(figs)==8 and set(figs)==expected_figures, figs
    for name in figs:
        assert (ROOT/"paper/figures"/(name+".pdf")).is_file(),name
    extra={p.stem for p in (ROOT/"paper/figures").glob("*.pdf")}-set(figs)
    assert extra == set(), extra
    joint = read("joint_flow_models.csv")
    assert len(joint) == 36
    predictions = read("joint_flow_predictions.csv")
    groups = {}
    for r in predictions:
        key=(r["grp"],r["analysis_cell"],r["win_name"])
        groups.setdefault(key,[]).append(r)
    assert len(groups) == 12
    for r in joint:
        pp=groups[(r["grp"],r["cell"],r["window"])]
        assert len(pp) == int(r["n"])
        cluster_folds={}
        gains=[]
        for row in pp:
            cl=row["ev_cluster"]; fold=int(row["fold"])
            assert cl not in cluster_folds or cluster_folds[cl] == fold
            cluster_folds[cl]=fold
            y=float(row["Y"]);p0=float(row["pred_price"]);p1=float(row["pred_"+r["model"]])
            assert 0 < p0 < 1 and 0 < p1 < 1
            gains.append((p0-y)**2-(p1-y)**2)
        assert math.isclose(sum(gains)/len(gains),float(r["brier_gain"]),abs_tol=1e-12)
        assert set(cluster_folds.values()) == set(range(5))
    coefficients=read("joint_flow_coefficients.csv")
    assert len(coefficients)==24
    independent=[dict(r,p=float(r["p"]),q_bh=0.) for r in coefficients]
    module.bh(independent)
    for a,b in zip(coefficients,independent):
        assert math.isclose(float(a["q_bh"]),b["q_bh"],abs_tol=1e-12)
    gradients=read("size_response_gradients.csv")
    for r in gradients:
        assert abs(float(r["PI_100"])-float(r["info"])-float(r["over_100"]))<1e-8
    checks=r2_checks(tex)+v3_checks(tex)+editorial_checks(tex)+manuscript_checks(tex)+cross_platform_checks(tex)
    if "--layout" in sys.argv:
        aux=(ROOT/"paper/main.aux").read_text()
        def page(label):
            match=re.search(r"\\newlabel\{"+re.escape(label)+r"\}\{\{[^}]*\}\{(\d+)\}",aux)
            assert match,label
            return int(match.group(1))
        info=subprocess.run(["pdfinfo",str(ROOT/"paper/main.pdf")],capture_output=True,text=True,check=True).stdout
        total=int(re.search(r"Pages:\s+(\d+)",info).group(1))
        body=page("paper:references_start")-1
        assert page("paper:body_end")==body==8, (page("paper:body_end"),body)
        assert total-body==4 and total==12, (body,total)
        assert body+1 <= page("paper:appendix_start") <= total
        print(f"Layout PASS: body pages 1-{body}; references and appendix {total-body} pages; total {total}.")
    for p in (ROOT/"analysis").rglob("*.py"):
        ast.parse(p.read_text(),filename=str(p))
    # Validate manuscript inputs directly; the workspace also keeps historical analyses and templates.
    rows=[]
    for p in sorted(RES.rglob("*.csv")):
        kind=("derived" if p.name.startswith(("paper_","robustness_matrix","joint_","weight_","size_response","account_persistence"))
              else "event-data analysis (scripts 13b-17)" if p.name.startswith(("clk_","qv_","fp_","wb_F","repro_","audit_C_weights_YesNo"))
              else "saved source summary")
        rows.append(dict(file=str(p.relative_to(RES)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),kind=kind))
    manifest=dict(raw_estimate_run_date="2026-10-14",event_data_rerun_date="2026-10-19",unit="probability bp unless specified in README.md",
                  source_values_preserved=True,raw_data_rerun=False,
                  canonical_sample_policy="observed_h100 for fixed-horizon claims; main for pre-trade metrics",
                  files=rows)
    (RES/"SOURCES.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"PASS: {len(source)} source estimates, 21 headline tests, {len(eq)} equivalence tests, 36 common-market models, 24 supplementary coefficients, {len(figs)} figures, {len(cites)} citation keys, {checks} clock/quote/fill/reproduction checks and source/structure checks.")


def r2_checks(tex):
    """Generated macros and tables agree with the result files; identities close on the new analyses."""
    macros=dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$",(ROOT/"paper/numbers_r2.tex").read_text(),re.M))
    used=set(re.findall(r"\\([A-Za-z]+)",tex))
    n=0
    def num(m): return float(macros[m].replace("\\ensuremath{-}","-").replace(",",""))
    clk={(r["cutoff"],r["cell"],r["metric"]):r for r in read("clk_A_cells.csv")}
    for cut,kk in [("R","R"),("S","S"),("S1","Sone"),("C","C")]:
        for cell,pre in [("Politics | s1 <$100","PolS"),("Crypto | s1 <$100","CryS"),("Politics | s3 >=$1k","PolL")]:
            r=clk[(cut,cell,"over_c3600")]
            assert num(f"{pre}Hour{kk}")==round(float(r["est_bp"]),1), (cut,cell); n+=1
            assert abs(float(r["est_bp"])-(float(clk[(cut,cell,"PI_c3600")]["est_bp"])-float(clk[(cut,cell,"info")]["est_bp"])))<1e-6; n+=1
    for r in read("clk_E_diff.csv"):
        a=clk[(r["cutoff"],"Politics | s1 <$100",r["metric"])]; b=clk[(r["cutoff"],"Crypto | s1 <$100",r["metric"])]
        assert abs(float(r["diff_bp"])-float(a["est_bp"])+float(b["est_bp"]))<1e-6; n+=1
    w=[r for r in read("audit_C_weights_YesNo.csv") if r["weight"]=="shares" and r["cat"]==""]
    e={r["metric"]:float(r["est"]) for r in w}
    assert abs(e["eff"]-e["info"]-e["RS"])<1e-10 and abs(e["PI_100"]-e["info"]-e["over_100"])<1e-10; n+=1
    assert num("FPRSSh")==round(1e4*e["RS"],1); n+=1
    for r in read("wb_F_size.csv"):
        assert abs(float(r["total_bp"])-float(r["between_bp"])-float(r["within_bp"]))<1e-6; n+=1
    dif={r["cutoff"]:float(r["diff_bp"]) for r in read("clk_E_diff.csv") if r["metric"]=="over_c3600"}
    for r in read("ctl_A_residual_controls.csv"):
        if not r["spec"].startswith("+"):
            assert abs(float(r["contrast_bp"])-dif[r["cutoff"]])<1e-6, r; n+=1
        else:
            assert num(f"Ctl{r['cutoff']}")==round(float(r["contrast_bp"]),1); n+=1
    assert len(read("repro_sim_grid.csv"))==34; n+=1
    rules=read("repro_anchor_rules.csv"); ann=read("anchor_A_annotation_markets.csv")
    assert sum(int(r["n_markets"]) for r in rules)<=len(ann); n+=1
    if macros.get("HaveQuotes")=="1":
        for r in read("qv_D_metrics.csv"):
            if r["diff_bp"] and r["proxy_bp"]:
                assert abs(float(r["proxy_bp"])-float(r["quote_bp"])-float(r["diff_bp"]))<1e-6; n+=1
        for r in read("qv_E_wb.csv"):
            assert abs(float(r["total_bp"])-float(r["between_bp"])-float(r["within_bp"]))<1e-6; n+=1
        cov=[r for r in read("qv_F_coverage.csv") if r["cat"] in ("Politics","Crypto")]
        assert num("QMktsAll")==sum(float(r["markets_with_quotes"] or 0) for r in cov); n+=1
    missing=[m for m in used if m[0].isupper() and m not in macros and m not in NUMBERS_MACROS()]
    assert not [m for m in missing if m.startswith(("Q","Pol","Cry","Diff","Fwd","WB","FP","Ctl"))], missing
    return n


def v3_checks(tex):
    """Third-round macros match the result files. Equivalence p-values are recomputed; the contrast file stores twice the one-sided tail."""
    macros=dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$",(ROOT/"paper/numbers_v3.tex").read_text(),re.M))
    def num(m): return float(macros[m].replace("\\ensuremath{-}","-"))
    n=0
    C=read("ctr_A_contrasts.csv")
    def crow(sample, metric, name):
        rows=[r for r in C if r["sample"]==sample and r["metric"]==metric and r["contrast"].startswith(name)]
        assert len(rows)==1, (sample, metric, name)
        return rows[0]
    t2=crow("h100","over","theta2")
    assert num("ThTwo")==round(float(t2["est_bp"]),1) and num("ThTwoSE")==round(float(t2["se_ev_bp"]),1)
    assert num("ThTwoW")==round(float(t2["se_2way_bp"]),1); n+=2
    def tost(est, se, m):
        def tail(z): return 0.5*(1-math.erf(z/math.sqrt(2)))
        return max(tail((m-est)/se), tail((m+est)/se))
    e50=crow("h100","over","equiv Crypto small 50")
    assert abs(num("EqFiftyP")/100-tost(float(e50["est_bp"]), float(e50["se_ev_bp"]), 50))<5e-4; n+=1
    assert abs(num("EqTwentyP")/100-tost(float(e50["est_bp"]), float(e50["se_ev_bp"]), 25))<5e-4; n+=1
    B=read("dir_B_metrics.csv")
    ps=[r for r in B if r["step"]=="fixed midpoint" and r["cell"]=="Politics | s1 <$100" and r["metric"]=="over"][0]
    assert num("DirPolS")==round(float(ps["diff_bp"]),1); n+=1
    K=read("con_A_concentration.csv")
    u=[r for r in K if r["cat"]=="Yes/No" and r["weight"]=="usd"][0]
    assert num("ConUsd")==round(float(u["mean_bp"]),1) and num("ConDrop")==round(float(u["drop5_bp"]),1); n+=1
    S=read("spl_A_prediction.csv")
    m2=[r for r in S if r["design"]=="cv" and r["model"]=="M2"][0]
    assert abs(num("SplCv")-1e4*float(m2["dbrier_vs_linear"]))<0.05; n+=1
    ext=dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$",(ROOT/"paper/numbers_ext.tex").read_text(),re.M))
    def enum(m): return float(ext[m].replace("\\ensuremath{-}","-").replace(",",""))
    def drow(sample, metric, cell):
        rows=[r for r in read("d1_B_metrics.csv") if r["sample"]==sample and r["metric"]==metric and r["cell"]==cell]
        assert len(rows)==1, (sample, metric, cell)
        return rows[0]
    info=drow("quote minus proxy","info","pooled")
    resid=drow("quote minus proxy","over_100","pooled")
    assert enum("DOneInfo")==round(float(info["est_bp"]),1) and enum("DOneInfoSE")==round(float(info["se_bp"]),1); n+=1
    assert enum("DOneRes")==round(float(resid["est_bp"]),1) and enum("DOneResSE")==round(float(resid["se_bp"]),1); n+=1
    assert enum("DOneLead")==int(float(read("d1_C_coverage.csv")[0]["lead_s"])); n+=1
    hit=read("d5_D_coverage.csv")[0]
    assert enum("ExtBtcHit")==int(float(hit["n_candle"])) and enum("ExtBtcN")==int(float(hit["n_window"])); n+=1
    d6=read("d6_A_coverage.csv")[0]
    assert enum("ExtDsixHit")==int(float(d6["n_with_candle"])); n+=1
    def kget(scheme, b):
        rows=[r for r in read("k_B_margin.csv") if r["scheme"]==scheme and r["bin"]==b]
        assert len(rows)==1, (scheme, b)
        return rows[0]
    kd=kget("dollars","s3 >=$1k minus s1 <$100")
    kr=kget("within-market rank","r3 high minus r1 low")
    assert enum("KalDiff")==round(float(kd["trade_bp"]),1) and enum("KalDiffSE")==round(float(kd["trade_se"]),1); n+=1
    assert enum("KalRank")==round(float(kr["trade_bp"]),1) and enum("KalRankSE")==round(float(kr["trade_se"]),1); n+=1
    used=set(re.findall(r"\\([A-Za-z]+)", tex))
    missing=[m for m in used if m.startswith(("Th","Dir","Con","Ret","Spl","Eq","Ord")) and m not in macros]
    assert not missing, missing
    missing_ext=[m for m in used if m.startswith(("DOne","Ext","Kal")) and m not in ext]
    assert not missing_ext, missing_ext
    return n


def manuscript_checks(tex):
    """One output precision and one physical table style, independent of saved precision."""
    assert r"\fontsize{7.5}{9.0}" in tex and r"\arraystretch}{1.05}" in tex
    assert r"\bibliography{ref}" + "\n" + r"\appendix" in tex
    tables=re.findall(r"\\input\{(tables/[^}]+)\}",tex)
    for name in tables:
        table=(ROOT/"paper"/(name+".tex")).read_text()
        assert not re.search(r"\d+\.\d{2,}",table),name
        assert not re.search(r"\\(?:fontsize|scriptsize|footnotesize|small|resizebox|scalebox)\b|\\arraystretch",table),name
        assert not re.search(r"(?<![\d.])-0\.0(?!\d)",table),name
    macros={}
    for path in (ROOT/"paper").glob("numbers*.tex"):
        macros.update(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$",path.read_text(),re.M))
    used=set(re.findall(r"\\([A-Za-z]+)",tex))
    for name in used & macros.keys():
        assert not re.search(r"\d+\.\d{2,}",macros[name]),(name,macros[name])
    # Check that the explicit coefficient/loss scaling preserves the saved values.
    joint=(ROOT/"paper/tables/joint_flow.tex").read_text()
    pol=next(r for r in read("joint_flow_models.csv") if r["grp"]=="YesNo" and r["cell"]=="Politics" and r["window"]=="W_paper (life before 24h)" and r["model"]=="joint")
    assert f"{10*float(pol['small_b']):+.1f} ({10*float(pol['small_se']):.1f})" in joint
    spline=(ROOT/"paper/tables/spline.tex").read_text()
    for row in read("spl_A_prediction.csv"):
        assert f"{1e4*float(row['brier']):.1f}" in spline
    return len(tables)+4


def NUMBERS_MACROS():
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",(ROOT/"paper/numbers.tex").read_text()))


def cross_platform_checks(tex):
    """Venue displays preserve source samples, units and separate predictive cohorts."""
    shares=read('k_D_participation_comparison.csv')
    assert len(shares)==6
    kal=read('k_B_margin.csv')
    poly=read('wb_F_size.csv')
    n=0
    for venue in ['Polymarket','Kalshi']:
        rows=[r for r in shares if r['venue']==venue]
        assert math.isclose(sum(float(r['share_pct']) for r in rows),100,abs_tol=1e-10);n+=1
        for r in rows:
            if venue=='Polymarket':
                source=next(x for x in poly if x['sample']=='main' and x['cell']=='pooled' and x['size']==r['bin'])
            else:
                source=next(x for x in kal if x['scheme']=='dollars' and x['bin']==r['bin'])
            assert int(r['n'])==int(float(source['n']));n+=1
        assert next(float(r['share_pct']) for r in rows if r['bin']=='s1 <$100')>80;n+=1
    margins=read('k_E_margin_comparison.csv')
    assert len(margins)==9
    for r in margins:
        if r['venue']=='Kalshi':
            source=next(x for x in kal if x['scheme']=='dollars' and x['bin']==r['bin'])
            estimate,se=source['trade_bp'],source['trade_se']
        else:
            source=next(x for x in read('inf_A_horizon_samples.csv') if x['grp']=='YesNo' and x['sample']=='all' and x['metric']=='RS' and x['cell']==r['topic']+' | '+r['bin'])
            estimate,se=source['est_bp'],source['se_ev']
        assert float(r['estimate_bp'])==float(estimate) and float(r['se_bp'])==float(se);n+=1
        if r['bin']=='s1 <$100':
            assert float(estimate)-1.96*float(se)>0;n+=1
    prediction=read('k_C_prediction.csv')
    small,both=prediction
    assert small['sample']=='small flow, forward' and both['sample']=='both flows, forward'
    assert int(small['n'])>int(both['n'])
    assert float(small['hi'])<0<float(both['lo']);n+=3
    table=(ROOT/'paper/tables/external_kalshi_prediction.tex').read_text()
    for r in prediction:
        assert f"{int(r['n']):,}" in table
        assert f"{1e4*float(r['brier_price']):.1f}" in table;n+=2
    assert r'\label{sec:kalshi}' in tex and r'\label{tab:extkalshipred}' in tex
    return n+1


def editorial_checks(tex):
    """Presentation-only numbers retain their saved estimands and Brier scale."""
    macros=dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$",(ROOT/"paper/numbers_editorial.tex").read_text(),re.M))
    def num(m): return float(macros[m].replace("\\ensuremath{-}","-"))
    trade=next(r for r in read("con_A_concentration.csv") if r["cat"]=="Yes/No" and r["weight"]=="trade")
    assert num("EdConTradeDrop")==round(float(trade["drop5_bp"]),1)
    joint=next(r for r in read("joint_flow_models.csv") if r["grp"]=="YesNo" and r["cell"]=="Politics" and r["window"]=="W_paper (life before 24h)" and r["model"]=="joint")
    for macro,field in [("EdJointPolGain","brier_gain"),("EdJointPolLo","brier_gain_lo"),("EdJointPolHi","brier_gain_hi")]:
        assert abs(num(macro)-1e4*float(joint[field]))<=.05
    for macro,b in [("EdKalSmallMargin","s1 <$100"),("EdKalLargeMargin","s3 >=$1k")]:
        r=next(r for r in read("k_B_margin.csv") if r["scheme"]=="dollars" and r["bin"]==b)
        assert num(macro)==round(float(r["trade_bp"]),1)
    used=set(re.findall(r"\\(Ed[A-Za-z]+)",tex))
    assert used <= macros.keys(), used-macros.keys()
    joint_table=(ROOT/"paper/tables/joint_flow.tex").read_text()
    assert r"10^4\Delta" in joint_table and r"10^3\Delta" not in joint_table
    assert "+18.7 [-5.6, +41.5]" in joint_table
    return 8


if __name__ == "__main__":
    main()
