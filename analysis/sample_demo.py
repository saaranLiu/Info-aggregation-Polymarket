"""Check the supplied examples and calculate demonstration Brier gains (stdlib).

This runs on saved observations/predictions. It does not refit the paper's models
or estimate a full-population effect from an outcome-stratified sample.
"""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math

ROOT=Path(__file__).resolve().parents[1]
SAMPLES=ROOT/'data'/'samples'


def read(name):
    with (SAMPLES/name).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    estimates=defaultdict(dict)
    for r in read('settlement_estimates.csv'):
        estimates[(r['grp'],r['cell'],r['sample'])][r['metric']]=float(r['est_bp'])
    checks=0
    for values in estimates.values():
        assert math.isclose(values['eff']-values['info'],values['RS'],abs_tol=1e-6)
        assert math.isclose(values['PI_100']-values['info'],values['over_100'],abs_tol=1e-6)
        checks+=2

    grouped=defaultdict(list)
    event_folds={}
    for r in read('heldout_predictions.csv'):
        key=(r['grp'],r['analysis_cell'],r['win_name'])
        event=key+(r['ev_cluster'],)
        fold=int(r['fold'])
        assert fold in range(5)
        assert event not in event_folds or event_folds[event]==fold
        event_folds[event]=fold
        y=float(r['Y']); assert y in (0,1)
        for name in ['price','small','large','joint']:
            assert 0<float(r['pred_'+name])<1
        gain=(float(r['pred_price'])-y)**2-(float(r['pred_joint'])-y)**2
        grouped[key].append(gain)
    output=[]
    for key,gain in sorted(grouped.items()):
        output.append(dict(group=key[0],cell=key[1],window=key[2],sample_rows=len(gain),
                           joint_brier_gain_x10000=1e4*sum(gain)/len(gain)))

    kal=read('kalshi_predictions.csv')
    assert {r['sample'] for r in kal}=={'small flow, forward','both flows, forward'}
    assert len({r['n'] for r in kal})==2
    target=ROOT/'work'/'sample_demo'
    target.mkdir(parents=True,exist_ok=True)
    result=dict(scope='Demonstration subset only; not full-paper population estimates',
                accounting_identities_checked=checks, sample_predictive_groups=output,
                kalshi_cohorts='Separate observed-flow cohorts; not a nested common-sample comparison')
    (target/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Sample demo PASS: {checks} accounting identities; {len(output)} predictive groups; event folds intact.')
    print('Subset results: work/sample_demo/summary.json (demonstration only).')


if __name__ == '__main__':
    main()
