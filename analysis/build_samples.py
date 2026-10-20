"""Build small, authentic examples from the saved market-level study data.

Selection is deterministic and stratified by topic/outcome, so sample statistics
are demonstrations, not estimates of the paper's population. Entire available
window histories of each selected market are retained. No synthetic trades or
midpoints are reconstructed from aggregate estimates.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'samples'
WINDOW = 'W_paper (life before 24h)'
PER_STRATUM = 100


def read(name):
    path = ROOT / 'results' / name
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def render(fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def valid_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify existing files without replacing them')
    args = parser.parse_args()
    payloads, entries, selections = {}, {}, {}

    def record(name, source, fields, rows, kind):
        payloads[name] = render(fields, rows)
        entries[name] = dict(source='results/'+source, kind=kind, rows=len(rows),
                             source_sha256=hashlib.sha256((ROOT/'results'/source).read_bytes()).hexdigest(),
                             sha256=hashlib.sha256(payloads[name]).hexdigest())

    for group, output in [('YesNo','polymarket_windows.csv'), ('SportsML','sports_moneyline_windows.csv')]:
        source=f'mkt_A_windows_{group}.csv'
        fields, rows=read(source)
        strata=defaultdict(set)
        for r in rows:
            if r['win_name'] != WINDOW or int(r['n_win']) < 20:
                continue
            if not all(valid_number(r[k]) for k in ['p_post','imb_small','imb_large']):
                continue
            if not .02 < float(r['p_post']) < .98:
                continue
            strata[(r['cat'], r['Y'])].add(r['condition_id'])
        chosen=set()
        for key, ids in sorted(strata.items()):
            ordered=sorted(ids,key=lambda x:(hashlib.sha256(x.encode()).hexdigest(),x))
            chosen.update(ordered[:PER_STRATUM])
        selections[group]=chosen
        sample=sorted((r for r in rows if r['condition_id'] in chosen),
                      key=lambda r:(r['condition_id'],r['win_name']))
        record(output,source,fields,sample,'market-window observations')
        entries[output]['markets']=len(chosen)
        entries[output]['topic_outcome_markets']={f'{cat}|Y={y}':len(chosen & ids)
            for (cat,y),ids in sorted(strata.items())}

    fields, rows=read('joint_flow_predictions.csv')
    selected=[r for r in rows if r['condition_id'] in selections[r['grp']]]
    selected.sort(key=lambda r:(r['grp'],r['analysis_cell'],r['win_name'],r['condition_id']))
    record('heldout_predictions.csv','joint_flow_predictions.csv',fields,selected,'saved out-of-fold predictions')
    for name, source, kind in [
        ('settlement_estimates.csv','paper_estimates.csv','saved estimates; not individual trades'),
        ('kalshi_margins.csv','k_B_margin.csv','saved cluster-level inference summaries; not individual trades'),
        ('kalshi_predictions.csv','k_C_prediction.csv','saved predictions evaluated on separate cohorts')]:
        fields,rows=read(source)
        record(name,source,fields,rows,kind)
    manifest=dict(version='2026-10-20', selection=dict(
        unit='complete market across its available windows', stratification='topic and binary outcome',
        order='ascending SHA-256 of condition_id', max_markets_per_stratum=PER_STRATUM,
        eligibility='paper window; >=20 window trades; observed small/large flow; 0.02<p_post<0.98',
        use='demonstration; not a representative sample for reproducing population estimates'), files=entries)
    payloads['manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    if args.check:
        errors=[name for name,data in payloads.items() if not (OUT/name).is_file() or (OUT/name).read_bytes()!=data]
        if errors:
            raise SystemExit('Sample/source mismatch: '+', '.join(errors))
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        for name,data in payloads.items(): (OUT/name).write_bytes(data)
    print(('Verified' if args.check else 'Wrote')+f' {len(entries)} sample files; '+
          ', '.join(f'{g}: {len(ids):,} markets' for g,ids in selections.items()))


if __name__ == '__main__':
    main()
