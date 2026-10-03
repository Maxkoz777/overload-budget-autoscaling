"""Independent, read-only replay checks.

Writes only review/readiness_checks_2026_09_23/.
Window 1440 is an explicitly retrospective sensitivity, not a new primary policy.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent / 'experiments'
OUT = ROOT / 'review/readiness_checks_2026_09_23'
OUT.mkdir(exist_ok=True)
selection = pd.read_csv(EXP / 'data/splits/selected_services_200.csv')
split = {s['name']: s for s in json.loads((EXP / 'data/splits/split_definition.json').read_text())['splits']}
ts = split['test']

def metrics(capacity, demand, denominator):
    ol = float(np.mean(demand > capacity))
    total = capacity.sum() + .05 * np.abs(np.diff(capacity)).sum() + 10 * np.count_nonzero(demand > capacity)
    return ol, float(total / denominator)

def add_guard(nominal, demand):
    out = nominal.copy()
    streak = 0
    for t in range(len(out)):
        if streak >= 2:
            out[t] += 1
        streak = streak + 1 if demand[t] > .7 * out[t] else 0
    return out

rows = []
for i, item in enumerate(selection.itertuples(), 1):
    f = pd.read_parquet(EXP / f'data/service_timeseries_200_verified/{item.service_id}.parquet').sort_values('timestamp')
    pre = f[f.timestamp < ts['timestamp_start']].cpu_sum.to_numpy(float)
    test = f[(f.timestamp >= ts['timestamp_start']) & (f.timestamp < ts['timestamp_end_exclusive'])]
    d = test.cpu_sum.to_numpy(float)
    predicted = np.concatenate(([pre[-1]], d[:-1]))
    signed = np.diff(np.concatenate((pre, d)))
    observed = np.maximum(1, np.ceil(test.replica_count.to_numpy(float)))
    denominator = observed.sum() + .05 * np.abs(np.diff(observed)).sum() + 10 * np.count_nonzero(d > observed)
    for W in (240, 1440):
        start = len(pre) - W - 1
        win = np.lib.stride_tricks.sliding_window_view(signed, W)[start:start+len(d)]
        assert np.array_equal(win[0], np.diff(pre)[-W:])
        ranks = sorted(set([math.ceil((W+1)*(1-x))-1 for x in (.05, .01)] + [math.ceil(W*(1-x))-1 for x in (.05,.01)]))
        part = np.partition(np.maximum(win, 0), ranks, axis=1)
        for delta in (.05,.01):
            arms = {
                'conformal': part[:,math.ceil((W+1)*(1-delta))-1],
                'invcdf': part[:,math.ceil(W*(1-delta))-1],
                'gaussian': np.maximum(0,win.mean(1)+norm.ppf(1-delta)*win.std(1)),
            }
            if delta == .01:
                arms['gaussian_z4'] = np.maximum(0,win.mean(1)+4*win.std(1))
            for method, margin in arms.items():
                nominal = np.maximum(1,np.ceil(predicted+margin))
                for guard in (False,True):
                    c = add_guard(nominal,d) if guard else nominal
                    ol,rc = metrics(c,d,denominator)
                    rows.append(dict(service_id=item.service_id,focused=bool(item.is_focused_20),W=W,delta=delta,method=method,guard=guard,ol=ol,rc=rc))
    if i % 50 == 0:
        print(f'{i}/200 services',flush=True)
per = pd.DataFrame(rows)
per.to_csv(OUT/'independent_replay.csv',index=False)
characteristics = pd.read_csv(ROOT/'audit/verified_results/external_validity/service_characteristics.csv')
q4 = set(characteristics.loc[characteristics.pretest_size_quartile.eq('Q4'),'service_id'])
panels=[]
for cohort, block in [('all200',per),('focused20',per[per.focused]),('Q4',per[per.service_id.isin(q4)])]:
    for keys,g in block.groupby(['W','delta','method','guard']):
        W,delta,method,guard=keys
        panels.append(dict(cohort=cohort,W=W,delta=delta,method=method,guard=guard,n=len(g),compliant=int((g.ol<=delta).sum()),mean_ol=g.ol.mean(),median_ol=g.ol.median(),mean_rc=g.rc.mean(),median_rc=g.rc.median()))
summary=pd.DataFrame(panels)
summary.to_csv(OUT/'independent_summary.csv',index=False)

# Compare two independent saved implementations at their shared operating points.
canonical=pd.read_csv(ROOT/'audit/verified_results/rank_ablation/rank_ablation_per_service.csv').replace({'method':{'empirical_inverse_cdf':'invcdf'}})
m=per[per.W.eq(240)&per.method.ne('gaussian_z4')].merge(canonical,on=['service_id','delta','method','guard'],validate='one_to_one')
assert len(m)==2400
errors={'canonical_rows':len(m),'max_ol_error':float(abs(m.ol-m.overload_fraction).max()),'max_cost_error':float(abs(m.rc-m.relative_cost).max())}
assert errors['max_ol_error']<1e-12 and errors['max_cost_error']<1e-12
other=pd.read_csv(ROOT/'review/round2_checks_2026_09_23/frontier_all200.csv')
other=other[other.method.eq('conformal')&other.knob.isin([.05,.01])].rename(columns={'svc':'service_id','knob':'delta','ol':'other_ol','rc':'other_rc'})
m=per[per.method.eq('conformal')].merge(other,on=['service_id','W','delta','method','guard'],validate='one_to_one')
errors.update(round2_rows=len(m),round2_max_ol_error=float(abs(m.ol-m.other_ol).max()),round2_max_cost_error=float(abs(m.rc-m.other_rc).max()))
assert len(m)==1600 and errors['round2_max_ol_error']<1e-12 and errors['round2_max_cost_error']<1e-12

# Reselect reactive parameters using only archived calibration outcomes.
cal=pd.read_csv(ROOT/'audit/verified_results/comparative/reactive_calibration_grid_per_service.csv')
test=pd.read_csv(ROOT/'audit/verified_results/comparative/reactive_test_grid_per_service.csv')
reactive=[]
for budget in (.01,.0025):
    choices=[]
    for _,g in cal.groupby('service_id'):
        feasible=g[g.overload_fraction<=budget]
        cols=['relative_cost','overload_fraction','threshold','cooldown'] if len(feasible) else ['overload_fraction','relative_cost','threshold','cooldown']
        source=feasible if len(feasible) else g
        choices.append(source.sort_values(cols,ascending=[True,True,False,True]).iloc[0])
    selected=pd.DataFrame(choices)
    assert len(selected)==200
    selected=selected[['service_id','threshold','cooldown']].merge(test,on=['service_id','threshold','cooldown'],validate='one_to_one')
    for cohort,g in [('all200',selected),('focused20',selected[selected.is_focused_20]),('Q4',selected[selected.service_id.isin(q4)])]:
        reactive.append(dict(cohort=cohort,calibration_budget=budget,n=len(g),compliant_at_01=int((g.overload_fraction<=.01).sum()),mean_ol=g.overload_fraction.mean(),median_rc=g.relative_cost.median(),mean_rc=g.relative_cost.mean()))
pd.DataFrame(reactive).to_csv(OUT/'reactive_reselection.csv',index=False)

# Counterexample to monotonicity in variance inferred from Jensen alone.
jensen=[]
for lo,hi in [(4.,10.),(4.6,100.)]:
    prob=(.01-np.exp(-hi))/(np.exp(-lo)-np.exp(-hi))
    mean=prob*lo+(1-prob)*hi
    var=prob*(lo-mean)**2+(1-prob)*(hi-mean)**2
    jensen.append(dict(lo=lo,hi=hi,p_lo=float(prob),mean=float(mean),variance=float(var),expected_tail=float(prob*np.exp(-lo)+(1-prob)*np.exp(-hi))))
(OUT/'check_results.json').write_text(json.dumps({'replay_agreement':errors,'jensen_counterexample':jensen},indent=2))
print(json.dumps(errors,indent=2))
print(summary[(summary.delta==.01)&summary.method.isin(['conformal','gaussian_z4'])].to_string(index=False))
print(pd.DataFrame(reactive).to_string(index=False))
