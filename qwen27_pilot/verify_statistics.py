"""Independently recompute summaries and statistics from raw instance metrics."""
from pathlib import Path
import json,csv,math
import numpy as np
from scipy.stats import wilcoxon
b=Path(__file__).resolve().parent.parent;p=b/'data/qwen27_pilot'
m=json.loads((p/'llm_pilot_metadata.json').read_text())
rows=list(csv.DictReader((p/'llm_pilot_raw_metrics.csv').open()))
labels=['LLM pool + nominal evaluator','LLM pool + fuzzy MO evaluator (C-R-EoH)']
values={label:{k:np.array([float(r[k]) for r in rows if r['evaluator']==label]) for k in ['f1','f2','f3','p95','hv','ood']} for label in labels}
assert all(len(x)==30 for d in values.values() for x in d.values())
scale=100/values[labels[0]]['f1'].mean()
for label,d in values.items():
 for k,x in d.items():
  assert np.isfinite(x).all()
  scaled=x*(1 if k in ['hv','ood'] else scale)
  pair=[round(float(scaled.mean()),3),round(float(1.96*scaled.std(ddof=1)/math.sqrt(len(scaled))),3)]
  assert pair==m['summary'][label][k],(label,k,pair,m['summary'][label][k])
raw={};delta={}
for k in ['p95','f2','f3','hv','f1']:
 a,c=values[labels[0]][k],values[labels[1]][k]
 try:prob=float(wilcoxon(c,a).pvalue)
 except ValueError:prob=1.0
 raw[k]=prob;delta[k]=round(float(np.sign(c[:,None]-a[None,:]).mean()),3)
adjusted={};prev=0
for i,k in enumerate(sorted(raw,key=raw.get)):
 prev=max(prev,min(1,(len(raw)-i)*raw[k]));adjusted[k]=prev
for k,pval,padj,d in m['stats']:
 assert abs(pval-raw[k])<1e-12 and abs(padj-adjusted[k])<1e-12 and d==delta[k]
reduction=round(100*(values[labels[0]]['p95'].mean()-values[labels[1]]['p95'].mean())/values[labels[0]]['p95'].mean(),2)
assert reduction==m['tail_reduction_pct']
original=json.loads((b/'qwen27_pilot/experiment/evaluation_gen3/llm_pilot_metadata.json').read_text())
for d in [original,m]:d.pop('mean_program_runtime_ms')
assert original==m
report={'raw_metric_rows':len(rows),'instances_per_selector':30,'means_and_intervals_recomputed':True,'wilcoxon_holm_and_cliffs_delta_recomputed':True,'tail_reduction_recomputed':reduction,'replay_matches_original_excluding_timing':True}
(b/'qwen27_pilot/experiment/statistical_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
