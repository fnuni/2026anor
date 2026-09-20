"""Check provenance, completeness and consistency of the completed Ollama pilot."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import pilot_ollama as connector

root = Path(__file__).resolve().parent / 'experiment'
meta = json.loads((root/'evaluation_gen3/llm_pilot_metadata.json').read_text())
backbone = json.loads((root/'backbone.json').read_text())
expected = {f'gen{g}_{i:02d}' for g in range(1,4) for i in range(1,7)}
assert {p.stem for p in (root/'candidates').glob('*.py')} == expected
sources = collections.defaultdict(list)
for name in sorted(expected):
    request = json.loads((root/'logs'/f'{name}_request.json').read_text())
    response = json.loads((root/'logs'/f'{name}_response.json').read_text())
    g, i = map(int, name.replace('gen','').split('_'))
    prompt = (connector.SOURCE/'llm_candidates/prompts/gen1_prompt.txt').read_text()
    if g > 1:
        prompt += '\n\nFEEDBACK FROM PREVIOUS OLLAMA GENERATIONS\n' + (root.parent/f'feedback_gen{g}.txt').read_text()
    assert request['messages'] == [{'role':'user','content':prompt}]
    assert request['options'] == {**connector.OPTIONS, 'seed':1000*g+i}
    assert request['model'] == connector.MODEL
    assert response['done'] and response['done_reason'] == 'stop'
    source = (root/'candidates'/f'{name}.py').read_text()
    assert source == connector.extract(response['message']['content'])
    sources[hashlib.sha256(source.encode()).hexdigest()].append(name)
for name, digest in backbone['evaluator_sha256'].items():
    assert hashlib.sha256((connector.SOURCE/name).read_bytes()).hexdigest() == digest
assert meta['backbone'] == backbone
rows = list(csv.DictReader((root/'evaluation_gen3/llm_pilot_executions.csv').open()))
assert len(rows) == 540
assert {(r['program'],int(r['seed'])) for r in rows} == {(p,s) for p in expected for s in range(30)}
counts = dict(collections.Counter(r['status'] for r in rows))
assert counts == meta['execution_status']
assert all(r['repaired'] == 'False' for r in rows)
assert all(sum(r['status']=='ok' for r in rows if int(r['seed'])==s) == 7 for s in range(30))
for d in meta['summary'].values():
    assert all(math.isfinite(x) for v in d.values() for x in v)
main = list(csv.DictReader((root/'evaluation_gen3/llm_pilot_main.csv').open()))
for row in main:
    d = meta['summary'][row['evaluator_on_llm_pool']]
    for metric in ['f1','f2','f3','p95','hv','ood']:
        assert [float(row[metric+'_mean']), float(row[metric+'_ci'])] == d[metric]
report = dict(checks='passed', candidates=18, requests=18, responses=18,
              executions=540, instances=30, usable_candidates_each_instance=7,
              status_counts=counts, exact_duplicate_sources=[v for v in sources.values() if len(v)>1],
              candidate_text_matches_raw_responses=True, evaluator_sources_unchanged=True,
              feedback_matches_requests=True, all_responses_completed_without_token_truncation=True,
              note='Integrity and consistency checks; not an independent scientific replication.')
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
