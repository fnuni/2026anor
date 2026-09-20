"""Verify request/response provenance and final evaluation completeness."""
from pathlib import Path
import collections, csv, hashlib, json
import pilot_ollama as connector
ROOT = Path(__file__).resolve().parent / 'experiment'

def main():
    expected = {f'gen{g}_{i:02d}' for g in range(1, 4) for i in range(1, 7)}
    assert {p.stem for p in (ROOT/'candidates').glob('gen*_*.py')} == expected
    stop = collections.Counter()
    for name in sorted(expected):
        req=json.loads((ROOT/'logs'/f'{name}_request.json').read_text())
        res=json.loads((ROOT/'logs'/f'{name}_response.json').read_text())
        g,i=map(int,name.replace('gen','').split('_'))
        prompt=(connector.SOURCE/'llm_candidates/prompts/gen1_prompt.txt').read_text()
        if g>1: prompt+='\n\nFEEDBACK FROM PREVIOUS OLLAMA GENERATIONS\n'+(ROOT.parent/f'feedback_gen{g}.txt').read_text()
        assert req['messages']==[{'role':'user','content':prompt}]
        assert req['model']==connector.MODEL and req['think'] is False
        assert req['options']=={**connector.OPTIONS,'seed':1000*g+i}
        assert res['done'] and not res['message'].get('thinking')
        stop[res.get('done_reason','unspecified')]+=1
        assert (ROOT/'candidates'/f'{name}.py').read_text()==connector.extract(res['message']['content'])
    meta=json.loads((ROOT/'backbone.json').read_text())
    for name,sha in meta['evaluator_sha256'].items():
        assert hashlib.sha256((connector.SOURCE/name).read_bytes()).hexdigest()==sha
    final=json.loads((ROOT/'evaluation_gen3/llm_pilot_metadata.json').read_text())
    assert final['backbone']==meta and final['programs_generated']==18
    rows=list(csv.DictReader((ROOT/'evaluation_gen3/llm_pilot_executions.csv').open()))
    admitted={r['program'] for r in rows}
    assert len(admitted)==final['programs_admitted']
    assert {(r['program'],int(r['seed'])) for r in rows}=={(p,s) for p in admitted for s in range(30)}
    assert len(rows)==final['executions']
    statuses=dict(collections.Counter(r['status'] for r in rows))
    assert statuses==final['execution_status']
    report={'checks':'passed','candidate_count':18,'admitted':len(admitted),
            'executions':len(rows),'status_counts':statuses,'stop_reasons':dict(stop),
            'source_matches_responses':True,'feedback_matches_requests':True,
            'evaluator_hashes_match':True,'note':'Integrity checks, not independent replication.'}
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
