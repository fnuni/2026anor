"""Local Ollama connector; the published pilot and its results are unchanged.

Smoke test: python3 pilot_ollama.py smoke
Generation 1: python3 pilot_ollama.py generate --generation 1
Evaluate cumulative pool: python3 pilot_ollama.py evaluate --generation 1
Later generations require an explicit, evidence-based feedback text file.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent  # repository root
MODEL = 'qwen2.5-coder:7b'
API = 'http://localhost:11434/api/'
OPTIONS = dict(temperature=0.7, top_p=0.9, num_ctx=8192, num_predict=4096)


def api(endpoint, data=None):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(API + endpoint, data=payload,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1200) as response:
        return json.load(response)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def extract(text):
    # Strip an enclosing Markdown fence only; never repair generated Python.
    match = re.fullmatch(r'\s*```(?:python|py)?\s*\n(.*?)\n```\s*', text, re.S)
    return match.group(1) + '\n' if match else text


def metadata():
    model = next(m for m in api('tags')['models'] if m['name'] == MODEL)
    return dict(provider='Qwen via local Ollama', model=MODEL,
                model_info=model, ollama=api('version'),
                model_configuration=api('show', {'model': MODEL}),
                queried=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                interface='Ollama local /api/chat', sampling_parameters=OPTIONS,
                tools='none supplied', api_logs=True, fine_tuned=False,
                hardware=platform.platform() + ' ' + platform.machine(),
                reflection_blocks='author-supplied feedback, archived verbatim',
                evaluator_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [SOURCE / n for n in
                              ('creoh_llm_pilot.py', 'creoh_scheduling.py', 'creoh_routing.py')]})


def generate(root, generation, count, feedback=None):
    root.mkdir(parents=True, exist_ok=True)
    candidates = root / 'candidates'
    candidates.mkdir(exist_ok=True)
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    meta = metadata()
    previous = root / 'backbone.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if (old['model_info']['digest'] != meta['model_info']['digest'] or
                old['ollama'] != meta['ollama'] or
                old['evaluator_sha256'] != meta['evaluator_sha256']):
            raise RuntimeError('Model, Ollama version or evaluator changed during run.')
    else:
        save(previous, meta)
    prompt = (SOURCE / 'llm_candidates/prompts/gen1_prompt.txt').read_text()
    if generation > 1:
        if not feedback:
            raise ValueError('Later generations need --feedback PATH based on new results.')
        prompt += '\n\nFEEDBACK FROM PREVIOUS OLLAMA GENERATIONS\n' + Path(feedback).read_text()
    for i in range(1, count + 1):
        name = f'gen{generation}_{i:02d}'
        request_path = logs / f'{name}_request.json'
        response_path = logs / f'{name}_response.json'
        destination = candidates / f'{name}.py'
        request = dict(model=MODEL, stream=False,
                       messages=[{'role': 'user', 'content': prompt}],
                       options={**OPTIONS, 'seed': 1000 * generation + i})
        if request_path.exists():
            if json.loads(request_path.read_text()) != request:
                raise RuntimeError(f'Cannot change existing request {name}.')
            if not response_path.exists():
                raise RuntimeError(f'Incomplete request {name}; inspect logs before retrying.')
            response = json.loads(response_path.read_text())
        else:
            save(request_path, request)
            print(f'Generating {name}...', flush=True)
            response = api('chat', request)
            save(response_path, response)
        code = extract(response['message']['content'])
        if destination.exists() and destination.read_text() != code:
            raise RuntimeError(f'Candidate {name} was modified.')
        destination.write_text(code)
        print(f'Saved {name}; stop reason: {response.get("done_reason")}', flush=True)


def harness(root):
    sys.path.insert(0, str(SOURCE))
    import creoh_llm_pilot as pilot
    pilot.CANDIDATE_DIR = str(root / 'candidates')
    pilot.BACKBONE = json.loads((root / 'backbone.json').read_text())
    return pilot


def smoke(root):
    generate(root, 1, 1)
    pilot = harness(root)
    evaluator = pilot.LLMProposerPilot(seeds=1)
    program = evaluator.admit(pilot.ProgramLoader().load())[0]
    result = dict(program=program.name, admission=program.outcome, detail=program.detail,
                  purpose='connection and integration test only; not manuscript results')
    if program.outcome == 'loaded':
        inst = pilot.sched.ScheduleGenerator(n=40, spread=0.15).generate(0)
        outcome, elapsed = evaluator.evaluate_on(program, inst, None, None,
            dict(orness=0.7, alpha_grid=[0, .25, .5, .75, 1], train_spread=.15))
        result.update(execution=outcome.status, execution_detail=outcome.detail,
                      repaired=outcome.repaired, runtime_seconds=elapsed)
    save(root / 'smoke_result.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['smoke', 'generate', 'evaluate'])
    parser.add_argument('--generation', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--feedback')
    args = parser.parse_args()
    root = HERE / ('smoke_test' if args.action == 'smoke' else 'experiment')
    if args.action == 'smoke':
        smoke(root)
    elif args.action == 'generate':
        if args.generation > 1 and not (root / f'evaluation_gen{args.generation-1}' /
                                      'llm_pilot_metadata.json').exists():
            parser.error('Evaluate the preceding generation first.')
        generate(root, args.generation, 6, args.feedback)
    else:
        expected = {f'gen{g}_{i:02d}.py' for g in range(1, args.generation + 1)
                    for i in range(1, 7)}
        actual = {p.name for p in (root / 'candidates').glob('gen*_*.py')}
        if actual != expected:
            parser.error('Candidate pool does not match the requested cumulative generation.')
        output = root / f'evaluation_gen{args.generation}'
        if output.exists():
            parser.error('Evaluation directory already exists; preserve existing results.')
        output.mkdir()
        pilot = harness(root)
        try:
            pilot.run(seeds=30, outdir=str(output))
        except Exception as exc:
            save(output / 'evaluation_failure.json',
                 dict(error=repr(exc), note='No valid final result; retain all candidates.'))
            raise


if __name__ == '__main__':
    main()
