# Second pilot: Qwen-7B (Qwen2.5-Coder 7B, local Ollama)

Model: Qwen2.5-Coder 7B, Ollama tag `qwen2.5-coder:7b`, quantization `Q4_K_M`,
served by Ollama 0.34.2 on 18 September 2026 (Mac16,6, 36 GiB RAM). The weights
are distributed under Apache 2.0. The model digest, full configuration, Ollama
version and evaluator hashes are recorded in `experiment/backbone.json`.
In the manuscript this pilot is called **Qwen-7B**.

## Protocol

Three generations of six independent requests; every request starts a fresh
conversation. Temperature 0.7, top_p 0.9, num_ctx 8192, num_predict 4096,
seed = 1000 * generation + candidate index. No tools are supplied. The base
prompt is `llm_candidates/prompts/gen1_prompt.txt`, identical to the Claude and
Qwen-27B pilots; generations 2 and 3 append `feedback_gen2.txt` and
`feedback_gen3.txt`. The feedback was drafted by the assisting AI from the actual
execution logs and training summaries; it was not generated automatically by the
evaluator and contains no OOD metric (provenance in `experiment/protocol_note.json`).

Evaluation is cumulative (6, 12 and finally 18 programs on the same 30 instance
seeds); only `experiment/evaluation_gen3/` is the final pool. The same seeds
inform feedback and final evaluation, so the paired statistics describe an
adaptive pilot, not an independent held-out test. Generated code is never
edited: only an enclosing Markdown fence is removed. Invalid candidates are kept,
without selective replacement. A separate one-program connection test is not
part of the 18-candidate pool.

## Outcome

All 18 programs pass static admission; 7 run successfully on every instance.
The 540 execution attempts comprise 210 successes, 210 runtime errors (unknown
parameter keys `ornois`/`ornoess`, a float list index) and 120 invalid outputs
(non-integer task ids, a technician not given as a list). No repair fires.
Within the usable pool, fuzzy selection reduces P95 by 1.2% relative to nominal
selection. This is not a controlled comparison with the Claude or Qwen-27B pilots:
interfaces, sampling control and feedback differ, and hypervolume is normalized
within each pool.

## Files

- `experiment/candidates/`: the 18 programs, including failures.
- `experiment/logs/`: 18 complete requests and responses (prompts, seeds, options).
- `experiment/evaluation_gen1/` ... `evaluation_gen3/`: original cumulative evaluations.
- `feedback_gen2.txt`, `feedback_gen3.txt`: feedback actually used.
- `experiment/backbone.json`, `experiment/protocol_note.json`: provenance and limits.
- `experiment/verification.json`: integrity checks, re-run with `verify_results.py`.
- `pilot_ollama.py`: original generator (requires a running local Ollama).

## Reproduction

From the repository root (no Ollama required):

```bash
python creoh_ollama_pilot.py            # offline replay -> data/ollama_pilot/
python make_ollama_macros.py            # ollama_macros.tex
python ollama_pilot/verify_results.py   # integrity of the recorded run
```

Original generation (requires Ollama, run from this folder):

```bash
python pilot_ollama.py generate --generation 1
python pilot_ollama.py evaluate --generation 1
python pilot_ollama.py generate --generation 2 --feedback feedback_gen2.txt
python pilot_ollama.py evaluate --generation 2
python pilot_ollama.py generate --generation 3 --feedback feedback_gen3.txt
python pilot_ollama.py evaluate --generation 3
```

The execution policy is the static AST check and wall-clock timeout of the
harness, not operating-system isolation.
