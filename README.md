# C-R-EoH — Evaluator-Centred Collaborative Intelligence for Robust AHD under Fuzzy Uncertainty

Reproducibility repository for the manuscript *Evaluator-Centred Collaborative
Intelligence for Robust Automatic Heuristic Design under Fuzzy Uncertainty*
(revision of ANOR-D-26-02279, Annals of Operations Research, Special Issue on
Collaborative Intelligence in Operations Research).

**Benchmark results are computed by the released evaluators and typeset from generated macros. Rehearsal records are generated from stated assumptions and are not empirical findings.** `run_all.py` regenerates the
deterministic artefacts in `data/`, `results_macros.tex`, and the scripted
planner-rehearsal records and `planner_rehearsal_macros.tex`, plus the computed Figure 2 in `interface_example.tex`. Runtime values are hardware-specific
and are regenerated only when that stage is included.

```bash
pip install -r requirements-lock.txt   # pinned numpy + scipy
python run_all.py                      # everything (15-35 min, hardware-dependent)
python run_all.py --skip-runtime       # deterministic results; keep reference timings
python run_all.py --fast               # skip the public-benchmark and theory sweeps
python test_smoke.py                   # correctness checks, incl. sandbox probes
```

No API key and no network access are required. The language-model pilot
re-executes the programs released verbatim in `llm_candidates/`; it does not
re-query a model.

## Modules

| Module | Manuscript section | What it does |
| --- | --- | --- |
| `creoh_routing.py` | 4, 6 | Object-oriented core: `TriangularFuzzyNumber` and α-cuts, the `ScenarioBuilder` hierarchy (`EndpointBuilder`, `LatinHypercubeBuilder`, `StressBuilder`), the `Objective` hierarchy (`ExpectedCost`, `OWARisk`, `ScenarioStability`), `ParetoArchive` with an exact 2-D hypervolume, the routing heuristic family and evaluator, the method classes and the statistics. Running it reproduces the routing contrast benchmark. |
| `creoh_scheduling.py` | 8.1 | Primary benchmark: fuzzy field-service / parallel-machine scheduling with a convex overtime penalty, reusing the core components. |
| `creoh_public.py` | 8.3 | Public libraries. `InstanceReader` subclasses for CVRPLIB and Solomon; `FuzzyInjector` subclasses applying the unchanged injection protocol; instance-level experiment drivers. |
| `creoh_baselines.py` | 8.4 | Six OR selectors — Wald min-max, Savage min-max regret, Bertsimas–Sim budgeted robustness on zone deviations, mean-CVaR SAA, a racing selector (irace-style elimination) and a choice-function selector over the fixed pool — on a held-out scenario split. |
| `creoh_mechanism.py` | 8.5 | Cost-term decomposition, crossing analysis, workforce-equity measures, archive size and spread. Answers *why* the tail, the stability and the hypervolume move. |
| `creoh_ablation.py` | 8.6 | Ablation over the same 30 instances with CIs, paired Wilcoxon tests and Cliff's δ; the full configuration is the reported method (checked in `test_smoke.py`). |
| `creoh_runtime.py` | 8.7 | Measured wall-clock overhead, scaling in *n*, *M*, *B*, and the verified structural/scenario cost decomposition. |
| `creoh_llm_pilot.py` | 8.8 | Sandboxed harness for the real-LLM proposer pilot: static policy, execution timeout, deterministic repair policy, evaluation with the unchanged evaluator. |
| `creoh_theory.py` | 5, 8.9 | Numerical verification of the eight propositions, the invariance-breaking study, and the robustness opportunity index with the realised reduction on the same pools. |
| `creoh_stress.py` | Fig. 4 | Out-of-distribution degradation at three amplification levels. |
| `creoh_planner.py` | 9 | Dial response sweep, selection stability, planner regret over six risk profiles, two-way parameter guidance. |
| `generate_planner_rehearsal.py` | 9.4 | Fixed-seed **scripted prototype rehearsal** (feasibility dry-run) of the planner-study protocol. **No human participants**: every record is generated from the per-profile assumptions stated in the script and is flagged `SCRIPTED_REHEARSAL_NOT_HUMAN_DATA`. Writes `data/planner_rehearsal_records.csv`, `data/planner_rehearsal_summary.json` and `planner_rehearsal_macros.tex`, plus the computed Figure 2 in `interface_example.tex`. |
| `make_macros.py` | — | Emits `results_macros.tex` from `data/`. |

## Instances

```
instances/
├── cvrplib/     43 CVRPLIB X-set instances, 100 ≤ n ≤ 300 (Uchoa et al., 2017)
└── solomon/     56 Solomon 100-customer VRPTW source files (Solomon, 1987)
```

Redistributed under their original public terms, unmodified. The Solomon files
serve two roles: capacitated routing instances (time windows relaxed) and the
source of the public field-service scheduling family, whose durations are
`p_i = s_i + κ·d(0,i)` from the published service time and depot distance.
Because time windows are not used, files identical in the fields consumed by
the model are collapsed before inference: 6 distinct routing inputs and 4
distinct scheduling inputs. Counting all 56 files would be pseudoreplication.

## Language-model pilot

```
llm_candidates/
├── prompts/              the three prompts used, as used
├── gen1_*.py (6)         generation 1, from the bare interface
├── gen2_*.py (6)         generation 2, after the first author-written reflection
├── gen3_*.py (6)         generation 3, after the second author-written reflection
└── policy_probes/        AUTHOR-WRITTEN adversarial probes — not LLM output,
                          not part of the pilot; used only by test_smoke.py to
                          prove the rejection, timeout and repair paths fire
```

Backbone: Anthropic Claude Opus 5 (`claude-opus-5`), used in September 2026
through an interactive chat session, not the API. Sampling parameters, tool
availability and session context were not controlled and no API logs exist.
The reflection blocks in `gen2_prompt.txt` and `gen3_prompt.txt` were written
by the authors after inspecting the previous generation and contain design
guidance. The sentence in `gen2_prompt.txt` stating that two candidates
required repair does not match the execution log (no repair fired); the
prompt is released unchanged and the discrepancy is disclosed in the
manuscript. The program files are not edited.

## Data

`data/` holds, for every study, the aggregated table, the statistics with raw
and Holm-adjusted p-values and effect sizes, the raw per-unit records, and a
metadata JSON with the full protocol. `scheduling_raw_runs.csv` and `routing_raw_runs.csv` include rounded per-scenario cost vectors. Other studies release unit-level summaries and the code needed to reconstruct scenario evaluations.

## Notes

- Archive dominance uses all three objectives `(f1,f2,f3)`; the reported
  hypervolume is the 2-D cost–risk indicator on `(f1,f2)`.
- The exact 2-D hypervolume sweep is checked against Monte Carlo integration in `test_smoke.py`. Historical corrections are described in the response letter, not in the standalone manuscript.
- The experimental unit is a distinct model input. On public families,
  duplicates after removal of unused fields are collapsed and scenario seeds
  are averaged within the remaining unit before any test.
- The legacy Python argument name `orness` denotes the dial value θ, which is
  a tail-confidence parameter: the exact empirical-CVaR level is beta = 1 - q/M, with q = max(1, ceil((1-theta)*M)). Its induced OWA orness for uniform
  top-q weights is `[M-(q+1)/2]/(M-1)`; the two values are not identical.
- The dial rule `(1-θ)·ñ1 + θ·ñ2` over the archive (`creoh_routing.dial_scores`)
  is the single selection rule of every C-R-EoH component; stability acts
  through archive dominance.
- The placement rule switches between best fit and worst fit; zones are independently stratified. The tail cutoff is `max(1, ceil((1-theta)*M))`, with a floating-point tolerance.
- `config/experiment_config.yaml` documents the protocol; the modules use the
  same values as class defaults and do not read the file.
- Runtime figures depend on the machine; `data/runtime_metadata.json` records
  the reference platform. Use `--skip-runtime` when reproducing non-timing
  results without intentionally replacing those measurements.

- `--skip-runtime` preserves the standalone runtime-study files; pilot execution times are measured again and may change even under that flag. Scientific outputs are deterministic in the pinned environment.
- The scripted rehearsal checks record structure, summaries and break-even arithmetic only. It does not run a user interface or administer a SUS questionnaire; timing, confidence and SUS fields are assumed placeholders.
- The strong-selector test uses held-out ensemble rows, with repeated modal scenarios potentially present on both sides. Nominal selection uses the modal instance; budgeted selection uses separately constructed uncertainty-set vertices.

- `generate_interface_example.py` regenerates Figure 2 from synthetic scheduling seed 10, using raw cost units and the actual archive and selections. The example was chosen for its multi-point projected frontier and is not an aggregate performance estimate.

- Full manuscript regeneration uses the default `data/` directory beside the scripts. `--outdir` redirects study tables only; macro and rehearsal generators use the package directory. For isolated reproduction, copy the whole package and run from that copy.
- The pilot harness requires POSIX timers on the main thread. It enforces a restricted Python policy and a timeout, not operating-system or memory isolation. Repair rejects non-integer identifiers, records empty-technician removal, and assigns omissions using nominal workload.
