# C-R-EoH — Collaborative Robust Evolution of Heuristics under Fuzzy Uncertainty

Reproducibility repository for the manuscript *Collaborative Intelligence for
Robust Automatic Heuristic Design under Fuzzy Uncertainty: A Human–AI
Decision-Support Framework for Operations Research* (prepared for submission to
*Annals of Operations Research*, Springer).

**Every number in the paper is computed by the code in this repository, from
fixed seeds, using only `numpy` and `scipy`.** No LLM key is required: the
language-model proposer of the full framework is replaced, for the controlled
experiments, by a reproducible evolutionary search over a parametric heuristic
family. This isolates the effect of the *evaluator* — the object of the
contribution — from the choice of any particular generator backbone.

## What this repository contains

- `creoh_routing.py` — object-oriented core: `TriangularFuzzyNumber` and
  α-cuts, the `ScenarioBuilder` hierarchy (`EndpointBuilder`,
  `LatinHypercubeBuilder`, `StressBuilder`), the `Objective` hierarchy
  (`ExpectedCost`, `OWARisk`, `Fairness`), `ParetoArchive` with hypervolume,
  the routing heuristic family and evaluator, the method/selector classes and
  the statistics (Wilcoxon, Friedman, Cliff's δ, Holm). Running it reproduces
  the routing *contrast* benchmark.
- `creoh_scheduling.py` — the primary benchmark: fuzzy field-service /
  parallel-machine scheduling with a convex overtime penalty, reusing the core
  components. Running it reproduces the scheduling tables, statistics, ablation
  and OOD results.

```bash
pip install -r requirements-lock.txt   # pinned numpy + scipy
python run_all.py                       # regenerates ALL data/ artefacts
# or individually:
python creoh_scheduling.py   # primary benchmark  -> data/scheduling_*.csv  (30 seeds)
python creoh_routing.py      # contrast benchmark -> data/routing_*.csv      (30 seeds)
python test_smoke.py         # minimal correctness checks
```

**Notes.** Archive dominance uses the full 3 objectives `(f1,f2,f3)`; the reported
hypervolume is the 2D cost–risk indicator on `(f1,f2)` (fairness enters
selection, not the HV indicator). The candidate proposer is an LLM-free,
reproducible search, so the reported effects are properties of the evaluator,
not of any backbone.

## Data (raw inputs and outputs)

```
data/
├── scheduling_main.csv       per-method aggregated metrics (mean ± 95% CI)
├── scheduling_stats.csv      Wilcoxon (raw + Holm) and Cliff's δ
├── scheduling_ablation.csv   component ablation
├── scheduling_raw_runs.csv   RAW per-seed, per-method outputs including the
│                             full per-scenario cost vector of each selected solution
├── scheduling_metadata.json  full protocol + headline numbers
├── routing_contrast.csv      routing benchmark (near-null effect, honest)
├── routing_raw_runs.csv      raw per-seed routing outputs
├── routing_stats.csv         routing statistics
└── llm_backbone.csv          deployment-time proposer template (relative,
                              model-agnostic cost units; not part of the
                              computed evaluator evidence)
```

## Headline result (scheduling, 30 seeds, delta=0.15, orness=0.70)

The fuzzy multi-objective evaluator vs. deterministic (nominal) AHD:

| Metric | Effect | p (Holm) | Cliff's delta |
|---|---|---|---|
| P95 tail-cost reduction | -15.0% | <1e-3 | -0.61 |
| OWA risk reduction | -12.2% | <1e-3 | - |
| Fairness (MAD) reduction | -47.9% | - | - |
| OOD degradation | 66% -> 46% | <1e-3 | -0.66 |
| Hypervolume gain | +138% | <1e-3 | +0.16 |
| Nominal-cost premium (price of robustness) | +8.8% | - | - |
| Expected-cost change | -1.7% | - | -0.11 (ns) |

On the routing contrast the same machinery yields a negligible tail change:
minimizing nominal travel cost already absorbs most of the
available robustness. The scheduling-vs-routing contrast is deliberate — it
identifies *when* a robust evaluator earns its cost.

## License

MIT. See `LICENSE`.
