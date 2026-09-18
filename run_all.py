#!/usr/bin/env python3
"""Regenerate every released artefact, then the manuscript's number macros.

Deterministic given the fixed seeds. No API key and no network access are
required: the language-model pilot re-executes the generated programs that are
released verbatim in ``llm_candidates/``, it does not re-query a model.

    python run_all.py            # everything
    python run_all.py --fast     # skip the long public-benchmark sweep
    python run_all.py --skip-runtime  # preserve machine-specific timing data
"""
from __future__ import annotations

import argparse
import importlib
import runpy
import time
from dataclasses import dataclass, field


@dataclass
class Stage:
    """One released study, its module and the artefacts it writes."""
    key: str
    module: str
    label: str
    kwargs: dict = field(default_factory=dict)
    slow: bool = False

    def run(self, outdir: str) -> float:
        t0 = time.time()
        importlib.import_module(self.module).run(outdir=outdir, **self.kwargs)
        return time.time() - t0


STAGES = [
    Stage("scheduling", "creoh_scheduling", "primary scheduling benchmark"),
    Stage("routing", "creoh_routing", "routing contrast benchmark",
          dict(seeds=30, n=25, budget=80)),
    Stage("stress", "creoh_stress", "out-of-distribution stress sweep"),
    Stage("baselines", "creoh_baselines", "strong OR selectors (held-out)"),
    Stage("ablation", "creoh_ablation", "ablation with confidence intervals"),
    Stage("mechanism", "creoh_mechanism", "cost and archive mechanism analysis"),
    Stage("runtime", "creoh_runtime", "runtime and scalability"),
    Stage("planner", "creoh_planner", "planner decision analysis"),
    Stage("llm_pilot", "creoh_llm_pilot", "real-LLM proposer pilot"),
    Stage("public", "creoh_public", "public benchmark libraries", slow=True),
    Stage("theory", "creoh_theory", "propositions and opportunity index",
          slow=True),
]


def main(outdir: str = "data", fast: bool = False,
         skip_runtime: bool = False) -> None:
    stages = [s for s in STAGES
              if not (fast and s.slow) and not (skip_runtime and s.key == "runtime")]
    t0 = time.time()
    times = {}
    for i, st in enumerate(stages, start=1):
        print(f"\n[{i}/{len(stages)}] {st.label} ...", flush=True)
        times[st.label] = st.run(outdir)
    print("\n--- timing ---")
    for k, v in times.items():
        print(f"  {k:44s} {v:8.1f}s")
    if not fast:
        print("\n[planner rehearsal] regenerating the scripted rehearsal records (no human participants) ...")
        runpy.run_module("generate_planner_rehearsal", run_name="__main__")
        print("\n[macros] regenerating results_macros.tex ...")
        importlib.import_module("make_macros").main()
    print(f"\ndone in {time.time()-t0:.1f}s; artefacts in {outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--fast", action="store_true",
                    help="skip the public-benchmark and theory sweeps")
    ap.add_argument("--skip-runtime", action="store_true",
                    help="preserve released hardware-specific timing artefacts")
    a = ap.parse_args()
    main(a.outdir, a.fast, a.skip_runtime)
