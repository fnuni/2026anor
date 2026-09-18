"""Out-of-distribution stress sweep (revision).

The first submission illustrated out-of-distribution behaviour with a figure
whose three stress levels were drawn schematically.  This module computes them:
each method is selected exactly as in the main scheduling table and the selected program is then
re-evaluated on stress ensembles whose skewed-pessimistic shocks are amplified
by increasing factors beyond the training support.  The figure in the paper is
plotted from the macros this module produces.
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

from creoh_routing import ExpectedCost, OWARisk, ScenarioStability
import creoh_scheduling as sched

LEVELS = {"mild": 1.2, "medium": 1.6, "severe": 2.2}


def run(seeds: int = 30, n: int = 40, budget: int = 120, orness: float = 0.7,
        spread: float = 0.15, outdir: str = "data") -> dict:
    gen = sched.ScheduleGenerator(n=n, spread=spread)
    eb, lb = sched.EndpointBuilder(), sched.LatinHypercubeBuilder()
    methods = [sched.ClassicalMethod(), sched.DeterministicMethod(),
               sched.MONoFuzzyMethod(), sched.CREoHMethod(orness)]
    f1, f2, f3 = ExpectedCost(), OWARisk(orness), ScenarioStability()
    acc = {m.name: {k: [] for k in LEVELS} for m in methods}
    rows = []
    for seed in range(seeds):
        inst = gen.generate(seed)
        train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
        ood_sel = sched.StressBuilder().build(inst, seed)     # selection-time OOD
        pool = sched.CandidatePool(budget).grow(inst, train, ood_sel, seed)
        stress = {k: sched.StressBuilder(m=12, extra=v).build(inst, seed)
                  for k, v in LEVELS.items()}
        for m in methods:
            sel, _arch = m.select(pool)
            h = sched.DispatchHeuristic(sel.genome)
            base = float(sel.train.mean())
            for k, scn in stress.items():
                deg = 100.0 * (float(h.cost_vector(inst, scn).mean()) - base) / base
                acc[m.name][k].append(deg)
                rows.append(dict(seed=seed, method=m.name, level=k,
                                 amplification=LEVELS[k], degradation=round(deg, 4)))

    def ci(x):
        x = np.asarray(x, float)
        return 1.96 * x.std(ddof=1) / math.sqrt(len(x))

    summary = {m.name: {k: [round(float(np.mean(v)), 3), round(float(ci(v)), 3)]
                        for k, v in d.items()} for m, d in
               ((m, acc[m.name]) for m in methods)}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "stress_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    meta = dict(seeds=seeds, levels=LEVELS, summary=summary)
    with open(os.path.join(outdir, "stress_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"# OOD stress sweep, {seeds} instances (relative degradation, %)")
    print(f"{'method':28s} " + "".join(f"{k:>12s}" for k in LEVELS))
    for m in methods:
        s = summary[m.name]
        print(f"{m.name:28s} " + "".join(f"{s[k][0]:12.1f}" for k in LEVELS))
    return meta


if __name__ == "__main__":
    run(outdir="data")
