"""Ablation at the statistical standard of the main tables (revision).

The first submission reported the component ablation as bare point estimates.
This module re-runs the identical ablation over the same R = 30 seeds of the
primary scheduling benchmark and reports, for every variant and every metric,
the mean, the 95% confidence interval over seeds, a paired Wilcoxon test
against the full configuration with Holm correction, and Cliff's delta.  It
also states explicitly which instance family the ablation is run on (the
primary synthetic fuzzy field-service scheduling family, the same 30 instances
as the main scheduling table), and adds two variants that were missing:
removal of the Pareto archive and removal of the alpha-cut ensemble in favour
of the modal instance alone.
"""
from __future__ import annotations

import csv
import json
import math
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import wilcoxon

from creoh_routing import (ExpectedCost, OWARisk, ScenarioStability, FitnessVector,
                           ParetoArchive, p95, cliffs_delta, holm, dial_scores)
import creoh_scheduling as sched


class AblationVariant(ABC):
    """One design element removed from the full C-R-EoH configuration.

    The full configuration is *exactly* the method reported in every other
    table: archive dominance on ``(f1, f2, f3)`` followed by the dial rule
    ``(1-theta) n1 + theta n2`` over the archive.  Removing the OWA objective
    drops ``f2`` from dominance and from the score; removing the stability
    objective drops ``f3`` from dominance, which is where it acts.
    """

    name: str
    #: components active in this variant, for the paper's ablation table
    owa = True
    fairness = True
    buffer = True
    archive = True
    ensemble = True

    def __init__(self, orness: float = 0.7):
        self.orness = orness
        self.f1, self.f2, self.f3 = ExpectedCost(), OWARisk(orness), ScenarioStability()

    def eligible(self, pool):
        """Candidate programs the variant is allowed to select from."""
        if self.buffer:
            return pool
        return [c for c in pool if c.genome.buffer < 0.05] or pool

    def objectives(self, c) -> FitnessVector:
        if not self.ensemble:                 # modal instance only
            v = float(c.train[0])
            return FitnessVector(v, v, 0.0)
        v = c.train
        return FitnessVector(self.f1(v), self.f2(v) if self.owa else 0.0,
                             self.f3(v) if self.fairness else 0.0)

    @abstractmethod
    def select(self, pool):
        """Return ``(selected_candidate, portfolio)``."""


class ScalarisedVariant(AblationVariant):
    """Shared implementation of the full method and its ablations.

    With ``archive=True`` the dial rule is applied to the non-dominated
    archive (as in the full method); with ``archive=False`` the whole pool is
    scalarised directly and no portfolio is kept.
    """

    def select(self, pool):
        cands = self.eligible(pool)
        fvs = [self.objectives(c) for c in cands]
        if self.archive:
            arch = ParetoArchive()
            for fv, c in zip(fvs, cands):
                arch.add(fv, c)
            cands = [c for _fv, c in arch.items]
            fvs = [fv for fv, _c in arch.items]
            portfolio = cands
        else:
            portfolio = None
        score = dial_scores(fvs, self.orness, use_tail=self.owa and self.ensemble)
        chosen = cands[int(np.argmin(score))]
        return chosen, (portfolio if portfolio is not None else [chosen])


def _variant(cls_name, label, **flags):
    return type(cls_name, (ScalarisedVariant,), dict(name=label, **flags))


FullCREoH = _variant("FullCREoH", "Full C-R-EoH")
NoOWA = _variant("NoOWA", "No OWA risk objective", owa=False)
NoFairness = _variant("NoFairness", "No stability objective", fairness=False)
NoBuffer = _variant("NoBuffer", "No robustness buffer", buffer=False)
NoArchive = _variant("NoArchive", "No Pareto archive (scalarised)", archive=False)
NoEnsemble = _variant("NoEnsemble", "No alpha-cut ensemble (modal only)",
                      ensemble=False, owa=False, fairness=False)

VARIANTS = [FullCREoH, NoOWA, NoFairness, NoBuffer, NoArchive, NoEnsemble]
METRICS = ("f1", "f2", "f3", "p95", "hv", "ood", "nominal", "portfolio")


def _ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x))


def run(seeds: int = 30, n: int = 40, budget: int = 120, orness: float = 0.7,
        spread: float = 0.15, outdir: str = "data") -> dict:
    gen = sched.ScheduleGenerator(n=n, spread=spread)
    eb, lb, sb = (sched.EndpointBuilder(), sched.LatinHypercubeBuilder(),
                  sched.StressBuilder())
    f1, f2, f3 = ExpectedCost(), OWARisk(orness), ScenarioStability()
    variants = [V(orness) for V in VARIANTS]
    per = {v.name: {k: [] for k in METRICS} for v in variants}
    rows = []
    identical_to_main = 0

    for seed in range(seeds):
        inst = gen.generate(seed)
        train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
        ood = sb.build(inst, seed)
        pool = sched.CandidatePool(budget).grow(inst, train, ood, seed)
        a1 = np.array([f1(c.train) for c in pool])
        a2 = np.array([f2(c.train) for c in pool])
        lo1, hi1, lo2, hi2 = a1.min(), a1.max(), a2.min(), a2.max()

        def norm(fv):
            return FitnessVector((fv.f1 - lo1) / (hi1 - lo1 + 1e-9),
                                 (fv.f2 - lo2) / (hi2 - lo2 + 1e-9), fv.f3)

        main_sel, _ = sched.CREoHMethod(orness).select(pool)
        for v in variants:
            sel, portfolio = v.select(pool)
            if v.name == "Full C-R-EoH" and sel is main_sel:
                identical_to_main += 1
            arch = ParetoArchive()
            for c in portfolio:
                arch.add(norm(FitnessVector(f1(c.train), f2(c.train),
                                            f3(c.train))), c)
            d = per[v.name]
            d["f1"].append(f1(sel.train)); d["f2"].append(f2(sel.train))
            d["f3"].append(f3(sel.train)); d["p95"].append(p95(sel.train))
            d["hv"].append(arch.hypervolume((1.05, 1.05)))
            d["ood"].append(100.0 * (sel.ood.mean() - sel.train.mean())
                            / sel.train.mean())
            d["nominal"].append(sel.train[0])
            d.setdefault("portfolio", []).append(len(portfolio))
            rows.append(dict(seed=seed, variant=v.name,
                             portfolio_size=len(portfolio),
                             **{k: round(d[k][-1], 4) for k in METRICS},
                             pack=round(sel.genome.pack, 4),
                             buffer=round(sel.genome.buffer, 4)))

    # index units consistent with the main scheduling table: classical mean := 100
    classical = []
    for seed in range(seeds):
        inst = gen.generate(seed)
        train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
        ood = sb.build(inst, seed)
        pool = sched.CandidatePool(budget).grow(inst, train, ood, seed)
        sel, _a = sched.ClassicalMethod().select(pool)
        classical.append(f1(sel.train))
    scale = 100.0 / float(np.mean(classical))

    summary, raw_p, stats_rows = {}, {}, []
    full = per["Full C-R-EoH"]
    for v in variants:
        d = per[v.name]
        summary[v.name] = {}
        for k in METRICS:
            arr = np.array(d[k], float) * (
                1.0 if k in ("hv", "ood", "portfolio") else scale)
            summary[v.name][k] = (float(arr.mean()), float(_ci(arr)))
        if v.name == "Full C-R-EoH":
            continue
        for k in ("p95", "f2", "f3", "hv"):
            try:
                _, p = wilcoxon(d[k], full[k])
            except ValueError:
                p = 1.0
            raw_p[f"{v.name}|{k}"] = float(p)
            stats_rows.append([v.name, k, float(p), cliffs_delta(d[k], full[k])])
    adj = holm(raw_p)
    for r in stats_rows:
        r.insert(3, adj[f"{r[0]}|{r[1]}"])

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "ablation_ci.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "owa", "stability", "buffer", "archive", "ensemble"]
                   + [f"{k}_{s}" for k in METRICS for s in ("mean", "ci")])
        for v in variants:
            s = summary[v.name]
            w.writerow([v.name, v.owa, v.fairness, v.buffer, v.archive, v.ensemble]
                       + [f"{x:.3f}" for k in METRICS for x in s[k]])
    with open(os.path.join(outdir, "ablation_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant_vs_full", "metric", "wilcoxon_p_raw",
                    "wilcoxon_p_holm", "cliffs_delta"])
        for r in stats_rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.3f}"])
    with open(os.path.join(outdir, "ablation_raw_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    meta = dict(seeds=seeds, benchmark="primary synthetic fuzzy field-service "
                "scheduling family, identical instances as the main scheduling table",
                full_identical_to_main_method=f"{identical_to_main}/{seeds}",
                orness=orness, index_scale="classical mean cost := 100",
                summary={k: {m: [round(x, 3) for x in v[m]] for m in METRICS}
                         for k, v in summary.items()},
                stats=[[r[0], r[1], r[2], r[3], round(r[4], 3)] for r in stats_rows])
    with open(os.path.join(outdir, "ablation_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"# ablation, {seeds} seeds, mean +/- 95% CI (index units)")
    print(f"{'variant':36s} {'f1':>14} {'P95':>14} {'stability':>14} {'HV':>13}")
    for v in variants:
        s = summary[v.name]
        print(f"{v.name:36s} "
              f"{s['f1'][0]:8.1f}+-{s['f1'][1]:4.1f} "
              f"{s['p95'][0]:8.1f}+-{s['p95'][1]:4.1f} "
              f"{s['f3'][0]:8.1f}+-{s['f3'][1]:4.1f} "
              f"{s['hv'][0]:7.3f}+-{s['hv'][1]:5.3f} "
              f"|P|={s['portfolio'][0]:5.1f}")
    print("\nvs full configuration (Holm-adjusted Wilcoxon):")
    for r in stats_rows:
        print(f"  {r[0]:36s} {r[1]:4s} p={r[3]:.3e} delta={r[4]:+.3f}")
    return meta


if __name__ == "__main__":
    run(outdir="data")
