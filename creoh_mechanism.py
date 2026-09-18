"""Mechanism analysis: *why* the fuzzy evaluator moves f3, P95 and HV.

Reviewers asked, for the scheduling benchmark, why the proposed evaluator
improves the dispersion objective and the tail so strongly (main scheduling table) and why
the archive hypervolume rises so far above the multi-objective baseline
(archive hypervolume).  Answering "because it optimises them" is not an explanation.  This
module opens the cost function and the archive and measures the actual
causal chain.

Three analyses:

``CostDecomposition``
    Every scenario cost of the selected schedule is split into its three
    additive terms -- technician opening cost ``K * c_tech``, regular work
    ``sum(p)`` and the convex overtime penalty ``rho * sum(max(0, L_k - S))``.
    Only the third term is convex in the realised loads, so only the third term
    can generate tail risk and dispersion.  Measuring the three terms per method
    shows exactly which one the evaluator removes.

``ConvexityAnalysis``
    Quantifies the mechanism itself.  A marginal unit of load on technician
    ``k`` raises the expected overtime cost by ``rho * P(L_k > S)``, estimated
    on the training ensemble; it exceeds the unit cost of regular work exactly
    when the empirical shift-crossing probability exceeds ``1/rho``.  The
    module reports, per method, the share of technicians in that regime.

``ArchiveGeometry``
    Decomposes the hypervolume gain into its two geometric causes: archive
    *size* (how many mutually non-dominated programs survive) and archive
    *spread* (the extent of the cost-risk range covered).  A multi-objective
    loop that ranks candidates on nominal quantities (modal cost and nominal
    load imbalance) is compared with the fuzzy archive in size, spread and
    cost-risk collinearity.  Collinearity is undefined for archives with fewer
    than three distinct points and is then reported as missing, not as 1.
"""
from __future__ import annotations

import csv
import json
import math
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

from creoh_routing import (ExpectedCost, OWARisk, ScenarioStability, FitnessVector,
                           ParetoArchive, p95, cliffs_delta, holm)
import creoh_scheduling as sched


# ==========================================================================
# Cost decomposition of a selected schedule
# ==========================================================================
class ScheduleAnalyser:
    """Opens one selected schedule and reports structural diagnostics."""

    def __init__(self, inst: sched.FuzzyScheduleInstance):
        self.inst = inst

    def machines(self, genome) -> list[list[int]]:
        return sched.DispatchHeuristic(genome).assign(self.inst)

    def terms(self, machines, scenarios: np.ndarray) -> dict:
        """Additive cost terms per scenario: opening, regular work, overtime."""
        inst = self.inst
        K = len(machines)
        opening = K * inst.tech_cost
        work = np.array([pt[np.concatenate(machines)].sum() for pt in scenarios])
        over = np.array([sum(max(0.0, pt[m].sum() - inst.shift) for m in machines)
                         for pt in scenarios]) * inst.overtime_rate
        total = opening + work + over
        return dict(technicians=K, opening=opening, work=work, overtime=over,
                    total=total)

    def nominal_loads(self, machines) -> np.ndarray:
        return np.array([self.inst.proc[m].sum() for m in machines])

    def crossing_fraction(self, machines, scenarios: np.ndarray) -> float:
        """Share of technicians whose empirical crossing probability exceeds 1/rho."""
        loads = np.array([[pt[m].sum() for m in machines] for pt in scenarios])
        p_cross = (loads > self.inst.shift).mean(axis=0)
        return float((p_cross > 1.0 / self.inst.overtime_rate).mean())

    def workload_fairness(self, machines, scenarios: np.ndarray) -> dict:
        """Workforce-level equity, distinct from inter-scenario stability.

        ``f3`` in the manuscript measures dispersion of *total cost across
        scenarios* (renamed 'scenario stability' in the revision).  Planners
        normally mean by 'fairness' the equity of workload *across
        technicians*.  Both are reported here: Gini and MAD of the per-
        technician realised workload, and the same for realised overtime,
        averaged over the scenario ensemble.
        """
        loads = np.array([[pt[m].sum() for m in machines] for pt in scenarios])
        over = np.maximum(0.0, loads - self.inst.shift)

        def gini(v):
            v = np.sort(np.asarray(v, float))
            k = len(v)
            if k == 0 or v.sum() <= 0:
                return 0.0
            idx = np.arange(1, k + 1)
            return float((2 * (idx * v).sum()) / (k * v.sum()) - (k + 1) / k)

        return dict(
            workload_gini=float(np.mean([gini(r) for r in loads])),
            workload_mad=float(np.mean([np.abs(r - r.mean()).mean() for r in loads])),
            overtime_gini=float(np.mean([gini(r) for r in over])),
            overtime_share=float((over.sum(axis=1) > 0).mean()),
            max_tech_overtime=float(over.max(axis=1).mean()))


# ==========================================================================
# Archive geometry
# ==========================================================================
class ArchiveGeometry:
    """Size and spread decomposition of a Pareto archive in the (f1,f2) plane."""

    @staticmethod
    def describe(arch: ParetoArchive) -> dict:
        pts = np.array(sorted({(fv.f1, fv.f2) for fv, _ in arch.items}))
        if len(pts) == 0:
            return dict(size=0, spread_f1=0.0, spread_f2=0.0,
                        collinearity=float("nan"))
        spread1 = float(np.ptp(pts[:, 0])) if len(pts) > 1 else 0.0
        spread2 = float(np.ptp(pts[:, 1])) if len(pts) > 1 else 0.0
        if len(pts) > 2 and spread1 > 0 and spread2 > 0:
            r = abs(float(np.corrcoef(pts[:, 0], pts[:, 1])[0, 1]))
        else:
            r = float("nan")                     # undefined, not collinear
        return dict(size=len(pts), spread_f1=spread1, spread_f2=spread2,
                    collinearity=r)


# ==========================================================================
# Driver
# ==========================================================================
def _ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x))


def run(seeds: int = 30, n: int = 40, budget: int = 120, orness: float = 0.7,
        spread: float = 0.15, outdir: str = "data") -> dict:
    gen = sched.ScheduleGenerator(n=n, spread=spread)
    eb, lb, sb = (sched.EndpointBuilder(), sched.LatinHypercubeBuilder(),
                  sched.StressBuilder())
    f1, f2, f3 = ExpectedCost(), OWARisk(orness), ScenarioStability()
    methods = [sched.ClassicalMethod(), sched.DeterministicMethod(),
               sched.MONoFuzzyMethod(), sched.CREoHMethod(orness)]
    keys = ("technicians", "opening", "work_mean", "overtime_mean",
            "overtime_p95", "overtime_share_of_tail", "crossing_frac",
            "workload_gini", "workload_mad", "overtime_gini",
            "max_tech_overtime", "arch_size", "arch_spread_f1",
            "arch_spread_f2", "arch_collinearity", "p95", "f3", "hv",
            "mean_buffer", "mean_pack")
    per = {m.name: {k: [] for k in keys} for m in methods}
    rows = []

    for seed in range(seeds):
        inst = gen.generate(seed)
        train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
        ood = sb.build(inst, seed)
        pool = sched.CandidatePool(budget).grow(inst, train, ood, seed)
        an = ScheduleAnalyser(inst)
        a1 = np.array([f1(c.train) for c in pool])
        a2 = np.array([f2(c.train) for c in pool])
        lo1, hi1, lo2, hi2 = a1.min(), a1.max(), a2.min(), a2.max()

        def norm(fv):
            return FitnessVector((fv.f1 - lo1) / (hi1 - lo1 + 1e-9),
                                 (fv.f2 - lo2) / (hi2 - lo2 + 1e-9), fv.f3)

        for m in methods:
            sel, arch = m.select(pool)
            mach = an.machines(sel.genome)
            t = an.terms(mach, train)
            fair = an.workload_fairness(mach, train)
            narch = ParetoArchive()
            for _fv, c in arch.items:
                narch.add(norm(c.fitness(f1, f2, f3)), c)
            geo = ArchiveGeometry.describe(narch)
            tail_idx = t["total"] >= np.percentile(t["total"], 95)
            excess = t["total"][tail_idx].mean() - t["total"].mean()
            over_excess = t["overtime"][tail_idx].mean() - t["overtime"].mean()
            d = per[m.name]
            d["technicians"].append(t["technicians"])
            d["opening"].append(t["opening"])
            d["work_mean"].append(float(t["work"].mean()))
            d["overtime_mean"].append(float(t["overtime"].mean()))
            d["overtime_p95"].append(float(np.percentile(t["overtime"], 95)))
            d["overtime_share_of_tail"].append(
                float(over_excess / excess) if excess > 1e-9 else 0.0)
            d["crossing_frac"].append(an.crossing_fraction(mach, train))
            for k in ("workload_gini", "workload_mad", "overtime_gini",
                      "max_tech_overtime"):
                d[k].append(fair[k])
            d["arch_size"].append(geo["size"])
            d["arch_spread_f1"].append(geo["spread_f1"])
            d["arch_spread_f2"].append(geo["spread_f2"])
            d["arch_collinearity"].append(geo["collinearity"])
            d["p95"].append(p95(sel.train)); d["f3"].append(f3(sel.train))
            d["hv"].append(narch.hypervolume((1.05, 1.05)))
            d["mean_buffer"].append(sel.genome.buffer)
            d["mean_pack"].append(sel.genome.pack)
            rows.append(dict(seed=seed, method=m.name,
                             **{k: round(float(d[k][-1]), 5) for k in keys}))

    def _nan_summary(v):
        a = np.asarray(v, float); a = a[~np.isnan(a)]
        if len(a) == 0:
            return (None, None)
        return (float(a.mean()), float(_ci(a)) if len(a) > 1 else 0.0)

    summary = {m.name: {k: _nan_summary(per[m.name][k]) for k in keys}
               for m in methods}
    defined_coll = {m.name: int(np.sum(~np.isnan(np.asarray(
        per[m.name]["arch_collinearity"], float)))) for m in methods}

    # --- mechanism correlations, pooled over methods and seeds --------------
    allrows = rows
    x_cross = np.array([r["crossing_frac"] for r in allrows])
    y_p95 = np.array([r["p95"] for r in allrows])
    y_over = np.array([r["overtime_p95"] for r in allrows])
    x_size = np.array([r["arch_size"] for r in allrows])
    x_coll = np.array([r["arch_collinearity"] for r in allrows], float)
    y_hv = np.array([r["hv"] for r in allrows])
    ok = ~np.isnan(x_coll)
    x_buf = np.array([r["mean_buffer"] for r in allrows])
    y_f3 = np.array([r["f3"] for r in allrows])
    corr = dict(
        crossing_vs_p95=[round(v, 4) for v in pearsonr(x_cross, y_p95)],
        crossing_vs_overtime_p95=[round(v, 4) for v in pearsonr(x_cross, y_over)],
        buffer_vs_stability=[round(v, 4) for v in spearmanr(x_buf, y_f3)],
        archive_size_vs_hv=[round(v, 4) for v in spearmanr(x_size, y_hv)],
        collinearity_vs_hv=[round(v, 4) for v in spearmanr(x_coll[ok], y_hv[ok])])

    # --- regular work is invariant by construction (Proposition 6) ----------
    work_identity = max(abs(a - b) for m in methods for a, b in
                        zip(per[m.name]["work_mean"], per["C-R-EoH"]["work_mean"]))

    # --- paired tests: schedule terms vs deterministic AHD, archive vs MO ----
    det, cr = per["Deterministic AHD"], per["C-R-EoH"]
    mo = per["MO AHD (no fuzzy risk)"]
    raw_p, stats_rows = {}, []
    comparisons = [(k, det) for k in ("overtime_mean", "overtime_p95",
                                      "technicians", "crossing_frac",
                                      "workload_gini", "overtime_gini",
                                      "max_tech_overtime")] + \
                  [(k, mo) for k in ("arch_size", "arch_spread_f2")]
    for k, ref in comparisons:
        try:
            _, p = wilcoxon(cr[k], ref[k])
        except ValueError:
            p = 1.0
        raw_p[k] = float(p)
        stats_rows.append([k, float(p), cliffs_delta(cr[k], ref[k])])
    adj = holm(raw_p)
    for r in stats_rows:
        r.insert(2, adj[r[0]])

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "mechanism_decomposition.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method"] + [f"{k}_{s}" for k in keys
                                 for s in ("mean", "ci")])
        for m in methods:
            s = summary[m.name]
            w.writerow([m.name] + [("" if x is None else f"{x:.4f}")
                                   for k in keys for x in s[k]])
    with open(os.path.join(outdir, "mechanism_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "wilcoxon_p_raw", "wilcoxon_p_holm",
                    "cliffs_delta"])
        for r in stats_rows:
            w.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.3f}"])
    with open(os.path.join(outdir, "mechanism_raw_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    meta = dict(seeds=seeds, correlations=corr,
                work_identity_max_abs_diff=float(work_identity),
                collinearity_defined_units=defined_coll,
                comparison_reference={"arch_size": "MO AHD (no fuzzy risk)",
                                      "arch_spread_f2": "MO AHD (no fuzzy risk)",
                                      "other": "Deterministic AHD"},
                summary={m: {k: [None if v[0] is None else round(v[0], 4),
                                 None if v[1] is None else round(v[1], 4)]
                             for k, v in s.items()} for m, s in summary.items()},
                stats=[[r[0], r[1], r[2], round(r[3], 3)] for r in stats_rows])
    with open(os.path.join(outdir, "mechanism_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"# mechanism analysis, {seeds} seeds")
    print(f"{'method':26s} {'K':>5} {'work':>8} {'OT mean':>8} {'OT P95':>8} "
          f"{'OT/tail':>8} {'cross':>6} {'wGini':>6} {'|P|':>5} {'coll':>5}")
    for m in methods:
        s = summary[m.name]
        print(f"{m.name:26s} {s['technicians'][0]:5.2f} {s['work_mean'][0]:8.1f} "
              f"{s['overtime_mean'][0]:8.2f} {s['overtime_p95'][0]:8.2f} "
              f"{s['overtime_share_of_tail'][0]:8.2f} {s['crossing_frac'][0]:6.2f} "
              f"{s['workload_gini'][0]:6.3f} {s['arch_size'][0]:5.1f} "
              f"{(s['arch_collinearity'][0] or float('nan')):5.2f}")
    print("\nmechanism correlations (r or rho, p):")
    for k, v in corr.items():
        print(f"  {k:28s} {v[0]:+.3f}  p={v[1]:.2e}")
    print(f"\nregular-work identity, max |diff| = {work_identity:.2e}")
    print("\nC-R-EoH vs deterministic AHD (archive rows: vs MO AHD), Holm:")
    for r in stats_rows:
        print(f"  {r[0]:22s} p={r[2]:.3e} delta={r[3]:+.3f}")
    return meta


if __name__ == "__main__":
    run(outdir="data")
