"""Strong OR baselines for the C-R-EoH evaluator study (revision).

The first submission compared the fuzzy multi-objective evaluator against a
classical construction rule, a deterministic (nominal-scalar) AHD loop and a
multi-objective AHD loop without fuzzy risk.  Reviewers observed, correctly,
that a contribution addressed to an operations-research readership must also be
measured against the established ways of *selecting a robust decision* once a
candidate set is available.  This module adds six such selectors, all of them
standard in the OR literature and all of them operating on the **same shared
candidate pool** as every other method, so that the comparison continues to
isolate the selection criterion rather than the search:

============================  ==================================================
Selector                      Criterion applied to the scenario-cost vector
============================  ==================================================
``MinMaxRobustMethod``        Wald min-max: minimise the worst scenario cost
                              (Ben-Tal, El Ghaoui and Nemirovski, 2009).
``MinMaxRegretMethod``        Savage min-max regret against the per-scenario
                              best achievable cost (Kouvelis and Yu, 1997).
``BudgetedRobustMethod``      Bertsimas-Sim budgeted uncertainty on the zone
                              deviations: at most ``Gamma`` zones take their
                              pessimistic bound simultaneously; the worst such
                              case is minimised (Bertsimas and Sim, 2004).
``SAACVaRMethod``             Sample-average approximation of the stochastic
                              programme with a CVaR_beta tail objective
                              (Rockafellar and Uryasev, 2000; Shapiro et al.).
``IraceLikeMethod``           Racing selector over the fixed candidate pool
                              with Friedman-based elimination on the training
                              scenarios, in the style of irace
                              (Lopez-Ibanez et al., 2016); no configuration
                              sampling.
``HyperHeuristicMethod``      Choice-function selection over the fixed pool of
                              low-level heuristics, updated per scenario block
                              (Burke et al., 2013).
============================  ==================================================

Two properties are essential to the fairness of the comparison and are enforced
by construction:

1. every selector receives the identical candidate pool produced by the shared
   reproducible proposer, so no method benefits from a better search;
2. the robust and stochastic selectors are given a *training* scenario subset
   and are scored on the held-out remainder plus the out-of-distribution
   ensemble, so that none of them is evaluated on the very scenarios used to
   pick its solution.  The same split is applied to C-R-EoH.
"""
from __future__ import annotations

import csv
import json
import math
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

from itertools import combinations

from creoh_routing import (ExpectedCost, OWARisk, ScenarioStability, FitnessVector,
                           ParetoArchive, p95, cliffs_delta, holm, dial_scores,
                           tail_count)
import creoh_scheduling as sched


# ==========================================================================
# Scenario split: selection sees only the training half
# ==========================================================================
class ScenarioSplit:
    """Deterministic train/holdout partition of the scenario ensemble."""

    def __init__(self, n_scenarios: int, seed: int, train_frac: float = 0.5):
        rng = np.random.default_rng(seed + 20260914)
        idx = rng.permutation(n_scenarios)
        cut = int(round(train_frac * n_scenarios))
        self.train_idx = np.sort(idx[:cut])
        self.test_idx = np.sort(idx[cut:])


# ==========================================================================
# Selector hierarchy
# ==========================================================================
class Selector(ABC):
    """Chooses one candidate from a shared pool using its own OR criterion."""

    name: str
    #: does the selector return a Pareto archive (portfolio) or a single point?
    portfolio = False

    def __init__(self, split: ScenarioSplit | None = None):
        self.split = split

    def _train(self, c) -> np.ndarray:
        return c.train if self.split is None else c.train[self.split.train_idx]

    @abstractmethod
    def select(self, pool: list): ...

    def archive(self, pool, chosen):
        arch = ParetoArchive()
        arch.add(chosen.fitness(ExpectedCost(), OWARisk(), ScenarioStability()), chosen)
        return arch


class MinMaxRobustMethod(Selector):
    """Wald min-max: the candidate with the lowest worst-case scenario cost."""

    name = "Robust min-max (Wald)"

    def select(self, pool):
        best = min(pool, key=lambda c: self._train(c).max())
        return best, self.archive(pool, best)


class MinMaxRegretMethod(Selector):
    """Savage min-max regret against the per-scenario best achievable cost."""

    name = "Min-max regret (Savage)"

    def select(self, pool):
        C = np.array([self._train(c) for c in pool])       # (P, M_train)
        best_per_scenario = C.min(axis=0)
        regret = (C - best_per_scenario[None, :]).max(axis=1)
        best = pool[int(np.argmin(regret))]
        return best, self.archive(pool, best)


class BudgetedRobustMethod(Selector):
    """Bertsimas-Sim budgeted uncertainty on the congestion-zone deviations.

    Each zone ``g`` may deviate from its modal value to the pessimistic
    alpha-cut bound (shock ``+1``, i.e. factor ``1 + nu_g``); at most ``Gamma``
    zones deviate simultaneously.  Because the scheduling cost is non-decreasing
    in every task duration, the worst case over the budgeted set is attained
    with exactly ``Gamma`` zones at their bound, so the uncertainty set is the
    ``C(Z, Gamma)`` vertex scenarios.  The selector minimises the worst cost
    over that set.  ``Gamma = 0`` recovers nominal selection and
    ``Gamma = Z`` the all-zones-pessimistic worst case.  The vertex scenarios
    are built by the driver from the instance and attached to each candidate
    as ``budget_costs``; they do not use the sampled ensemble.
    """

    def __init__(self, split=None, gamma: int = 2):
        super().__init__(split)
        self.gamma = gamma
        self.name = f"Budgeted robust ($\\Gamma$={gamma} zones)"

    @staticmethod
    def vertex_scenarios(inst, gamma: int) -> np.ndarray:
        Z = len(inst.zone_vol)
        mats = []
        for sub in combinations(range(Z), gamma):
            shocks = np.zeros(Z); shocks[list(sub)] = 1.0
            mats.append(inst.proc * (1 + inst.vol * shocks[inst.zone]))
        return np.array(mats)

    def select(self, pool):
        best = min(pool, key=lambda c: float(np.max(c.budget_costs)))
        return best, self.archive(pool, best)


class SAACVaRMethod(Selector):
    """Sample-average approximation with a CVaR tail objective.

    Minimises ``(1-lam) * E[c] + lam * CVaR_beta[c]`` over the training
    scenarios, the canonical mean-risk stochastic programming formulation.
    Unlike C-R-EoH it produces a single point, has no stability term and no
    archive, so the comparison isolates what the Pareto portfolio and the
    dispersion objective add on top of a mean-CVaR criterion.
    """

    def __init__(self, split=None, beta: float = 0.9, lam: float = 0.5):
        super().__init__(split)
        self.beta, self.lam = beta, lam
        self.name = f"SAA mean-CVaR ($\\beta$={beta:.2f})"

    def _cvar(self, v: np.ndarray) -> float:
        k = tail_count(self.beta, len(v))
        return float(np.sort(v)[::-1][:k].mean())

    def select(self, pool):
        def score(c):
            v = self._train(c)
            return (1 - self.lam) * v.mean() + self.lam * self._cvar(v)
        best = min(pool, key=score)
        return best, self.archive(pool, best)


class IraceLikeMethod(Selector):
    """Iterated racing (irace-style) over the shared candidate pool.

    Candidates race on a growing prefix of the training scenarios; after each
    block a Friedman test on the per-scenario ranks eliminates the candidates
    whose mean rank exceeds the surviving median, until a single configuration
    survives or the training budget is exhausted.  This is the standard
    algorithm-configuration answer to "just pick the best-performing
    heuristic", and it is deliberately given the *same* evaluation budget as
    the fuzzy evaluator.
    """

    def __init__(self, split=None, block: int = 6, min_alive: int = 2):
        super().__init__(split)
        self.block, self.min_alive = block, min_alive
        self.name = "Iterated racing (irace-style)"

    def select(self, pool):
        C = np.array([self._train(c) for c in pool])       # (P, M)
        alive = np.arange(len(pool))
        m = C.shape[1]
        pos = 0
        while pos < m and len(alive) > self.min_alive:
            pos = min(m, pos + self.block)
            sub = C[np.ix_(alive, np.arange(pos))]
            ranks = np.apply_along_axis(rankdata, 0, sub)   # rank within scenario
            mean_rank = ranks.mean(axis=1)
            if pos >= 2 * self.block:
                try:
                    _, pval = friedmanchisquare(*[sub[i] for i in range(len(alive))]) \
                        if len(alive) > 2 else (None, 1.0)
                except ValueError:
                    pval = 1.0
                if pval is not None and pval < 0.05:
                    keep = mean_rank <= np.median(mean_rank)
                    if keep.sum() >= self.min_alive:
                        alive = alive[keep]
                        continue
            keep = mean_rank <= np.quantile(mean_rank, 0.7)
            if keep.sum() >= self.min_alive:
                alive = alive[keep]
        final = C[alive].mean(axis=1)
        best = pool[int(alive[int(np.argmin(final))])]
        return best, self.archive(pool, best)


class HyperHeuristicMethod(Selector):
    """Choice-function hyper-heuristic over the candidate pool.

    The pool is treated as a set of low-level heuristics.  Training scenarios
    arrive in blocks; after each block the choice function
    ``f = w1 * recent performance + w2 * improvement rate + w3 * elapsed since
    last use`` is updated and the incumbent is switched to the highest-scoring
    heuristic.  The heuristic selected most often over the training stream is
    deployed.  This is the classical selection hyper-heuristic control loop and
    represents the "search over heuristics" alternative to redefining the
    evaluator.
    """

    def __init__(self, split=None, block: int = 4,
                 w=(1.0, 0.5, 0.02), pool_cap: int = 40):
        super().__init__(split)
        self.block, self.w, self.pool_cap = block, w, pool_cap
        self.name = "Choice-function hyper-heuristic"

    def select(self, pool):
        C = np.array([self._train(c) for c in pool])
        # keep the pool_cap best nominal candidates as low-level heuristics
        cap = min(self.pool_cap, len(pool))
        cand = np.argsort(C.mean(axis=1), kind="stable")[:cap]
        C = C[cand]
        P, M = C.shape
        last_used = np.zeros(P)
        f1_ = np.zeros(P); f2_ = np.zeros(P)
        usage = np.zeros(P, dtype=int)
        prev_best = None
        t = 0
        for start in range(0, M, self.block):
            blk = C[:, start:start + self.block]
            perf = -blk.mean(axis=1)
            f1_ = 0.7 * f1_ + 0.3 * perf
            if prev_best is not None:
                f2_ = 0.7 * f2_ + 0.3 * (perf - prev_best)
            score = (self.w[0] * f1_ + self.w[1] * f2_
                     + self.w[2] * (t - last_used))
            k = int(np.argmax(score))
            usage[k] += 1
            last_used[k] = t
            prev_best = perf
            t += 1
        best = pool[int(cand[int(np.argmax(usage))])]
        return best, self.archive(pool, best)


class CREoHSelector(Selector):
    """C-R-EoH under the identical train/holdout protocol (portfolio output)."""

    portfolio = True

    def __init__(self, split=None, orness: float = 0.7):
        super().__init__(split)
        self.orness = orness
        self.name = "C-R-EoH (fuzzy MO evaluator)"

    def select(self, pool):
        f1, f2, f3 = ExpectedCost(), OWARisk(self.orness), ScenarioStability()
        arch = ParetoArchive()
        for c in pool:
            v = self._train(c)
            arch.add(FitnessVector(f1(v), f2(v), f3(v)), c)
        fvs = [fv for fv, _ in arch.items]
        knee = arch.items[int(np.argmin(dial_scores(fvs, self.orness)))][1]
        return knee, arch


class NominalSelector(Selector):
    name = "Deterministic AHD (nominal)"

    def select(self, pool):
        best = min(pool, key=lambda c: c.train[0])
        return best, self.archive(pool, best)


# ==========================================================================
# Experiment driver: held-out evaluation of every selector
# ==========================================================================
def _ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x))


def run(seeds: int = 30, n: int = 40, budget: int = 120, orness: float = 0.7,
        spread: float = 0.15, outdir: str = "data") -> dict:
    """Scheduling benchmark; every selector scored on held-out scenarios."""
    gen = sched.ScheduleGenerator(n=n, spread=spread)
    eb, lb, sb = sched.EndpointBuilder(), sched.LatinHypercubeBuilder(), sched.StressBuilder()
    f1, f2, f3 = ExpectedCost(), OWARisk(orness), ScenarioStability()

    per: dict[str, dict[str, list]] = {}
    rows = []
    for seed in range(seeds):
        inst = gen.generate(seed)
        train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
        ood = sb.build(inst, seed)
        pool = sched.CandidatePool(budget).grow(inst, train, ood, seed)
        split = ScenarioSplit(train.shape[0], seed)
        vertices = BudgetedRobustMethod.vertex_scenarios(inst, 2)
        for c in pool:
            h = sched.DispatchHeuristic(c.genome)
            c.budget_costs = h.cost_vector(inst, vertices)
        selectors = [NominalSelector(split), MinMaxRobustMethod(split),
                     MinMaxRegretMethod(split), BudgetedRobustMethod(split, 2),
                     SAACVaRMethod(split, 0.9, 0.5), IraceLikeMethod(split),
                     HyperHeuristicMethod(split), CREoHSelector(split, orness)]
        a1 = np.array([f1(c.train[split.test_idx]) for c in pool])
        a2 = np.array([f2(c.train[split.test_idx]) for c in pool])
        lo1, hi1, lo2, hi2 = a1.min(), a1.max(), a2.min(), a2.max()

        def _norm(fv):
            return FitnessVector((fv.f1 - lo1) / (hi1 - lo1 + 1e-9),
                                 (fv.f2 - lo2) / (hi2 - lo2 + 1e-9), fv.f3)

        for s in selectors:
            sel, arch = s.select(pool)
            held = sel.train[split.test_idx]        # HELD-OUT scenarios only
            narch = ParetoArchive()
            for _fv, cand in arch.items:
                h = cand.train[split.test_idx]
                narch.add(_norm(FitnessVector(f1(h), f2(h), f3(h))), cand)
            hv = narch.hypervolume((1.05, 1.05))
            d = per.setdefault(s.name, {k: [] for k in
                                        ("f1", "f2", "f3", "p95", "ood",
                                         "nominal", "hv")})
            d["f1"].append(f1(held)); d["f2"].append(f2(held))
            d["f3"].append(f3(held)); d["p95"].append(p95(held))
            d["hv"].append(hv)
            d["ood"].append(100.0 * (sel.ood.mean() - sel.train.mean())
                            / sel.train.mean())
            d["nominal"].append(sel.train[0])
            rows.append(dict(seed=seed, selector=s.name, f1=f1(held), f2=f2(held),
                             f3=f3(held), p95=p95(held), hv=hv, ood=d["ood"][-1],
                             nominal=sel.train[0], pack=sel.genome.pack,
                             buffer=sel.genome.buffer,
                             portfolio_size=len(arch.items)))

    names = list(per)
    base = np.mean(per["Deterministic AHD (nominal)"]["f1"])
    scale = 100.0 / base
    summary = {}
    for nm in names:
        d = per[nm]
        summary[nm] = {k: (float(np.mean(np.array(v) * (1 if k in ("ood", "hv")
                                                        else scale))),
                           float(_ci(np.array(v) * (1 if k in ("ood", "hv")
                                                    else scale))))
                       for k, v in d.items()}

    cr = per["C-R-EoH (fuzzy MO evaluator)"]
    raw_p, stats_rows = {}, []
    for nm in names:
        if nm == "C-R-EoH (fuzzy MO evaluator)":
            continue
        for metric in ("p95", "f2", "f3", "nominal", "hv"):
            try:
                _, p = wilcoxon(cr[metric], per[nm][metric])
            except ValueError:
                p = 1.0
            raw_p[f"{nm}|{metric}"] = float(p)
            stats_rows.append([nm, metric, float(p),
                               cliffs_delta(cr[metric], per[nm][metric])])
    adj = holm(raw_p)
    for r in stats_rows:
        r.insert(3, adj[f"{r[0]}|{r[1]}"])

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "baselines_main.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["selector", "f1_mean", "f1_ci", "f2_mean", "f2_ci",
                    "f3_mean", "f3_ci", "p95_mean", "p95_ci", "nominal_mean",
                    "nominal_ci", "hv_mean", "hv_ci", "ood_mean", "ood_ci"])
        for nm in names:
            s = summary[nm]
            w.writerow([nm] + [f"{v:.3f}" for k in
                               ("f1", "f2", "f3", "p95", "nominal", "hv", "ood")
                               for v in s[k]])
    with open(os.path.join(outdir, "baselines_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["c_r_eoh_vs", "metric", "wilcoxon_p_raw", "wilcoxon_p_holm",
                    "cliffs_delta"])
        for r in stats_rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.3f}"])
    with open(os.path.join(outdir, "baselines_raw_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fr = friedmanchisquare(*[per[nm]["p95"] for nm in names])
    meta = dict(seeds=seeds, held_out_fraction=0.5, orness=orness,
                budget_gamma_zones=2, saa_beta=0.9, saa_lambda=0.5,
                friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue),
                summary={nm: {k: [round(v[0], 3), round(v[1], 3)]
                              for k, v in s.items()} for nm, s in summary.items()},
                stats=[[r[0], r[1], r[2], r[3], round(r[4], 3)] for r in stats_rows])
    with open(os.path.join(outdir, "baselines_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"# strong baselines, held-out scenarios, {seeds} seeds")
    print(f"{'selector':34s} {'f1':>8} {'f2(OWA)':>9} {'f3':>7} {'P95':>8} "
          f"{'nom':>8} {'HV':>6} {'OOD%':>7}")
    for nm in names:
        s = summary[nm]
        print(f"{nm:34s} {s['f1'][0]:8.2f} {s['f2'][0]:9.2f} {s['f3'][0]:7.2f} "
              f"{s['p95'][0]:8.2f} {s['nominal'][0]:8.2f} {s['hv'][0]:6.3f} "
              f"{s['ood'][0]:7.2f}")
    print("\nC-R-EoH vs each baseline (Holm-adjusted Wilcoxon):")
    for r in stats_rows:
        print(f"  {r[0]:34s} {r[1]:5s} p={r[3]:.3e} delta={r[4]:+.3f}")
    return meta


if __name__ == "__main__":
    run(outdir="data")
