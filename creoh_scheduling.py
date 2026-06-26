"""C-R-EoH reproducible pipeline -- field-service / parallel-machine scheduling
under fuzzy task times with a convex overtime penalty (manuscript case CS2).

This is the empirical core where robustness genuinely trades off against
nominal cost: a nominal-optimal schedule packs technicians tight (few shifts,
low nominal cost) but explodes into overtime when fuzzy task times realise
adversely; the fuzzy multi-objective evaluator buys buffer, paying a small
nominal premium for a large reduction in overtime tail risk and inter-scenario
fairness.

Every number written by this file is computed, not assigned.  The LLM proposer
of the full framework is replaced by a reproducible search over a parametric
dispatch-rule family, so the evaluator effect (RQ2/RQ3) is measured without any
proprietary backbone.  The LLM-backbone table (RQ4) requires real API runs and
is not synthesised.
"""
from __future__ import annotations

import csv
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare

from creoh_routing import (TriangularFuzzyNumber, ExpectedCost, OWARisk,
                            Fairness, FitnessVector, ParetoArchive, p95,
                            cliffs_delta, holm)


# --------------------------------------------------------------------------
# Fuzzy scheduling instance
# --------------------------------------------------------------------------
@dataclass
class FuzzyScheduleInstance:
    proc: np.ndarray         # (n,) modal task times
    zone: np.ndarray         # (n,) latent congestion zone of each task
    zone_vol: np.ndarray     # (Z,) volatility per zone
    vol: np.ndarray          # (n,) volatility inherited from the task's zone
    shift: float             # regular shift length
    overtime_rate: float     # convex multiplier on time beyond the shift
    tech_cost: float         # fixed cost of opening one technician/shift

    @property
    def n(self) -> int:
        return len(self.proc)

    def task_fuzzy(self, j: int) -> TriangularFuzzyNumber:
        v = self.vol[j]
        return TriangularFuzzyNumber(self.proc[j] * (1 - 0.5 * v),
                                     self.proc[j], self.proc[j] * (1 + v))


class ScheduleGenerator:
    def __init__(self, n: int = 40, spread: float = 0.15, n_zones: int = 6,
                 shift: float = 100.0, overtime_rate: float = 1.5,
                 tech_cost: float = 40.0):
        self.n, self.spread, self.n_zones = n, spread, n_zones
        self.shift, self.overtime_rate, self.tech_cost = (
            shift, overtime_rate, tech_cost)

    def generate(self, seed: int) -> FuzzyScheduleInstance:
        rng = np.random.default_rng(seed)
        proc = rng.uniform(5, 20, size=self.n)
        zone = rng.integers(0, self.n_zones, size=self.n)
        zone_vol = rng.uniform(0.05, 0.95, size=self.n_zones) * (self.spread / 0.15)
        vol = zone_vol[zone]
        return FuzzyScheduleInstance(proc, zone, zone_vol, vol, self.shift,
                                     self.overtime_rate, self.tech_cost)


# --------------------------------------------------------------------------
# Scenario builders (common-mode zone shocks)
# --------------------------------------------------------------------------
class ScenarioBuilder(ABC):
    def __init__(self, alpha_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)):
        self.alpha_grid = list(alpha_grid)

    @staticmethod
    def _apply(inst: FuzzyScheduleInstance, shocks: np.ndarray) -> np.ndarray:
        return inst.proc * (1 + inst.vol * shocks[inst.zone])

    @abstractmethod
    def build(self, inst: FuzzyScheduleInstance, seed: int) -> np.ndarray:
        """Return (M, n) realised task-time vectors."""


class EndpointBuilder(ScenarioBuilder):
    def __init__(self, alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0), reps: int = 3):
        super().__init__(alpha_grid)
        self.reps = reps

    def build(self, inst, seed):
        rng = np.random.default_rng(seed + 1543)
        Z = len(inst.zone_vol)
        mats = [inst.proc.copy()]
        for a in self.alpha_grid:
            for _ in range(self.reps):
                sign = rng.choice([-0.5, 1.0], size=Z)
                mats.append(self._apply(inst, sign * (1 - a)))
        return np.array(mats)


class LatinHypercubeBuilder(ScenarioBuilder):
    def __init__(self, alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0), m: int = 24):
        super().__init__(alpha_grid)
        self.m = m

    def build(self, inst, seed):
        rng = np.random.default_rng(seed + 7919)
        Z = len(inst.zone_vol)
        cuts = np.linspace(-0.5, 1.0, self.m + 1)
        mats = []
        for k in range(self.m):
            shocks = np.array([rng.uniform(cuts[k], cuts[k + 1]) for _ in range(Z)])
            rng.shuffle(shocks)
            mats.append(self._apply(inst, shocks))
        return np.array(mats)


class StressBuilder(ScenarioBuilder):
    def __init__(self, alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0), m: int = 12,
                 extra: float = 1.6):
        super().__init__(alpha_grid)
        self.m, self.extra = m, extra

    def build(self, inst, seed):
        rng = np.random.default_rng(seed + 104729)
        Z = len(inst.zone_vol)
        return np.array([self._apply(inst, rng.beta(5, 2, size=Z) * self.extra)
                         for _ in range(self.m)])


# --------------------------------------------------------------------------
# Parametric dispatch heuristic (the searched "program")
# --------------------------------------------------------------------------
@dataclass
class Genome:
    pack: float = 1.0       # load-cap fraction of the shift (tightness)
    buffer: float = 0.0     # reserve proportional to task volatility (robustness)
    balance: float = 0.0    # prefer load balancing across technicians

    def mutate(self, rng) -> "Genome":
        return Genome(
            pack=float(np.clip(self.pack + rng.normal(0, 0.10), 0.55, 1.12)),
            buffer=float(np.clip(self.buffer + rng.normal(0, 0.25), 0.0, 1.5)),
            balance=float(np.clip(self.balance + rng.normal(0, 0.2), 0.0, 1.0)),
        )


class DispatchHeuristic:
    def __init__(self, genome: Genome):
        self.g = genome

    def assign(self, inst: FuzzyScheduleInstance) -> list[list[int]]:
        cap = inst.shift * self.g.pack
        order = np.argsort(-inst.proc)
        machines: list[list[int]] = []
        loads: list[float] = []
        for j in order:
            size = inst.proc[j] * (1 + self.g.buffer * inst.vol[j])  # robustness reserve
            best, best_slack = -1, -1.0
            for k, ld in enumerate(loads):
                if ld + size <= cap:
                    slack = cap - (ld + size)
                    # balance gene -> pick emptiest; else first-fit (tighter)
                    score = -ld if self.g.balance > 0.5 else slack
                    if best == -1 or score > best_slack:
                        best, best_slack = k, score
            if best == -1:
                machines.append([j]); loads.append(inst.proc[j] * (1 + self.g.buffer * inst.vol[j]))
            else:
                machines[best].append(j); loads[best] += size
        return machines

    def cost_vector(self, inst, scenarios: np.ndarray) -> np.ndarray:
        machines = self.assign(inst)
        out = np.empty(len(scenarios))
        for s, pt in enumerate(scenarios):
            tot = 0.0
            for mach in machines:
                load = pt[mach].sum()
                tot += inst.tech_cost + load + inst.overtime_rate * max(0.0, load - inst.shift)
            out[s] = tot
        return out


# --------------------------------------------------------------------------
# Candidate pool + methods (evaluator-as-specification)
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    genome: Genome
    train: np.ndarray
    ood: np.ndarray

    def fitness(self, f1, f2, f3) -> FitnessVector:
        return FitnessVector(f1(self.train), f2(self.train), f3(self.train))


class CandidatePool:
    def __init__(self, budget: int = 120):
        self.budget = budget

    def grow(self, inst, train, ood, seed) -> list[Candidate]:
        rng = np.random.default_rng(seed + 31)
        pool, genomes = [], [Genome()]
        for _ in range(9):
            genomes.append(Genome().mutate(rng))
        for b in range(self.budget):
            g = genomes[b] if b < len(genomes) else pool[rng.integers(0, len(pool))].genome.mutate(rng)
            h = DispatchHeuristic(g)
            pool.append(Candidate(g, h.cost_vector(inst, train), h.cost_vector(inst, ood)))
        return pool


class Method(ABC):
    name: str

    @abstractmethod
    def select(self, pool): ...


class ClassicalMethod(Method):
    name = "Classical heuristic"   # textbook tight LPT packing (pack=1.0)

    def select(self, pool):
        cand = min(pool, key=lambda c: abs(c.genome.pack - 1.0) + c.genome.buffer)
        arch = ParetoArchive()
        arch.add(cand.fitness(ExpectedCost(), OWARisk(), Fairness()), cand)
        return cand, arch


class DeterministicMethod(Method):
    name = "Deterministic AHD"  # nominal evaluator, blind to fuzzy tail

    def select(self, pool):
        best = min(pool, key=lambda c: c.train[0])      # modal cost only
        arch = ParetoArchive()
        arch.add(best.fitness(ExpectedCost(), OWARisk(), Fairness()), best)
        return best, arch


class MONoFuzzyMethod(Method):
    name = "MO AHD (no fuzzy risk)"

    def select(self, pool):
        arch = ParetoArchive()
        for c in pool:
            arch.add(FitnessVector(c.train[0], c.genome.balance, c.genome.pack), c)
        knee = min(arch.items, key=lambda it: it[0].f1)[1]
        return knee, arch


class CREoHMethod(Method):
    name = "C-R-EoH"

    def __init__(self, orness: float = 0.7):
        self.orness = orness
        self.f1, self.f2, self.f3 = ExpectedCost(), OWARisk(orness), Fairness()

    def select(self, pool):
        arch = ParetoArchive()
        for c in pool:
            arch.add(c.fitness(self.f1, self.f2, self.f3), c)
        f1s = np.array([fv.f1 for fv, _ in arch.items])
        f2s = np.array([fv.f2 for fv, _ in arch.items])
        n1 = (f1s - f1s.min()) / (np.ptp(f1s) + 1e-9)
        n2 = (f2s - f2s.min()) / (np.ptp(f2s) + 1e-9)
        knee = arch.items[int(np.argmin((1 - self.orness) * n1 + self.orness * n2))][1]
        return knee, arch


# --------------------------------------------------------------------------
# Ablation variants (operate on the same pool / objective)
# --------------------------------------------------------------------------
def select_scalarised(pool, orness, use_owa=True, use_fair=True, use_buffer=True):
    f1, f2, f3 = ExpectedCost(), OWARisk(orness), Fairness()
    cands = [c for c in pool if use_buffer or c.genome.buffer < 0.05]
    best, bestscore = None, math.inf
    vals = np.array([[f1(c.train), f2(c.train), f3(c.train)] for c in cands])
    norm = (vals - vals.min(0)) / (np.ptp(vals, 0) + 1e-9)
    for c, nv in zip(cands, norm):
        score = nv[0]
        if use_owa:
            score = 0.3 * nv[0] + 0.7 * nv[1]
        if use_fair:
            score += 0.3 * nv[2]
        if score < bestscore:
            best, bestscore = c, score
    return best


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(seeds: int = 30, n: int = 40, budget: int = 120, orness: float = 0.7,
        spread: float = 0.15, outdir: str = "generated_data"):
    gen = ScheduleGenerator(n=n, spread=spread)
    eb, lb, sb = EndpointBuilder(), LatinHypercubeBuilder(), StressBuilder()
    methods = [ClassicalMethod(), DeterministicMethod(),
               MONoFuzzyMethod(), CREoHMethod(orness)]
    f1, f2, f3 = ExpectedCost(), OWARisk(orness), Fairness()

    per = {m.name: {k: [] for k in ["f1", "f2", "f3", "p95", "hv", "ood"]} for m in methods}
    abl = {k: {kk: [] for kk in ["f1", "p95", "fairness", "hv"]} for k in
           ["Full C-R-EoH", "No OWA risk objective", "No fairness objective",
            "No robustness buffer", "No archive (scalarized)"]}
    raw_rows = []

    for seed in range(seeds):
        inst = gen.generate(seed)
        train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
        ood = sb.build(inst, seed)
        pool = CandidatePool(budget).grow(inst, train, ood, seed)
        a1 = np.array([f1(c.train) for c in pool]); a2 = np.array([f2(c.train) for c in pool])
        lo1, hi1, lo2, hi2 = a1.min(), a1.max(), a2.min(), a2.max()

        def norm(fv):
            return FitnessVector((fv.f1 - lo1) / (hi1 - lo1 + 1e-9),
                                 (fv.f2 - lo2) / (hi2 - lo2 + 1e-9), fv.f3)

        for m in methods:
            sel, arch = m.select(pool)
            narch = ParetoArchive()
            for _fv, p in arch.items:
                narch.add(norm(p.fitness(f1, f2, f3)), p)
            hv = narch.hypervolume((1.05, 1.05))
            ood_deg = 100.0 * (sel.ood.mean() - sel.train.mean()) / sel.train.mean()
            per[m.name]["f1"].append(f1(sel.train)); per[m.name]["f2"].append(f2(sel.train))
            per[m.name]["f3"].append(f3(sel.train)); per[m.name]["p95"].append(p95(sel.train))
            per[m.name]["hv"].append(hv); per[m.name]["ood"].append(ood_deg)
            per[m.name].setdefault("nominal", []).append(sel.train[0])
            raw_rows.append(dict(seed=seed, method=m.name, f1=f1(sel.train),
                                 f2=f2(sel.train), f3=f3(sel.train), p95=p95(sel.train),
                                 hv=hv, ood_degradation=ood_deg, pack=sel.genome.pack,
                                 buffer=sel.genome.buffer,
                                 scenario_costs=";".join(f"{x:.2f}" for x in sel.train)))

        # ablation
        variants = {
            "Full C-R-EoH": dict(use_owa=True, use_fair=True, use_buffer=True),
            "No OWA risk objective": dict(use_owa=False, use_fair=True, use_buffer=True),
            "No fairness objective": dict(use_owa=True, use_fair=False, use_buffer=True),
            "No robustness buffer": dict(use_owa=True, use_fair=True, use_buffer=False),
        }
        for name, kw in variants.items():
            sel = select_scalarised(pool, orness, **kw)
            abl[name]["f1"].append(f1(sel.train)); abl[name]["p95"].append(p95(sel.train))
            abl[name]["fairness"].append(f3(sel.train))
            narch = ParetoArchive(); narch.add(norm(sel.fitness(f1, f2, f3)), sel)
            abl[name]["hv"].append(narch.hypervolume((1.05, 1.05)))
        sc = min(pool, key=lambda c: 0.3 * f1(c.train) + 0.7 * f2(c.train))  # no archive
        abl["No archive (scalarized)"]["f1"].append(f1(sc.train))
        abl["No archive (scalarized)"]["p95"].append(p95(sc.train))
        abl["No archive (scalarized)"]["fairness"].append(f3(sc.train))
        nn = ParetoArchive(); nn.add(norm(sc.fitness(f1, f2, f3)), sc)
        abl["No archive (scalarized)"]["hv"].append(nn.hypervolume((1.05, 1.05)))

    # normalise so Classical mean cost == 100 index units
    base = np.mean(per["Classical heuristic"]["f1"]); scale = 100.0 / base
    ci = lambda x: 1.96 * np.std(x, ddof=1) / math.sqrt(len(x))

    summary = {}
    for m in methods:
        d = per[m.name]
        summary[m.name] = dict(
            f1=(np.mean(d["f1"]) * scale, ci(np.array(d["f1"]) * scale)),
            f2=(np.mean(d["f2"]) * scale, ci(np.array(d["f2"]) * scale)),
            f3=(np.mean(d["f3"]) * scale, ci(np.array(d["f3"]) * scale)),
            p95=(np.mean(d["p95"]) * scale, ci(np.array(d["p95"]) * scale)),
            hv=(np.mean(d["hv"]), ci(d["hv"])), ood=(np.mean(d["ood"]), ci(d["ood"])))

    det, mo, cr = per["Deterministic AHD"], per["MO AHD (no fuzzy risk)"], per["C-R-EoH"]
    raw_p, stats_rows = {}, []

    def add_stat(comp, metric, a, b):
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = 1.0
        raw_p[f"{comp}|{metric}"] = p
        stats_rows.append([comp, metric, p, cliffs_delta(a, b)])

    add_stat("C-R-EoH vs deterministic AHD", "P95", cr["p95"], det["p95"])
    add_stat("C-R-EoH vs deterministic AHD", "OOD degradation", cr["ood"], det["ood"])
    add_stat("C-R-EoH vs MO AHD", "Hypervolume", cr["hv"], mo["hv"])
    add_stat("C-R-EoH vs deterministic AHD", "Mean cost", cr["f1"], det["f1"])
    adj = holm(raw_p)
    for r in stats_rows:
        r.insert(3, adj[f"{r[0]}|{r[1]}"])

    fr = friedmanchisquare(per["Classical heuristic"]["p95"], det["p95"], mo["p95"], cr["p95"])

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "scheduling_main.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "f1_mean", "f1_ci", "f2_mean", "f2_ci", "f3_mean",
                    "f3_ci", "p95_mean", "p95_ci", "hv_mean", "hv_ci", "ood_mean", "ood_ci"])
        for m in methods:
            s = summary[m.name]
            w.writerow([m.name] + [f"{v:.2f}" for pr in (s["f1"], s["f2"], s["f3"], s["p95"]) for v in pr]
                       + [f"{s['hv'][0]:.3f}", f"{s['hv'][1]:.3f}", f"{s['ood'][0]:.2f}", f"{s['ood'][1]:.2f}"])

    with open(os.path.join(outdir, "scheduling_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "metric", "wilcoxon_p_raw", "wilcoxon_p_holm", "cliffs_delta"])
        for r in stats_rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.3f}"])

    with open(os.path.join(outdir, "scheduling_ablation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "f1", "p95", "fairness", "hv"])
        for name, d in abl.items():
            w.writerow([name, f"{np.mean(d['f1'])*scale:.2f}", f"{np.mean(d['p95'])*scale:.2f}",
                        f"{np.mean(d['fairness'])*scale:.2f}", f"{np.mean(d['hv']):.3f}"])

    with open(os.path.join(outdir, "scheduling_raw_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
        w.writeheader(); w.writerows(raw_rows)

    det_p95, cr_p95 = summary["Deterministic AHD"]["p95"][0], summary["C-R-EoH"]["p95"][0]
    det_f1, cr_f1 = summary["Deterministic AHD"]["f1"][0], summary["C-R-EoH"]["f1"][0]
    det_f3, cr_f3 = summary["Deterministic AHD"]["f3"][0], summary["C-R-EoH"]["f3"][0]
    det_nom = np.mean(per["Deterministic AHD"]["nominal"]) * scale
    cr_nom = np.mean(per["C-R-EoH"]["nominal"]) * scale
    headline = dict(seeds=seeds, n_tasks=n, budget=budget, orness=orness,
                    spread=spread, alpha_grid=[0, .25, .5, .75, 1], shift=gen.shift,
                    overtime_rate=gen.overtime_rate, tech_cost=gen.tech_cost,
                    p95_reduction_pct=round(100 * (det_p95 - cr_p95) / det_p95, 2),
                    nominal_premium_pct=round(100 * (cr_nom - det_nom) / det_nom, 2),
                    expected_cost_change_pct=round(100 * (cr_f1 - det_f1) / det_f1, 2),
                    fairness_reduction_pct=round(100 * (det_f3 - cr_f3) / det_f3, 2),
                    owa_reduction_pct=round(100 * (summary["Deterministic AHD"]["f2"][0]
                                            - summary["C-R-EoH"]["f2"][0])
                                            / summary["Deterministic AHD"]["f2"][0], 2),
                    hv_gain_pct=round(100 * (summary["C-R-EoH"]["hv"][0]
                                      - summary["Deterministic AHD"]["hv"][0])
                                      / summary["Deterministic AHD"]["hv"][0], 2),
                    friedman_chi2=round(float(fr.statistic), 3),
                    friedman_p=float(fr.pvalue))
    with open(os.path.join(outdir, "scheduling_metadata.json"), "w") as fh:
        json.dump(headline, fh, indent=2)

    print(f"# scheduling  seeds={seeds} n={n} spread={spread} budget={budget}")
    print(f"{'method':32s} {'f1':>7} {'f2(OWA)':>8} {'f3':>7} {'P95':>8} {'HV':>6} {'OOD%':>7}")
    for m in methods:
        s = summary[m.name]
        print(f"{m.name:32s} {s['f1'][0]:7.2f} {s['f2'][0]:8.2f} {s['f3'][0]:7.2f} "
              f"{s['p95'][0]:8.2f} {s['hv'][0]:6.3f} {s['ood'][0]:7.2f}")
    print(f"\nP95 reduction vs det.  : {headline['p95_reduction_pct']:5.1f}%")
    print(f"OWA risk reduction     : {headline['owa_reduction_pct']:5.1f}%")
    print(f"Fairness reduction     : {headline['fairness_reduction_pct']:5.1f}%")
    print(f"Nominal premium (RQ3)  : {headline['nominal_premium_pct']:+5.1f}%")
    print(f"Expected-cost change   : {headline['expected_cost_change_pct']:+5.1f}%")
    print(f"Hypervolume gain        : {headline['hv_gain_pct']:5.1f}%")
    print("Statistics (Holm):")
    for r in stats_rows:
        print(f"  {r[0]:38s} {r[1]:16s} p_holm={r[3]:.4f} delta={r[4]:+.3f}")
    print(f"Friedman P95 chi2={fr.statistic:.2f} p={fr.pvalue:.3g}")
    return headline


if __name__ == "__main__":
    run(outdir="data")
