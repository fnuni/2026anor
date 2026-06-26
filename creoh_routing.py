"""C-R-EoH reproducible pipeline (deterministic core, no LLM required).

Generates REAL data for the central scientific claim of the manuscript:
selecting heuristic programs with a fuzzy multi-objective evaluator
(expected cost / OWA tail risk / inter-scenario fairness) yields more
robust deployed solutions than selecting the same candidate programs with
a deterministic scalar evaluator.

All numbers produced by this file are computed, not assigned.  The LLM
proposer of the full framework is replaced here by a reproducible
evolutionary search over a parametric routing-heuristic family, so that
the evaluator effect (RQ2/RQ3) can be measured without any proprietary
backbone.  The LLM-backbone table (RQ4) requires real API runs and is not
synthesised.
"""
from __future__ import annotations

import csv
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare


# --------------------------------------------------------------------------
# Fuzzy numbers and instances
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TriangularFuzzyNumber:
    lo: float          # optimistic z^l
    mode: float        # modal z^m
    hi: float          # pessimistic z^u

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        return (self.lo + alpha * (self.mode - self.lo),
                self.hi - alpha * (self.hi - self.mode))


@dataclass
class FuzzyInstance:
    coords: np.ndarray            # (n+1, 2), index 0 = depot
    demand: np.ndarray            # (n+1,), demand[0] = 0
    capacity: float
    base: np.ndarray              # (n+1, n+1) modal travel-time matrix
    region: np.ndarray            # (n+1,) latent congestion region of each node
    region_vol: np.ndarray        # (G,) volatility of each region
    arc_vol: np.ndarray           # (n+1, n+1) governing volatility per arc
    arc_region: np.ndarray        # (n+1, n+1) governing region per arc

    @property
    def n(self) -> int:
        return len(self.demand) - 1

    @property
    def spread_hi(self) -> np.ndarray:
        return self.arc_vol

    def arc_fuzzy(self, i: int, j: int) -> TriangularFuzzyNumber:
        t = self.base[i, j]
        v = self.arc_vol[i, j]
        return TriangularFuzzyNumber(t * (1 - 0.5 * v), t, t * (1 + v))


class FuzzyInstanceGenerator:
    """Synthetic Euclidean CVRP with documented triangular fuzzy travel times."""

    def __init__(self, n: int = 25, spread: float = 0.15,
                 regime: str = "regional", n_regions: int = 6):
        self.n = n
        self.spread = spread
        self.regime = regime
        self.n_regions = n_regions

    def generate(self, seed: int) -> FuzzyInstance:
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(self.n + 1, 2))
        coords[0] = [50, 50]
        demand = np.concatenate([[0.0], rng.integers(5, 25, size=self.n).astype(float)])
        capacity = float(np.ceil(demand.sum() / 4.0 / 5.0) * 5.0)  # ~4-6 routes
        diff = coords[:, None, :] - coords[None, :, :]
        base = np.sqrt((diff ** 2).sum(-1))

        # Latent congestion regions: nodes are partitioned by spatial centroids;
        # each region has its own volatility, INDEPENDENT of arc length. Some
        # regions are calm, some highly volatile, so risk is decoupled from cost.
        G = self.n_regions
        centroids = rng.uniform(0, 100, size=(G, 2))
        region = np.argmin(((coords[:, None, :] - centroids[None, :, :]) ** 2)
                           .sum(-1), axis=1)
        region[0] = -1                                   # depot: no region shock
        region_vol = rng.uniform(0.05, 0.85, size=G) * (self.spread / 0.15)
        node_vol = np.where(region < 0, 0.0, region_vol[region])
        # an arc inherits the volatility/region of its more volatile endpoint
        arc_vol = np.maximum(node_vol[:, None], node_vol[None, :])
        gov_i = (node_vol[:, None] >= node_vol[None, :])
        arc_region = np.where(gov_i, region[:, None], region[None, :])
        np.fill_diagonal(arc_vol, 0.0)
        return FuzzyInstance(coords, demand, capacity, base,
                             region, region_vol, arc_vol, arc_region)


# --------------------------------------------------------------------------
# Scenario construction (polymorphic builders)
# --------------------------------------------------------------------------
class ScenarioBuilder(ABC):
    """Builders realise travel-time matrices from common-mode regional shocks:
    a scenario assigns each congestion region a shock that simultaneously
    inflates every arc governed by that region. Tail exposure therefore depends
    on which regions a solution routes through."""

    def __init__(self, alpha_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)):
        self.alpha_grid = list(alpha_grid)

    @staticmethod
    def _apply(inst: FuzzyInstance, shocks: np.ndarray) -> np.ndarray:
        """shocks: (G,) in [-1,1]; arc factor = 1 + arc_vol * shock[arc_region]."""
        ar = inst.arc_region.copy()
        ar[ar < 0] = 0
        z = shocks[ar]
        z[inst.arc_region < 0] = 0.0
        return inst.base * (1 + inst.arc_vol * z)

    @abstractmethod
    def build(self, inst: FuzzyInstance, seed: int) -> np.ndarray:
        """Return array (M, n+1, n+1) of realised travel-time matrices."""


class EndpointBuilder(ScenarioBuilder):
    """Alpha-cut regional extremes: for each alpha level, several scenarios in
    which each region independently takes its lower (-(1-a)) or upper (+(1-a))
    cut bound. Common-mode within a region, independent across regions."""

    def __init__(self, alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0), reps: int = 3):
        super().__init__(alpha_grid)
        self.reps = reps

    def build(self, inst: FuzzyInstance, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 1543)
        G = len(inst.region_vol)
        mats = [inst.base.copy()]                      # modal scenario
        for a in self.alpha_grid:
            for _ in range(self.reps):
                sign = rng.choice([-0.5, 1.0], size=G)   # upper-skewed extremes
                mats.append(self._apply(inst, sign * (1 - a)))
        return np.array(mats)


class LatinHypercubeBuilder(ScenarioBuilder):
    def __init__(self, alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0), m: int = 24):
        super().__init__(alpha_grid)
        self.m = m

    def build(self, inst: FuzzyInstance, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 7919)
        G = len(inst.region_vol)
        cuts = np.linspace(-0.5, 1.0, self.m + 1)       # stratified shock levels
        mats = []
        for k in range(self.m):
            shocks = np.array([rng.uniform(cuts[k], cuts[k + 1]) for _ in range(G)])
            rng.shuffle(shocks)
            mats.append(self._apply(inst, shocks))
        return np.array(mats)


class StressBuilder(ScenarioBuilder):
    """Skewed-pessimistic out-of-distribution ensemble: region shocks are
    upper-biased and amplified beyond the training support."""

    def __init__(self, alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0), m: int = 12,
                 extra: float = 1.6):
        super().__init__(alpha_grid)
        self.m = m
        self.extra = extra

    def build(self, inst: FuzzyInstance, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 104729)
        G = len(inst.region_vol)
        mats = []
        for _ in range(self.m):
            shocks = rng.beta(5, 2, size=G) * self.extra
            mats.append(self._apply(inst, shocks))
        return np.array(mats)


# --------------------------------------------------------------------------
# Parametric routing heuristic (the "program" being searched)
# --------------------------------------------------------------------------
@dataclass
class Genome:
    lam: float = 1.0       # generalised savings shape
    slack: float = 0.0     # reserved capacity fraction (robustness margin)
    balance: float = 0.0   # route-load balancing penalty
    two_opt: int = 1       # intra-route 2-opt passes
    risk: float = 0.0      # aversion to uncertain (high-spread) arcs

    def mutate(self, rng) -> "Genome":
        return Genome(
            lam=float(np.clip(self.lam + rng.normal(0, 0.25), 0.4, 2.2)),
            slack=float(np.clip(self.slack + rng.normal(0, 0.06), 0.0, 0.35)),
            balance=float(np.clip(self.balance + rng.normal(0, 0.15), 0.0, 1.2)),
            two_opt=int(np.clip(self.two_opt + rng.integers(-1, 2), 0, 3)),
            risk=float(np.clip(self.risk + rng.normal(0, 0.5), 0.0, 3.0)),
        )


class RoutingHeuristic:
    """Clarke-Wright savings construction parameterised by a Genome."""

    PER_ROUTE_COST = 6.0   # vehicle dispatch cost -> makes slack a real trade-off

    def __init__(self, genome: Genome):
        self.g = genome

    def build_routes(self, inst: FuzzyInstance) -> list[list[int]]:
        n = inst.n
        D = inst.base
        cap = inst.capacity * (1 - self.g.slack)
        routes = {i: [i] for i in range(1, n + 1)}
        load = {i: inst.demand[i] for i in range(1, n + 1)}
        route_of = {i: i for i in range(1, n + 1)}
        savings = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                # risk gene discourages merges over uncertain (high-spread) arcs
                risk_pen = self.g.risk * inst.spread_hi[i, j] * D[i, j]
                s = D[i, 0] + D[0, j] - self.g.lam * D[i, j] - risk_pen
                savings.append((s, i, j))
        savings.sort(reverse=True)
        for s, i, j in savings:
            ri, rj = route_of[i], route_of[j]
            if ri == rj:
                continue
            Ri, Rj = routes[ri], routes[rj]
            if Ri[-1] != i or Rj[0] != j:
                if Ri[0] == i and Rj[-1] == j:
                    Ri, Rj, ri, rj = Rj, Ri, rj, ri
                elif Ri[-1] == i and Rj[-1] == j:
                    Rj = Rj[::-1]
                elif Ri[0] == i and Rj[0] == j:
                    Ri = Ri[::-1]
                else:
                    continue
            if load[ri] + load[rj] > cap:
                continue
            if self.g.balance > 0:
                merged = load[ri] + load[rj]
                if merged > cap * (1 - 0.18 * self.g.balance):
                    continue
            new = Ri + Rj
            routes[ri] = new
            load[ri] = load[ri] + load[rj]
            for node in Rj:
                route_of[node] = ri
            del routes[rj]
        out = [r for r in routes.values()]
        if self.g.two_opt:
            out = [self._two_opt(r, D, self.g.two_opt) for r in out]
        return out

    @staticmethod
    def _route_cost(route: list[int], M: np.ndarray) -> float:
        prev = 0
        c = 0.0
        for node in route:
            c += M[prev, node]
            prev = node
        c += M[prev, 0]
        return c

    def _two_opt(self, route, D, passes):
        r = route[:]
        for _ in range(passes):
            improved = False
            for a in range(len(r) - 1):
                for b in range(a + 1, len(r)):
                    if a == 0 and b == len(r) - 1:
                        continue
                    pa = 0 if a == 0 else r[a - 1]
                    nb = 0 if b == len(r) - 1 else r[b + 1]
                    cur = D[pa, r[a]] + D[r[b], nb]
                    new = D[pa, r[b]] + D[r[a], nb]
                    if new + 1e-9 < cur:
                        r[a:b + 1] = r[a:b + 1][::-1]
                        improved = True
            if not improved:
                break
        return r

    def cost_vector(self, inst: FuzzyInstance, scenarios: np.ndarray) -> np.ndarray:
        routes = self.build_routes(inst)
        costs = np.empty(len(scenarios))
        for s, M in enumerate(scenarios):
            total = sum(self._route_cost(r, M) for r in routes)
            total += self.PER_ROUTE_COST * len(routes)
            costs[s] = total
        return costs


# --------------------------------------------------------------------------
# Objectives (polymorphic)
# --------------------------------------------------------------------------
class Objective(ABC):
    @abstractmethod
    def __call__(self, c: np.ndarray) -> float: ...


class ExpectedCost(Objective):
    def __call__(self, c): return float(c.mean())


class OWARisk(Objective):
    """Upper-tail OWA. With uniform mass on the worst q = ceil((1-orness)*M)
    scenarios this is the finite-scenario analogue of upper-tail CVaR, so
    f2 >= f1 always holds (Proposition 1)."""

    def __init__(self, orness: float = 0.7):
        self.orness = orness

    def __call__(self, c):
        M = len(c)
        if M == 1:
            return float(c[0])
        q = max(1, int(math.ceil((1.0 - self.orness) * M)))
        worst = np.sort(c)[::-1][:q]
        return float(worst.mean())


class Fairness(Objective):
    def __call__(self, c): return float(np.abs(c - c.mean()).mean())


@dataclass
class FitnessVector:
    f1: float
    f2: float
    f3: float

    def as_tuple(self): return (self.f1, self.f2, self.f3)


# --------------------------------------------------------------------------
# Pareto archive + 2D hypervolume on (f1, f2)
# --------------------------------------------------------------------------
class ParetoArchive:
    def __init__(self):
        self.items: list[tuple[FitnessVector, object]] = []

    @staticmethod
    def _dominates(a: FitnessVector, b: FitnessVector) -> bool:
        at, bt = a.as_tuple(), b.as_tuple()
        return all(x <= y for x, y in zip(at, bt)) and any(x < y for x, y in zip(at, bt))

    def add(self, fv: FitnessVector, payload):
        if any(self._dominates(o, fv) for o, _ in self.items):
            return
        self.items = [(o, p) for o, p in self.items if not self._dominates(fv, o)]
        self.items.append((fv, payload))

    def hypervolume(self, ref: tuple[float, float]) -> float:
        pts = sorted({(fv.f1, fv.f2) for fv, _ in self.items})
        nd, best = [], math.inf
        for x, y in pts:                       # keep non-dominated in (f1,f2)
            if y < best:
                nd.append((x, y)); best = y
        hv, prev_x = 0.0, ref[0]
        for x, y in sorted(nd):
            hv += max(0.0, prev_x - x) * max(0.0, ref[1] - y)
            prev_x = x
        return hv


# --------------------------------------------------------------------------
# Candidate pool + evaluator-as-specification (the experiment)
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    genome: Genome
    train: np.ndarray            # cost vector on training ensemble
    ood: np.ndarray              # cost vector on OOD stress ensemble

    def fitness(self, f1, f2, f3, scen="train") -> FitnessVector:
        c = self.train if scen == "train" else self.ood
        return FitnessVector(f1(c), f2(c), f3(c))


class CandidatePool:
    """Shared, evolutionarily-grown pool. Every method below selects from the
    same candidates, so differences are attributable only to the evaluator."""

    def __init__(self, budget: int = 80):
        self.budget = budget

    def grow(self, inst, train_scn, ood_scn, seed) -> list[Candidate]:
        rng = np.random.default_rng(seed + 31)
        f1 = ExpectedCost()
        pool, genomes = [], [Genome()]
        for _ in range(7):
            genomes.append(Genome().mutate(rng))
        for b in range(self.budget):
            if b < len(genomes):
                g = genomes[b]
            else:
                parent = pool[rng.integers(0, len(pool))].genome
                g = parent.mutate(rng)
            h = RoutingHeuristic(g)
            cand = Candidate(g, h.cost_vector(inst, train_scn),
                             h.cost_vector(inst, ood_scn))
            pool.append(cand)
        return pool


class Method(ABC):
    name: str

    @abstractmethod
    def select(self, pool: list[Candidate]) -> tuple[Candidate, ParetoArchive]: ...


class ClassicalMethod(Method):
    name = "Classical heuristic"

    def select(self, pool):
        # fixed textbook Clarke-Wright genome (first in the pool)
        cand = pool[0]
        arch = ParetoArchive()
        arch.add(cand.fitness(ExpectedCost(), OWARisk(), Fairness()), cand)
        return cand, arch


class DeterministicMethod(Method):
    name = "Deterministic AHD"  # scalar NOMINAL evaluator (blind to uncertainty)

    def select(self, pool):
        # deterministic AHD never sees the scenario ensemble: it ranks
        # candidates on the modal (nominal) cost only -- train[0] is the modal.
        best = min(pool, key=lambda c: c.train[0])
        arch = ParetoArchive()
        arch.add(best.fitness(ExpectedCost(), OWARisk(), Fairness()), best)
        return best, arch


class MONoFuzzyMethod(Method):
    name = "MO AHD (no fuzzy risk)"

    def select(self, pool):
        # multi-objective search but WITHOUT fuzzy scenarios: objectives are
        # evaluated on the modal instance only (nominal cost vs nominal route
        # imbalance), so the alpha-cut tail never enters selection.
        arch = ParetoArchive()
        for c in pool:
            routes_cost = c.train[0]
            fv = FitnessVector(routes_cost, c.genome.slack * routes_cost,
                               c.genome.balance)
            arch.add(fv, c)
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
        # OWA risk-dial selection: normalise (f1,f2) over the archive and pick
        # the member minimising the orness-weighted scalarisation (the knee).
        f1s = np.array([fv.f1 for fv, _ in arch.items])
        f2s = np.array([fv.f2 for fv, _ in arch.items])
        n1 = (f1s - f1s.min()) / (np.ptp(f1s) + 1e-9)
        n2 = (f2s - f2s.min()) / (np.ptp(f2s) + 1e-9)
        score = (1 - self.orness) * n1 + self.orness * n2
        knee = arch.items[int(np.argmin(score))][1]
        return knee, arch


# --------------------------------------------------------------------------
# Metrics and statistics
# --------------------------------------------------------------------------
def p95(c: np.ndarray) -> float:
    return float(np.percentile(c, 95))


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for k, (name, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - k) * p))
        out[name] = adj
        prev = adj
    return out


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------
def run(seeds: int = 30, n: int = 25, budget: int = 80, orness: float = 0.7,
        outdir: str = "."):
    gen = FuzzyInstanceGenerator(n=n, spread=0.15, regime="regional", n_regions=6)
    eb = EndpointBuilder(reps=3)
    lb = LatinHypercubeBuilder(m=24)
    sb = StressBuilder(m=12, extra=1.6)
    methods = [ClassicalMethod(), DeterministicMethod(),
               MONoFuzzyMethod(), CREoHMethod(orness)]
    f1, f2, f3 = ExpectedCost(), OWARisk(orness), Fairness()

    per_seed = {m.name: {k: [] for k in
                ["f1", "f2", "f3", "p95", "hv", "ood"]} for m in methods}
    raw_rows = []

    for seed in range(seeds):
        inst = gen.generate(seed)
        train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
        ood = sb.build(inst, seed)
        pool = CandidatePool(budget).grow(inst, train, ood, seed)
        # global normalisation for hypervolume reference
        allf1 = np.array([f1(c.train) for c in pool])
        allf2 = np.array([f2(c.train) for c in pool])
        lo1, hi1 = allf1.min(), allf1.max()
        lo2, hi2 = allf2.min(), allf2.max()

        def norm(fv):
            return FitnessVector((fv.f1 - lo1) / (hi1 - lo1 + 1e-9),
                                 (fv.f2 - lo2) / (hi2 - lo2 + 1e-9), fv.f3)

        for m in methods:
            sel, arch = m.select(pool)
            # Hypervolume is measured on the normalized 2D cost-risk plane
            # (f1, f2). Fairness remains part of archive dominance and
            # selection, but not of the reported HV indicator.
            narch = ParetoArchive()
            for _fv, p in arch.items:
                narch.add(norm(p.fitness(f1, f2, f3)), p)
            hv = narch.hypervolume((1.05, 1.05))
            mf1, mf2, mf3 = f1(sel.train), f2(sel.train), f3(sel.train)
            mp95 = p95(sel.train)
            ood_deg = 100.0 * (sel.ood.mean() - sel.train.mean()) / sel.train.mean()
            per_seed[m.name]["f1"].append(mf1)
            per_seed[m.name]["f2"].append(mf2)
            per_seed[m.name]["f3"].append(mf3)
            per_seed[m.name]["p95"].append(mp95)
            per_seed[m.name]["hv"].append(hv)
            per_seed[m.name]["ood"].append(ood_deg)
            raw_rows.append(dict(seed=seed, method=m.name, f1=mf1, f2=mf2,
                                 f3=mf3, p95=mp95, hv=hv, ood_degradation=ood_deg,
                                 lam=sel.genome.lam, slack=sel.genome.slack,
                                 balance=sel.genome.balance,
                                 scenario_costs=";".join(f"{x:.3f}" for x in sel.train)))

    # ---- normalise reporting so Classical mean f1 == 100 (index units) ----
    base = np.mean(per_seed["Classical heuristic"]["f1"])
    scale = 100.0 / base

    def ci(x):
        x = np.asarray(x); return 1.96 * x.std(ddof=1) / math.sqrt(len(x))

    summary = {}
    for m in methods:
        d = per_seed[m.name]
        summary[m.name] = dict(
            f1=(np.mean(d["f1"]) * scale, ci(np.array(d["f1"]) * scale)),
            f2=(np.mean(d["f2"]) * scale, ci(np.array(d["f2"]) * scale)),
            f3=(np.mean(d["f3"]) * scale, ci(np.array(d["f3"]) * scale)),
            p95=(np.mean(d["p95"]) * scale, ci(np.array(d["p95"]) * scale)),
            hv=(np.mean(d["hv"]), ci(d["hv"])),
            ood=(np.mean(d["ood"]), ci(d["ood"])),
        )

    # ---- statistics: C-R-EoH vs deterministic / MO / classical ----
    det = per_seed["Deterministic AHD"]
    mo = per_seed["MO AHD (no fuzzy risk)"]
    cr = per_seed["C-R-EoH"]
    raw_p = {}
    stats_rows = []

    def add_stat(comp, metric, a, b, fav_low=True):
        try:
            stat, p = wilcoxon(a, b)
        except ValueError:
            p = 1.0
        raw_p[(comp, metric)] = p
        stats_rows.append([comp, metric, p, cliffs_delta(a, b)])

    add_stat("C-R-EoH vs deterministic AHD", "P95", cr["p95"], det["p95"])
    add_stat("C-R-EoH vs deterministic AHD", "OOD degradation", cr["ood"], det["ood"])
    add_stat("C-R-EoH vs MO AHD", "Hypervolume", cr["hv"], mo["hv"])
    add_stat("C-R-EoH vs deterministic AHD", "Mean cost", cr["f1"], det["f1"])
    adj = holm({f"{c}|{mt}": p for (c, mt), p in raw_p.items()})
    for row in stats_rows:
        row.insert(3, adj[f"{row[0]}|{row[1]}"])

    fr = friedmanchisquare(
        per_seed["Classical heuristic"]["p95"],
        per_seed["Deterministic AHD"]["p95"],
        per_seed["MO AHD (no fuzzy risk)"]["p95"],
        per_seed["C-R-EoH"]["p95"])

    # ---- write artefacts ----
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "routing_contrast.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "f1_mean", "f1_ci", "f2_mean", "f2_ci",
                    "f3_mean", "f3_ci", "p95_mean", "p95_ci", "hv_mean", "hv_ci"])
        for m in methods:
            s = summary[m.name]
            w.writerow([m.name] + [f"{v:.2f}" for pair in
                       (s["f1"], s["f2"], s["f3"], s["p95"]) for v in pair]
                       + [f"{s['hv'][0]:.3f}", f"{s['hv'][1]:.3f}"])

    with open(os.path.join(outdir, "routing_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "metric", "wilcoxon_p_raw",
                    "wilcoxon_p_holm", "cliffs_delta"])
        for row in stats_rows:
            w.writerow([row[0], row[1], f"{row[2]:.4f}",
                        f"{row[3]:.4f}", f"{row[4]:.3f}"])

    with open(os.path.join(outdir, "routing_raw_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)

    meta = dict(seeds=seeds, n_customers=n, budget=budget, orness=orness,
                alpha_grid=[0, 0.25, 0.5, 0.75, 1.0], train_spread=0.15,
                ood_extra=1.6, friedman_stat=float(fr.statistic),
                friedman_p=float(fr.pvalue),
                index_scale="Classical mean f1 := 100")
    with open(os.path.join(outdir, "routing_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- console summary ----
    print(f"# seeds={seeds} n={n} budget={budget}")
    print(f"{'method':32s} {'f1':>8} {'f2':>8} {'f3':>7} {'P95':>8} {'HV':>6} {'OOD%':>7}")
    for m in methods:
        s = summary[m.name]
        print(f"{m.name:32s} {s['f1'][0]:8.2f} {s['f2'][0]:8.2f} "
              f"{s['f3'][0]:7.2f} {s['p95'][0]:8.2f} {s['hv'][0]:6.3f} {s['ood'][0]:7.2f}")
    det_p95 = summary["Deterministic AHD"]["p95"][0]
    cr_p95 = summary["C-R-EoH"]["p95"][0]
    det_f1 = summary["Deterministic AHD"]["f1"][0]
    cr_f1 = summary["C-R-EoH"]["f1"][0]
    print(f"\nCVRP P95 reduction vs det. AHD : {100*(det_p95-cr_p95)/det_p95:5.1f}%")
    print(f"CVRP nominal premium vs det.   : {100*(cr_f1-det_f1)/det_f1:5.1f}%")
    print("\nStatistics (Holm-adjusted):")
    for row in stats_rows:
        print(f"  {row[0]:38s} {row[1]:16s} p={row[3]:.4f}  delta={row[4]:+.3f}")
    print(f"Friedman P95: chi2={fr.statistic:.2f} p={fr.pvalue:.4g}")
    return summary, stats_rows


if __name__ == "__main__":
    run(seeds=30, n=25, budget=80, outdir="data")
