"""Public-benchmark validation of the C-R-EoH evaluator (revision).

Reviewers of the first submission asked for external validity: the same fuzzy
injection protocol applied to established, author-independent instance
libraries rather than to the synthetic generator alone.  This module supplies
exactly that.

Instance libraries (redistributed under their original public terms in
``instances/``):

* **CVRPLIB X-set** -- Uchoa, Pecin, Pessoa, Poggi, Subramanian and Vidal
  (2017) capacitated vehicle-routing instances, ``X-n101-k25`` ...
  ``X-n300-k*`` (43 instances, 100 <= n <= 300).
* **Solomon (1987)** -- the 56 classical 100-customer VRPTW source files, used
  here in two roles: (i) as capacitated routing instances (time windows
  relaxed, capacity and demands retained) and (ii) as the geometric and
  duration source of a public *field-service scheduling* family. Because the
  experiment does not use time-window fields, files that differ only in those
  fields are collapsed before analysis (six distinct routing inputs and four
  distinct scheduling inputs).

Both families are pushed through the *unchanged* fuzzy-injection protocol of
the manuscript: latent congestion zones, per-zone volatility, common-mode zone
shocks, alpha-cut scenario ensembles.  The heuristic families, evaluator,
archive, selection rule and statistics are the ones already released with the
first submission; nothing in the method is re-tuned for the public data.

Design note: the module is organised around two polymorphic hierarchies,
``InstanceReader`` (library-specific parsing) and ``FuzzyInjector``
(family-specific uncertainty structure), so that adding a further library
requires one subclass and no change to the experiment driver.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare

from creoh_routing import (FuzzyInstance, EndpointBuilder, LatinHypercubeBuilder,
                           StressBuilder, CandidatePool as RoutingPool,
                           ClassicalMethod as RClassical,
                           DeterministicMethod as RDeterministic,
                           MONoFuzzyMethod as RMONoFuzzy,
                           CREoHMethod as RCREoH,
                           ExpectedCost, OWARisk, ScenarioStability, FitnessVector,
                           ParetoArchive, p95, cliffs_delta, holm)
import creoh_scheduling as sched

HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCE_ROOT = os.path.join(HERE, "instances")


# ==========================================================================
# 1. Raw public instances and polymorphic readers
# ==========================================================================
@dataclass
class RawInstance:
    """Library-agnostic view of a public instance."""
    name: str
    library: str
    coords: np.ndarray        # (n+1, 2), row 0 = depot
    demand: np.ndarray        # (n+1,), demand[0] = 0
    capacity: float
    service: np.ndarray       # (n+1,), service[0] = 0 (zeros when unavailable)

    @property
    def n(self) -> int:
        return len(self.demand) - 1


class InstanceReader(ABC):
    """Reads one public library into :class:`RawInstance` objects."""

    library: str
    pattern: str

    def paths(self) -> list[str]:
        return sorted(glob.glob(os.path.join(INSTANCE_ROOT, self.pattern)))

    def read_all(self) -> list[RawInstance]:
        return [self.read(p) for p in self.paths()]

    @abstractmethod
    def read(self, path: str) -> RawInstance: ...


class CVRPLIBReader(InstanceReader):
    """TSPLIB/CVRPLIB ``.vrp`` format (EUC_2D)."""

    library = "CVRPLIB-X"
    pattern = "cvrplib/*.vrp"

    def read(self, path: str) -> RawInstance:
        cap, dim = None, None
        coords, demand, depot_idx = {}, {}, 1
        section = None
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                up = s.upper()
                if up.startswith("CAPACITY"):
                    cap = float(s.split(":")[1]); continue
                if up.startswith("DIMENSION"):
                    dim = int(s.split(":")[1]); continue
                if up.startswith("NODE_COORD_SECTION"):
                    section = "coord"; continue
                if up.startswith("DEMAND_SECTION"):
                    section = "demand"; continue
                if up.startswith("DEPOT_SECTION"):
                    section = "depot"; continue
                if up.startswith(("EOF", "EDGE_WEIGHT", "NAME", "COMMENT", "TYPE")):
                    if up.startswith("EOF"):
                        break
                    continue
                tok = s.split()
                if section == "coord" and len(tok) >= 3:
                    coords[int(tok[0])] = (float(tok[1]), float(tok[2]))
                elif section == "demand" and len(tok) >= 2:
                    demand[int(tok[0])] = float(tok[1])
                elif section == "depot" and tok[0] not in ("-1",):
                    depot_idx = int(tok[0])
        ids = sorted(coords)
        assert dim is None or len(ids) == dim
        order = [depot_idx] + [i for i in ids if i != depot_idx]
        c = np.array([coords[i] for i in order], dtype=float)
        d = np.array([demand.get(i, 0.0) for i in order], dtype=float)
        d[0] = 0.0
        return RawInstance(os.path.basename(path).replace(".vrp", ""),
                           self.library, c, d, float(cap), np.zeros(len(order)))


class SolomonReader(InstanceReader):
    """Solomon (1987) 100-customer VRPTW text format.

    Time windows are not used by the capacitated routing family; the service
    durations are retained because the public scheduling family is built from
    them (see :class:`SolomonSchedulingInjector`).
    """

    library = "Solomon"
    pattern = "solomon/*.txt"

    def read(self, path: str) -> RawInstance:
        rows, cap = [], None
        with open(path) as fh:
            lines = [l.rstrip("\n") for l in fh]
        for k, line in enumerate(lines):
            tok = line.split()
            if len(tok) == 2 and tok[0].isdigit() and tok[1].isdigit() and cap is None:
                cap = float(tok[1]); continue
            if len(tok) == 7 and all(_is_num(t) for t in tok):
                rows.append([float(t) for t in tok])
        arr = np.array(rows)
        arr = arr[np.argsort(arr[:, 0], kind="stable")]
        coords = arr[:, 1:3]
        demand = arr[:, 3].copy(); demand[0] = 0.0
        service = arr[:, 6].copy(); service[0] = 0.0
        name = os.path.basename(path).replace(".txt", "").upper()
        return RawInstance(name, self.library, coords, demand, float(cap), service)


def _is_num(t: str) -> bool:
    try:
        float(t); return True
    except ValueError:
        return False


def _array_key(a: np.ndarray) -> tuple:
    """Return an exact, stable key for a numeric model input."""
    x = np.ascontiguousarray(a, dtype=np.float64)
    return x.shape, x.tobytes()


def unique_routing_instances(raws: list[RawInstance]) -> list[RawInstance]:
    """Keep one source file per distinct static CVRP input."""
    seen, out = set(), []
    for raw in raws:
        key = (_array_key(raw.coords), _array_key(raw.demand), raw.capacity)
        if key not in seen:
            seen.add(key)
            out.append(raw)
    return out


def unique_scheduling_instances(raws: list[RawInstance]) -> list[RawInstance]:
    """Keep one source file per distinct scheduling input actually modelled."""
    seen, out = set(), []
    for raw in raws:
        key = (_array_key(raw.coords), _array_key(raw.service))
        if key not in seen:
            seen.add(key)
            out.append(raw)
    return out


# ==========================================================================
# 2. Fuzzy injection onto public instances (protocol of the manuscript, unchanged)
# ==========================================================================
class FuzzyInjector(ABC):
    """Applies the manuscript's fuzzy-uncertainty protocol to a public instance.

    The protocol has three steps, identical to the synthetic generator:
    (i) partition the instance into ``n_zones`` latent congestion zones,
    (ii) draw one volatility ``nu_g ~ U(0.05, 0.85) * (delta/0.15)`` per zone,
    independently of arc length or task size, (iii) let every uncertain
    quantity in zone ``g`` become the triangular fuzzy number
    ``((1-nu/2) z, z, (1+nu) z)``.  Only the *partition* is instance-derived:
    zones follow the public coordinates instead of synthetic centroids.
    """

    def __init__(self, spread: float = 0.15, n_zones: int = 6):
        self.spread = spread
        self.n_zones = n_zones

    def zones(self, coords: np.ndarray, seed: int) -> np.ndarray:
        """Lloyd k-means on the *public* coordinates: zones are a property of
        the instance geometry, not of the author's generator."""
        rng = np.random.default_rng(seed + 5701)
        pts = coords[1:]
        Z = self.n_zones
        cent = pts[rng.choice(len(pts), size=Z, replace=False)]
        lab = np.zeros(len(pts), dtype=int)
        for _ in range(25):
            d = ((pts[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            new = np.argmin(d, axis=1)
            if np.array_equal(new, lab):
                break
            lab = new
            for g in range(Z):
                if (lab == g).any():
                    cent[g] = pts[lab == g].mean(0)
        return np.concatenate([[-1], lab])          # depot has no zone

    @abstractmethod
    def inject(self, raw: RawInstance, seed: int): ...


class RoutingFuzzyInjector(FuzzyInjector):
    """Public CVRP instance -> :class:`FuzzyInstance` with fuzzy travel times."""

    def inject(self, raw: RawInstance, seed: int) -> FuzzyInstance:
        rng = np.random.default_rng(seed + 9161)
        coords = raw.coords
        diff = coords[:, None, :] - coords[None, :, :]
        base = np.sqrt((diff ** 2).sum(-1))
        region = self.zones(coords, seed)
        region_vol = rng.uniform(0.05, 0.85, size=self.n_zones) * (self.spread / 0.15)
        node_vol = np.where(region < 0, 0.0, region_vol[np.clip(region, 0, None)])
        arc_vol = np.maximum(node_vol[:, None], node_vol[None, :])
        gov_i = (node_vol[:, None] >= node_vol[None, :])
        arc_region = np.where(gov_i, region[:, None], region[None, :])
        np.fill_diagonal(arc_vol, 0.0)
        return FuzzyInstance(coords, raw.demand, float(raw.capacity), base,
                             region, region_vol, arc_vol, arc_region)


class SolomonSchedulingInjector(FuzzyInjector):
    """Public *field-service scheduling* family derived from Solomon instances.

    A field-service work order for customer ``i`` occupies the technician for
    the on-site service time plus the access time to the customer, so its
    modal duration is

        p_i = s_i + kappa * d(0, i),        kappa = 1.0,

    with ``s_i`` the Solomon service duration and ``d(0,i)`` the Euclidean
    depot distance.  Both quantities come from the published instance file; no
    duration is drawn by the authors.  The regular shift is set per instance to
    ``jobs_per_shift * mean(p)`` so that the family is scale-free across the
    C / R / RC groups, and the overtime rate and technician cost are the
    manuscript defaults.  Congestion zones follow the instance's own spatial
    structure, which differs by construction between the clustered (C),
    random (R) and mixed (RC) Solomon groups -- three genuinely different
    uncertainty geometries obtained without any author choice.
    """

    def __init__(self, spread: float = 0.15, n_zones: int = 6,
                 kappa: float = 1.0, jobs_per_shift: float = 6.0,
                 overtime_rate: float = 1.5):
        super().__init__(spread, n_zones)
        self.kappa = kappa
        self.jobs_per_shift = jobs_per_shift
        self.overtime_rate = overtime_rate

    def inject(self, raw: RawInstance, seed: int) -> sched.FuzzyScheduleInstance:
        rng = np.random.default_rng(seed + 9161)
        d0 = np.sqrt(((raw.coords[1:] - raw.coords[0]) ** 2).sum(-1))
        proc = raw.service[1:] + self.kappa * d0
        zone = self.zones(raw.coords, seed)[1:]
        zone_vol = rng.uniform(0.05, 0.85, size=self.n_zones) * (self.spread / 0.15)
        vol = zone_vol[zone]
        shift = float(self.jobs_per_shift * proc.mean())
        return sched.FuzzyScheduleInstance(proc, zone, zone_vol, vol, shift,
                                           self.overtime_rate, 0.4 * shift)


# ==========================================================================
# 3. Experiment drivers
# ==========================================================================
def _ci(x) -> float:
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x))


class PublicExperiment(ABC):
    """One public instance family run through the four competing evaluators.

    The experimental unit is a *distinct model input*: every method sees the same
    candidate pool on the same instance, per-instance results are averaged over
    ``seeds`` scenario-construction seeds, and every paired test is taken over
    the instance-level means.  This removes the dependence between repeated
    scenario evaluations of one instance that the first submission left
    implicit.
    """

    family: str
    metrics = ("f1", "f2", "f3", "p95", "hv", "ood")

    def __init__(self, orness: float = 0.7, budget: int = 80, seeds: int = 3,
                 spread: float = 0.15):
        # ``orness`` is retained as a backwards-compatible API name.  Its
        # value is the empirical-CVaR confidence / tail-cutoff dial, not the
        # induced OWA orness (see the manuscript for the exact mapping).
        self.orness, self.budget, self.seeds, self.spread = (
            orness, budget, seeds, spread)
        self.f1, self.f2, self.f3 = ExpectedCost(), OWARisk(orness), ScenarioStability()
        self.source_file_count = 0

    @abstractmethod
    def instances(self) -> list[RawInstance]: ...

    @abstractmethod
    def build(self, raw: RawInstance, seed: int): ...

    @abstractmethod
    def methods(self) -> list: ...

    def run(self) -> dict:
        raws = self.instances()
        names = [m.name for m in self.methods()]
        per_inst = {m: {k: [] for k in self.metrics} for m in names}
        rows = []
        t0 = time.time()
        for raw in raws:
            acc = {m: {k: [] for k in self.metrics} for m in names}
            for seed in range(self.seeds):
                inst, train, ood, pool = self.build(raw, seed)
                a1 = np.array([self.f1(c.train) for c in pool])
                a2 = np.array([self.f2(c.train) for c in pool])
                lo1, hi1, lo2, hi2 = a1.min(), a1.max(), a2.min(), a2.max()

                def norm(fv):
                    return FitnessVector((fv.f1 - lo1) / (hi1 - lo1 + 1e-9),
                                         (fv.f2 - lo2) / (hi2 - lo2 + 1e-9), fv.f3)

                for m in self.methods():
                    sel, arch = m.select(pool)
                    narch = ParetoArchive()
                    for _fv, p in arch.items:
                        narch.add(norm(p.fitness(self.f1, self.f2, self.f3)), p)
                    vals = dict(f1=self.f1(sel.train), f2=self.f2(sel.train),
                                f3=self.f3(sel.train), p95=p95(sel.train),
                                hv=narch.hypervolume((1.05, 1.05)),
                                ood=100.0 * (sel.ood.mean() - sel.train.mean())
                                / sel.train.mean())
                    for k, v in vals.items():
                        acc[m.name][k].append(v)
            for m in names:                       # instance-level aggregation
                base = np.mean(acc[names[0]]["f1"])
                for k in self.metrics:
                    per_inst[m][k].append(float(np.mean(acc[m][k])))
                rows.append(dict(instance=raw.name, library=raw.library,
                                 n=raw.n, method=m,
                                 **{k: round(float(np.mean(acc[m][k])), 4)
                                    for k in self.metrics}))
            print(f"  {raw.library:10s} {raw.name:14s} n={raw.n:4d} "
                  f"[{time.time()-t0:6.1f}s]", flush=True)
        return self._summarise(per_inst, rows, names)

    def _summarise(self, per_inst, rows, names) -> dict:
        base = np.mean(per_inst[names[0]]["f1"])
        scale = 100.0 / base
        summary = {}
        for m in names:
            d = per_inst[m]
            summary[m] = {}
            for k in self.metrics:
                arr = np.array(d[k], dtype=float)
                if k in ("f1", "f2", "f3", "p95"):
                    arr = arr * scale
                summary[m][k] = (float(arr.mean()), float(_ci(arr)))
        det, cr = per_inst["Deterministic AHD"], per_inst["C-R-EoH"]
        mo = per_inst["MO AHD (no fuzzy risk)"]
        raw_p, stats_rows = {}, []

        def add(comp, metric, a, b):
            try:
                _, p = wilcoxon(a, b)
            except ValueError:
                p = 1.0
            raw_p[f"{comp}|{metric}"] = float(p)
            stats_rows.append([comp, metric, float(p), cliffs_delta(a, b)])

        add("C-R-EoH vs deterministic AHD", "P95", cr["p95"], det["p95"])
        add("C-R-EoH vs deterministic AHD", "OOD degradation", cr["ood"], det["ood"])
        add("C-R-EoH vs MO AHD", "Hypervolume", cr["hv"], mo["hv"])
        add("C-R-EoH vs deterministic AHD", "Expected cost", cr["f1"], det["f1"])
        add("C-R-EoH vs deterministic AHD", "Scenario stability", cr["f3"], det["f3"])
        adj = holm(raw_p)
        for r in stats_rows:
            r.insert(3, adj[f"{r[0]}|{r[1]}"])
        fr = friedmanchisquare(*[per_inst[m]["p95"] for m in names])
        return dict(family=self.family, summary=summary, stats=stats_rows,
                    rows=rows, per_inst=per_inst,
                    friedman=dict(chi2=float(fr.statistic), p=float(fr.pvalue)),
                    n_instances=len(per_inst[names[0]]["p95"]),
                    source_file_count=self.source_file_count, scale=scale)


class PublicRoutingExperiment(PublicExperiment):
    family = "public_routing"

    def __init__(self, reader: InstanceReader, **kw):
        super().__init__(**kw)
        self.reader = reader
        self.injector = RoutingFuzzyInjector(self.spread)
        self.eb, self.lb, self.sb = (EndpointBuilder(reps=3),
                                     LatinHypercubeBuilder(m=24),
                                     StressBuilder(m=12, extra=1.6))

    def instances(self):
        raws = self.reader.read_all()
        self.source_file_count = len(raws)
        return unique_routing_instances(raws)

    def methods(self):
        return [RClassical(), RDeterministic(), RMONoFuzzy(), RCREoH(self.orness)]

    def build(self, raw, seed):
        inst = self.injector.inject(raw, seed)
        train = np.concatenate([self.eb.build(inst, seed), self.lb.build(inst, seed)])
        ood = self.sb.build(inst, seed)
        pool = RoutingPool(self.budget).grow(inst, train, ood, seed)
        return inst, train, ood, pool


class PublicSchedulingExperiment(PublicExperiment):
    family = "public_scheduling"

    def __init__(self, reader: InstanceReader, **kw):
        super().__init__(**kw)
        self.reader = reader
        self.injector = SolomonSchedulingInjector(self.spread)
        self.eb, self.lb, self.sb = (sched.EndpointBuilder(),
                                     sched.LatinHypercubeBuilder(),
                                     sched.StressBuilder())

    def instances(self):
        raws = self.reader.read_all()
        self.source_file_count = len(raws)
        return unique_scheduling_instances(raws)

    def methods(self):
        return [sched.ClassicalMethod(), sched.DeterministicMethod(),
                sched.MONoFuzzyMethod(), sched.CREoHMethod(self.orness)]

    def build(self, raw, seed):
        inst = self.injector.inject(raw, seed)
        train = np.concatenate([self.eb.build(inst, seed), self.lb.build(inst, seed)])
        ood = self.sb.build(inst, seed)
        pool = sched.CandidatePool(self.budget).grow(inst, train, ood, seed)
        return inst, train, ood, pool


# ==========================================================================
# 4. Driver
# ==========================================================================
def _write(res: dict, outdir: str, tag: str):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"public_{tag}_main.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "f1_mean", "f1_ci", "f2_mean", "f2_ci", "f3_mean",
                    "f3_ci", "p95_mean", "p95_ci", "hv_mean", "hv_ci",
                    "ood_mean", "ood_ci"])
        for m, s in res["summary"].items():
            w.writerow([m] + [f"{v:.2f}" for k in ("f1", "f2", "f3", "p95")
                              for v in s[k]]
                       + [f"{s['hv'][0]:.3f}", f"{s['hv'][1]:.3f}",
                          f"{s['ood'][0]:.2f}", f"{s['ood'][1]:.2f}"])
    with open(os.path.join(outdir, f"public_{tag}_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "metric", "wilcoxon_p_raw", "wilcoxon_p_holm",
                    "cliffs_delta"])
        for r in res["stats"]:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.3f}"])
    with open(os.path.join(outdir, f"public_{tag}_per_instance.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["rows"][0].keys()))
        w.writeheader(); w.writerows(res["rows"])


def run(outdir: str = "data", seeds: int = 3, budget: int = 80) -> dict:
    out = {}
    print("[public 1/3] CVRPLIB X-set (routing) ...")
    r1 = PublicRoutingExperiment(CVRPLIBReader(), seeds=seeds, budget=budget).run()
    _write(r1, outdir, "cvrplib")
    print("[public 2/3] Solomon (routing, TW relaxed) ...")
    r2 = PublicRoutingExperiment(SolomonReader(), seeds=seeds, budget=budget).run()
    _write(r2, outdir, "solomon_routing")
    print("[public 3/3] Solomon-derived field-service scheduling ...")
    r3 = PublicSchedulingExperiment(SolomonReader(), seeds=seeds,
                                    budget=budget + 40).run()
    _write(r3, outdir, "solomon_scheduling")

    for tag, res in (("cvrplib", r1), ("solomon_routing", r2),
                     ("solomon_scheduling", r3)):
        s = res["summary"]
        det, cr = s["Deterministic AHD"], s["C-R-EoH"]
        out[tag] = dict(
            n_instances=res["n_instances"],
            source_file_count=res["source_file_count"],
            p95_reduction_pct=round(100 * (det["p95"][0] - cr["p95"][0])
                                    / det["p95"][0], 2),
            owa_reduction_pct=round(100 * (det["f2"][0] - cr["f2"][0])
                                    / det["f2"][0], 2),
            stability_reduction_pct=round(100 * (det["f3"][0] - cr["f3"][0])
                                          / det["f3"][0], 2),
            expected_cost_change_pct=round(100 * (cr["f1"][0] - det["f1"][0])
                                           / det["f1"][0], 2),
            ood_det=round(det["ood"][0], 2), ood_creoh=round(cr["ood"][0], 2),
            hv_det=round(det["hv"][0], 3), hv_creoh=round(cr["hv"][0], 3),
            summary={m: {k: [round(v[0], 3), round(v[1], 3)] for k, v in d.items()}
                     for m, d in s.items()},
            stats=[[r[0], r[1], r[2], r[3], round(r[4], 3)] for r in res["stats"]],
            friedman=res["friedman"])
        print(f"\n== {tag}: {res['n_instances']} instances")
        print(f"{'method':32s} {'f1':>8} {'f2':>8} {'f3':>8} {'P95':>8} "
              f"{'HV':>6} {'OOD%':>7}")
        for m, d in s.items():
            print(f"{m:32s} {d['f1'][0]:8.2f} {d['f2'][0]:8.2f} {d['f3'][0]:8.2f} "
                  f"{d['p95'][0]:8.2f} {d['hv'][0]:6.3f} {d['ood'][0]:7.2f}")
        for r in res["stats"]:
            print(f"  {r[0]:34s} {r[1]:20s} p_holm={r[3]:.2e} delta={r[4]:+.3f}")

    with open(os.path.join(outdir, "public_benchmarks_metadata.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    run(outdir="data")
