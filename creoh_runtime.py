"""Wall-clock cost and scalability of robust evaluation (revision).

The first submission analysed the evaluation cost only asymptotically, as the
``O(B M T_h(n))`` scenario-ensemble factor.  Reviewers asked for the practical
figure: how much wall-clock time the robust evaluator actually costs relative
to a deterministic one, how that cost scales with instance size ``n``, ensemble
size ``M`` and candidate budget ``B``, and whether the cost-benefit trade-off
survives at decision-support timescales.

This module measures, on the same hardware and in the same process:

* ``EvaluatorTimer`` -- per-candidate evaluation time of the deterministic
  (modal-instance) evaluator and of the fuzzy ``M``-scenario evaluator, so the
  measured overhead factor can be compared with the theoretical ``M``;
* ``ScalingStudy`` -- the same two quantities over grids of ``n``, ``M`` and
  ``B``, with a fitted power law ``t = a n^b`` for the heuristic execution;
* ``CostDecomposition`` -- separation of the per-candidate cost into the
  structural part (executed once) and the scenario part (executed ``M`` times),
  which yields a closed-form prediction of the overhead factor that is checked
  against the measurement and bounds the achievable parallel wall-clock factor;
* ``CostBenefitTable`` -- seconds of extra evaluation per unit of tail-risk
  reduction, the quantity a planner actually trades off.

All timings are medians over repetitions to damp scheduler noise, and the
machine description is recorded in the metadata file so the numbers are
interpretable.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import time
from abc import ABC, abstractmethod

import numpy as np

from creoh_routing import ExpectedCost, OWARisk, ScenarioStability, p95
import creoh_scheduling as sched


def _median_time(fn, reps: int = 5) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ==========================================================================
# Timers
# ==========================================================================
class EvaluatorTimer(ABC):
    """Times one candidate evaluation under a given evaluator regime."""

    name: str

    @abstractmethod
    def evaluate(self, heur, inst, scenarios) -> float: ...

    def time(self, heur, inst, scenarios, reps: int = 5) -> float:
        return _median_time(lambda: self.evaluate(heur, inst, scenarios), reps)


class DeterministicTimer(EvaluatorTimer):
    """Modal instance only: one scenario, scalar fitness."""

    name = "deterministic (modal)"

    def evaluate(self, heur, inst, scenarios):
        c = heur.cost_vector(inst, scenarios[:1])
        return float(c[0])


class FuzzyTimer(EvaluatorTimer):
    """Full alpha-cut ensemble with the three-objective fitness."""

    name = "fuzzy (M-scenario)"

    def __init__(self, orness: float = 0.7):
        self.f1, self.f2, self.f3 = ExpectedCost(), OWARisk(orness), ScenarioStability()

    def evaluate(self, heur, inst, scenarios):
        c = heur.cost_vector(inst, scenarios)
        return self.f1(c) + self.f2(c) + self.f3(c)


# ==========================================================================
# Studies
# ==========================================================================
class ScalingStudy:
    """Evaluation time as a function of n, M and B on the scheduling family."""

    def __init__(self, n_grid=(20, 40, 80, 160, 320, 640),
                 m_grid=(1, 9, 18, 37, 74, 148),
                 b_grid=(30, 60, 120, 240), seed: int = 0):
        self.n_grid, self.m_grid, self.b_grid, self.seed = (
            n_grid, m_grid, b_grid, seed)
        self.det, self.fuz = DeterministicTimer(), FuzzyTimer()

    def _setup(self, n: int):
        inst = sched.ScheduleGenerator(n=n).generate(self.seed)
        train = np.concatenate([sched.EndpointBuilder().build(inst, self.seed),
                                sched.LatinHypercubeBuilder().build(inst, self.seed)])
        return inst, train

    def by_instance_size(self) -> list[dict]:
        out = []
        for n in self.n_grid:
            inst, train = self._setup(n)
            h = sched.DispatchHeuristic(sched.Genome(0.9, 0.6, 0.3))
            td = self.det.time(h, inst, train)
            tf = self.fuz.time(h, inst, train)
            out.append(dict(n=n, M=int(train.shape[0]),
                            t_det_ms=1e3 * td, t_fuzzy_ms=1e3 * tf,
                            overhead_factor=tf / td))
        return out

    def by_ensemble_size(self, n: int = 40) -> list[dict]:
        inst, train = self._setup(n)
        h = sched.DispatchHeuristic(sched.Genome(0.9, 0.6, 0.3))
        base = self.det.time(h, inst, train)
        out = []
        for m in self.m_grid:
            idx = np.linspace(0, train.shape[0] - 1, min(m, train.shape[0])).astype(int)
            sub = train[idx]
            t = _median_time(lambda: self.fuz.evaluate(h, inst, sub))
            out.append(dict(M=len(sub), t_ms=1e3 * t, overhead_factor=t / base))
        return out

    def by_budget(self, n: int = 40) -> list[dict]:
        inst, train = self._setup(n)
        ood = sched.StressBuilder().build(inst, self.seed)
        out = []
        for b in self.b_grid:
            t = _median_time(lambda: sched.CandidatePool(b).grow(
                inst, train, ood, self.seed), reps=3)
            out.append(dict(B=b, t_full_run_s=t, t_per_candidate_ms=1e3 * t / b))
        return out

    @staticmethod
    def power_law(rows, xkey, ykey) -> tuple[float, float]:
        x = np.log(np.array([r[xkey] for r in rows], float))
        y = np.log(np.array([r[ykey] for r in rows], float))
        b, loga = np.polyfit(x, y, 1)
        return float(math.exp(loga)), float(b)


class CostDecomposition:
    """Why the measured overhead is far below the asymptotic factor ``M``.

    A candidate evaluation has two parts: the *structural* part (building the
    assignment or the routes), executed once per candidate and independent of
    the ensemble, and the *scenario* part, executed ``M`` times.  Writing
    ``t_fuzzy(n, M) = t_struct(n) + M * t_scen(n)`` and
    ``t_det(n) = t_struct(n) + t_scen(n)``, the overhead factor is

        t_fuzzy / t_det = 1 + (M - 1) / (1 + t_struct / t_scen),

    which tends to ``M`` only when the structural cost is negligible and falls
    towards 1 as the structural cost dominates.  Both components are measured
    here, and the predicted factor is compared with the observed one, so the
    scalability claim is verified rather than asserted.  The same decomposition
    gives the attainable parallel wall-clock factor with ``p`` workers,
    ``(t_struct + ceil(M/p) * t_scen) / t_det``, assuming zero communication
    overhead, since only the scenario part is embarrassingly parallel.  This
    factor is computed, not measured.
    """

    def __init__(self, n_grid=(20, 40, 80, 160, 320, 640), seed: int = 0):
        self.n_grid, self.seed = n_grid, seed

    def run(self) -> list[dict]:
        out = []
        for n in self.n_grid:
            inst = sched.ScheduleGenerator(n=n).generate(self.seed)
            train = np.concatenate([
                sched.EndpointBuilder().build(inst, self.seed),
                sched.LatinHypercubeBuilder().build(inst, self.seed)])
            M = train.shape[0]
            h = sched.DispatchHeuristic(sched.Genome(0.9, 0.6, 0.3))
            t_struct = _median_time(lambda: h.assign(inst), reps=7)
            t_all = _median_time(lambda: h.cost_vector(inst, train), reps=7)
            t_one = _median_time(lambda: h.cost_vector(inst, train[:1]), reps=7)
            t_scen = max((t_all - t_struct) / M, 1e-9)
            predicted = 1 + (M - 1) / (1 + t_struct / t_scen)
            out.append(dict(n=n, M=M, t_struct_ms=1e3 * t_struct,
                            t_scenario_ms=1e3 * t_scen,
                            t_det_ms=1e3 * t_one, t_fuzzy_ms=1e3 * t_all,
                            predicted_factor=round(predicted, 2),
                            observed_factor=round(t_all / t_one, 2),
                            parallel_bound_p8=round(
                                (t_struct + math.ceil(M / 8) * t_scen) / t_one, 2)))
        return out


class CostBenefitTable:
    """Seconds of extra evaluation bought per point of tail-risk reduction."""

    def __init__(self, seeds: int = 30, n: int = 40, budget: int = 120,
                 orness: float = 0.7):
        self.seeds, self.n, self.budget, self.orness = seeds, n, budget, orness

    def run(self) -> dict:
        gen = sched.ScheduleGenerator(n=self.n)
        eb, lb, sb = (sched.EndpointBuilder(), sched.LatinHypercubeBuilder(),
                      sched.StressBuilder())
        f1 = ExpectedCost()
        det_p95, cr_p95, t_det, t_fuz = [], [], [], []
        det_t = DeterministicTimer(); fuz_t = FuzzyTimer(self.orness)
        for seed in range(self.seeds):
            inst = gen.generate(seed)
            train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
            ood = sb.build(inst, seed)
            t0 = time.perf_counter()
            pool = sched.CandidatePool(self.budget).grow(inst, train, ood, seed)
            t_pool = time.perf_counter() - t0
            h = sched.DispatchHeuristic(pool[0].genome)
            td = det_t.time(h, inst, train, reps=3)
            tf = fuz_t.time(h, inst, train, reps=3)
            t_det.append(self.budget * td)
            t_fuz.append(self.budget * tf)
            d, _ = sched.DeterministicMethod().select(pool)
            c, _ = sched.CREoHMethod(self.orness).select(pool)
            det_p95.append(p95(d.train)); cr_p95.append(p95(c.train))
            _ = t_pool
        det_p95 = np.array(det_p95); cr_p95 = np.array(cr_p95)
        red = float((det_p95 - cr_p95).mean() / det_p95.mean() * 100)
        extra = float(np.mean(t_fuz) - np.mean(t_det))
        return dict(seeds=self.seeds, n_tasks=self.n, budget=self.budget,
                    t_deterministic_search_s=float(np.mean(t_det)),
                    t_fuzzy_search_s=float(np.mean(t_fuz)),
                    extra_seconds_per_run=extra,
                    p95_reduction_pct=round(red, 2),
                    seconds_per_pct_tail_reduction=round(extra / red, 3)
                    if red > 0 else None)


# ==========================================================================
# Driver
# ==========================================================================
def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(outdir: str = "data") -> dict:
    os.makedirs(outdir, exist_ok=True)
    st = ScalingStudy()
    by_n = st.by_instance_size()
    by_m = st.by_ensemble_size()
    by_b = st.by_budget()
    a_det, b_det = ScalingStudy.power_law(by_n, "n", "t_det_ms")
    a_fuz, b_fuz = ScalingStudy.power_law(by_n, "n", "t_fuzzy_ms")
    par = CostDecomposition().run()
    cb = CostBenefitTable().run()

    with open(os.path.join(outdir, "runtime_scaling.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "x", "value", "t_ms_or_s", "overhead_factor"])
        for r in by_n:
            w.writerow(["instance_size_n", r["n"], r["t_det_ms"],
                        r["t_fuzzy_ms"], round(r["overhead_factor"], 2)])
        for r in by_m:
            w.writerow(["ensemble_size_M", r["M"], "", r["t_ms"],
                        round(r["overhead_factor"], 2)])
        for r in by_b:
            w.writerow(["budget_B", r["B"], "", r["t_full_run_s"],
                        round(r["t_per_candidate_ms"], 3)])
    with open(os.path.join(outdir, "runtime_decomposition.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(par[0].keys()))
        w.writeheader(); w.writerows(par)

    meta = dict(
        machine=dict(platform=platform.platform(), python=platform.python_version(),
                     processor=platform.processor() or platform.machine(),
                     cpu_model=_cpu_model(), logical_cpus=os.cpu_count()),
        max_relative_prediction_error=round(float(max(
            abs(r["predicted_factor"] - r["observed_factor"]) / r["observed_factor"]
            for r in par)), 4),
        by_instance_size=by_n, by_ensemble_size=by_m, by_budget=by_b,
        power_law=dict(deterministic=dict(a=a_det, exponent=round(b_det, 3)),
                       fuzzy=dict(a=a_fuz, exponent=round(b_fuz, 3))),
        decomposition=par, cost_benefit=cb,
        theoretical_overhead_factor=int(by_n[0]["M"]),
        measured_overhead_factor=round(float(np.median(
            [r["overhead_factor"] for r in by_n])), 2))
    with open(os.path.join(outdir, "runtime_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"# runtime study on {meta['machine']['processor']} "
          f"({meta['machine']['logical_cpus']} logical CPUs)")
    print(f"{'n':>6} {'M':>5} {'t_det (ms)':>11} {'t_fuzzy (ms)':>13} {'factor':>8}")
    for r in by_n:
        print(f"{r['n']:6d} {r['M']:5d} {r['t_det_ms']:11.3f} "
              f"{r['t_fuzzy_ms']:13.3f} {r['overhead_factor']:8.1f}")
    print(f"\nfitted t ~ n^{b_det:.2f} (deterministic), n^{b_fuz:.2f} (fuzzy); "
          f"M = {by_n[0]['M']}, median measured overhead "
          f"{meta['measured_overhead_factor']}x")
    print("\ncost decomposition (structural vs scenario part):")
    print(f"{'n':>6} {'t_struct':>10} {'t_scen':>9} {'pred.':>7} {'obs.':>7} "
          f"{'p=8 bound':>10}")
    for r in par:
        print(f"{r['n']:6d} {r['t_struct_ms']:10.3f} {r['t_scenario_ms']:9.4f} "
              f"{r['predicted_factor']:7.2f} {r['observed_factor']:7.2f} "
              f"{r['parallel_bound_p8']:10.2f}")
    print(f"\ncost-benefit: +{cb['extra_seconds_per_run']:.2f} s per run buys "
          f"{cb['p95_reduction_pct']:.1f}% tail reduction "
          f"({cb['seconds_per_pct_tail_reduction']} s per percentage point)")
    return meta


if __name__ == "__main__":
    run(outdir="data")
