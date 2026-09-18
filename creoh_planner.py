"""Planner-facing decision analysis of the OWA risk dial (revision).

The planner-facing interface of the first submission was designed and
illustrated but never analysed.  No human-subject data are used: an appendix of
the manuscript specifies a prospective practitioner study that has not yet been
conducted, and ``generate_planner_rehearsal.py`` produces a disclosed,
fixed-seed scripted rehearsal (feasibility dry-run) of that design.  What this module supplies is the *quantitative*
archive-based analysis that does not require human subjects: what
the risk dial actually does to the decision, how sensitive the decision is to
the dial setting, and what a planner loses by setting the dial wrongly.

Four analyses, all computed from the released per-seed archives:

``OrnessSweep``
    The selected portfolio member, and its realised cost, tail and stability,
    as a continuous function of the dial ``theta`` on a fine grid.  This turns
    the dial from an illustration into a measured response curve and yields the
    practical guidance the reviewers asked for (which ranges of ``theta``
    actually change the decision, and where the knee sits).

``SelectionStability``
    How often the dial changes the decision at all: the number of distinct
    portfolio members reachable over the dial range, the size of the ``theta``
    interval mapping to each of them, and the rank correlation between dial
    setting and selected buffer level.  A dial whose output changes at every
    infinitesimal move would be unusable; one that never changes the output
    would be pointless.

``PlannerRegret``
    The decision-theoretic cost of a wrong dial setting.  Six archetypal
    planner risk profiles are defined by an explicit utility over (expected
    cost, tail, stability).  For each profile the *oracle* portfolio member is
    the one maximising that planner's utility; regret is the utility loss of
    the member the dial actually returns, and is compared against the regret of
    accepting the deterministic AHD recommendation, which is what the planner
    would otherwise receive.  This quantifies the value of the portfolio to a
    planner without asking a planner anything.

``ParameterGuidance``
    A two-way sensitivity over the dial ``theta`` and the uncertainty spread
    ``delta``, reported as the setting that minimises each profile's regret, so
    that the paper can state operational recommendations instead of leaving
    parameter choice to the reader.
"""
from __future__ import annotations

import csv
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from creoh_routing import (ExpectedCost, OWARisk, ScenarioStability, FitnessVector,
                           ParetoArchive, p95, cliffs_delta, holm, dial_scores)
import creoh_scheduling as sched


# ==========================================================================
# Planner risk profiles
# ==========================================================================
@dataclass(frozen=True)
class RiskProfile:
    """An explicit planner preference over (expected cost, tail, stability).

    Utility is the negative of a normalised weighted sum; weights are stated in
    the paper so the profiles can be audited and replaced.  They are *not*
    fitted to the data.
    """
    label: str
    w_cost: float
    w_tail: float
    w_stability: float

    def utility(self, norm: np.ndarray) -> np.ndarray:
        """norm: (P, 3) array of normalised (f1, f2, f3), lower is better."""
        return -(self.w_cost * norm[:, 0] + self.w_tail * norm[:, 1]
                 + self.w_stability * norm[:, 2])


PROFILES = [
    RiskProfile("Cost-driven (budget-bound)", 0.80, 0.15, 0.05),
    RiskProfile("Balanced", 0.40, 0.40, 0.20),
    RiskProfile("Service-level (SLA-bound)", 0.20, 0.65, 0.15),
    RiskProfile("Overtime-averse", 0.25, 0.50, 0.25),
    RiskProfile("Stability-driven", 0.25, 0.25, 0.50),
    RiskProfile("Worst-case (regulatory)", 0.05, 0.85, 0.10),
]


# ==========================================================================
# Analyses
# ==========================================================================
class ArchiveView:
    """Normalised view of one seed's C-R-EoH archive plus the baseline point."""

    def __init__(self, pool, orness_ref: float = 0.7):
        f1, f2, f3 = ExpectedCost(), OWARisk(orness_ref), ScenarioStability()
        arch = ParetoArchive()
        for c in pool:
            arch.add(c.fitness(f1, f2, f3), c)
        self.members = [c for _fv, c in arch.items]
        self.raw = np.array([[f1(c.train), f2(c.train), f3(c.train)]
                             for c in self.members])
        lo, hi = self.raw.min(0), self.raw.max(0)
        self.norm = (self.raw - lo) / (np.ptp(self.raw, 0) + 1e-9)
        self.lo, self.hi = lo, hi
        det, _ = sched.DeterministicMethod().select(pool)
        self.det = det
        self.det_norm = (np.array([f1(det.train), f2(det.train), f3(det.train)])
                         - lo) / (np.ptp(self.raw, 0) + 1e-9)

    def select(self, theta: float) -> int:
        """The dial rule of the manuscript applied to the archive members.

        The tail objective is re-evaluated at the dial's own orness, so moving
        the dial changes both the risk measure and the scalarisation, exactly
        as in the interface.
        """
        f2 = OWARisk(theta)
        fvs = [FitnessVector(self.raw[i, 0], f2(c.train), self.raw[i, 2])
               for i, c in enumerate(self.members)]
        return int(np.argmin(dial_scores(fvs, theta)))


class PlannerAnalysis(ABC):
    """Base class for the per-seed analyses reported in the planner section."""

    name: str

    def __init__(self, seeds: int = 30, n: int = 40, budget: int = 120,
                 spread: float = 0.15):
        self.seeds, self.n, self.budget, self.spread = seeds, n, budget, spread

    def views(self, spread: float | None = None):
        gen = sched.ScheduleGenerator(n=self.n, spread=spread or self.spread)
        eb, lb, sb = (sched.EndpointBuilder(), sched.LatinHypercubeBuilder(),
                      sched.StressBuilder())
        for seed in range(self.seeds):
            inst = gen.generate(seed)
            train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
            ood = sb.build(inst, seed)
            pool = sched.CandidatePool(self.budget).grow(inst, train, ood, seed)
            yield seed, pool, ArchiveView(pool)

    @abstractmethod
    def run(self) -> dict: ...


class OrnessSweep(PlannerAnalysis):
    name = "orness_sweep"

    def __init__(self, theta_grid=None, **kw):
        super().__init__(**kw)
        self.theta_grid = list(theta_grid) if theta_grid is not None else \
            [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]

    def run(self) -> dict:
        acc = {t: {k: [] for k in ("f1", "f2", "f3", "p95", "buffer", "pack",
                                   "ood")} for t in self.theta_grid}
        f1, f3 = ExpectedCost(), ScenarioStability()
        classical = []
        for _seed, _pool, view in self.views():
            cl, _ = sched.ClassicalMethod().select(_pool)
            classical.append(f1(cl.train))
            for t in self.theta_grid:
                c = view.members[view.select(t)]
                f2 = OWARisk(t)
                acc[t]["f1"].append(f1(c.train)); acc[t]["f2"].append(f2(c.train))
                acc[t]["f3"].append(f3(c.train)); acc[t]["p95"].append(p95(c.train))
                acc[t]["buffer"].append(c.genome.buffer)
                acc[t]["pack"].append(c.genome.pack)
                acc[t]["ood"].append(100.0 * (c.ood.mean() - c.train.mean())
                                     / c.train.mean())
        # index units consistent with the main scheduling table: classical mean cost := 100
        scale = 100.0 / float(np.mean(classical))
        idx = ("f1", "f2", "f3", "p95")
        rows = []
        for t in self.theta_grid:
            d = {k: np.array(v, float) * (scale if k in idx else 1.0)
                 for k, v in acc[t].items()}
            rows.append(dict(theta=t, **{k: round(float(np.mean(v)), 4)
                                         for k, v in d.items()},
                             **{f"{k}_ci": round(float(1.96 * np.std(v, ddof=1)
                                                       / math.sqrt(len(v))), 4)
                                for k, v in d.items()}))
        thetas = np.array(self.theta_grid)
        rho_buf = spearmanr(np.repeat(thetas, self.seeds),
                            np.concatenate([acc[t]["buffer"] for t in thetas]))
        # knee of the mean response curve: point closest to the utopia corner
        x = np.array([r["f1"] for r in rows]); y = np.array([r["p95"] for r in rows])
        xs = (x - x.min()) / (np.ptp(x) + 1e-9); ys = (y - y.min()) / (np.ptp(y) + 1e-9)
        knee_idx = int(np.argmin(xs ** 2 + ys ** 2))
        return dict(rows=rows, theta_buffer_spearman=[round(float(rho_buf[0]), 4),
                                                      float(rho_buf[1])],
                    knee_theta=rows[knee_idx]["theta"],
                    index_scale="classical mean cost := 100")


class SelectionStability(PlannerAnalysis):
    name = "selection_stability"

    def __init__(self, theta_grid=None, **kw):
        super().__init__(**kw)
        self.theta_grid = list(theta_grid) if theta_grid is not None else \
            [round(x, 3) for x in np.arange(0.05, 0.951, 0.01)]

    def run(self) -> dict:
        distinct, plateau, switches = [], [], []
        per_theta_change = {t: 0 for t in self.theta_grid[1:]}
        for _seed, _pool, view in self.views():
            idx = [view.select(t) for t in self.theta_grid]
            distinct.append(len(set(idx)))
            sw = sum(1 for a, b in zip(idx, idx[1:]) if a != b)
            switches.append(sw)
            plateau.append(len(self.theta_grid) / max(1, len(set(idx))))
            for t, a, b in zip(self.theta_grid[1:], idx, idx[1:]):
                if a != b:
                    per_theta_change[t] += 1
        step = self.theta_grid[1] - self.theta_grid[0]
        return dict(
            mean_distinct_selections=round(float(np.mean(distinct)), 3),
            ci_distinct=round(float(1.96 * np.std(distinct, ddof=1)
                                    / math.sqrt(len(distinct))), 3),
            mean_switches=round(float(np.mean(switches)), 3),
            mean_plateau_width_theta=round(float(np.mean(plateau)) * step, 4),
            switch_thetas=sorted(((t, c) for t, c in per_theta_change.items()
                                  if c > 0), key=lambda kv: -kv[1])[:8],
            theta_grid_step=step)


class PlannerRegret(PlannerAnalysis):
    name = "planner_regret"

    def __init__(self, theta_map=None, **kw):
        super().__init__(**kw)
        #: dial setting each profile is assumed to choose, stated a priori
        self.theta_map = theta_map or {
            "Cost-driven (budget-bound)": 0.30,
            "Balanced": 0.50,
            "Service-level (SLA-bound)": 0.70,
            "Overtime-averse": 0.70,
            "Stability-driven": 0.60,
            "Worst-case (regulatory)": 0.90,
        }

    def run(self) -> dict:
        res = {p.label: dict(regret_dial=[], regret_det=[], regret_random=[],
                             oracle_util=[]) for p in PROFILES}
        rng = np.random.default_rng(11)
        for _seed, _pool, view in self.views():
            for p in PROFILES:
                u = p.utility(view.norm)
                oracle = float(u.max())
                dial = float(u[view.select(self.theta_map[p.label])])
                det = float(p.utility(view.det_norm[None, :])[0])
                rand = float(u[rng.integers(0, len(u))])
                res[p.label]["regret_dial"].append(oracle - dial)
                res[p.label]["regret_det"].append(oracle - det)
                res[p.label]["regret_random"].append(oracle - rand)
                res[p.label]["oracle_util"].append(oracle)
        out, raw_p, stats = {}, {}, []
        for p in PROFILES:
            d = res[p.label]
            out[p.label] = {k: [round(float(np.mean(v)), 4),
                                round(float(1.96 * np.std(v, ddof=1)
                                            / math.sqrt(len(v))), 4)]
                            for k, v in d.items()}
            try:
                _, pv = wilcoxon(d["regret_dial"], d["regret_det"])
            except ValueError:
                pv = 1.0
            raw_p[p.label] = float(pv)
            stats.append([p.label, float(pv),
                          cliffs_delta(d["regret_dial"], d["regret_det"])])
        adj = holm(raw_p)
        for r in stats:
            r.insert(2, adj[r[0]])
        return dict(profiles=[[p.label, p.w_cost, p.w_tail, p.w_stability]
                              for p in PROFILES],
                    theta_map=self.theta_map, regret=out,
                    stats=[[r[0], r[1], r[2], round(r[3], 3)] for r in stats])


class ParameterGuidance(PlannerAnalysis):
    name = "parameter_guidance"

    def __init__(self, theta_grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                 spreads=(0.08, 0.15, 0.25), **kw):
        super().__init__(**kw)
        self.theta_grid, self.spreads = list(theta_grid), list(spreads)

    def run(self) -> dict:
        grid = {}
        for delta in self.spreads:
            per = {p.label: {t: [] for t in self.theta_grid} for p in PROFILES}
            for _seed, _pool, view in self.views(spread=delta):
                for p in PROFILES:
                    u = p.utility(view.norm)
                    oracle = float(u.max())
                    for t in self.theta_grid:
                        per[p.label][t].append(oracle - float(u[view.select(t)]))
            grid[delta] = {lab: {t: round(float(np.mean(v)), 4)
                                 for t, v in d.items()} for lab, d in per.items()}
        best = {delta: {lab: min(d, key=d.get) for lab, d in g.items()}
                for delta, g in grid.items()}
        return dict(grid={str(k): v for k, v in grid.items()},
                    recommended_theta={str(k): v for k, v in best.items()},
                    spreads=self.spreads, theta_grid=self.theta_grid)


# ==========================================================================
# Driver
# ==========================================================================
def run(seeds: int = 30, outdir: str = "data") -> dict:
    os.makedirs(outdir, exist_ok=True)
    sweep = OrnessSweep(seeds=seeds).run()
    stab = SelectionStability(seeds=seeds).run()
    regret = PlannerRegret(seeds=seeds).run()
    guide = ParameterGuidance(seeds=seeds).run()

    with open(os.path.join(outdir, "planner_orness_sweep.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep["rows"][0].keys()))
        w.writeheader(); w.writerows(sweep["rows"])
    with open(os.path.join(outdir, "planner_regret.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["profile", "w_cost", "w_tail", "w_stability", "theta",
                    "regret_dial", "regret_dial_ci", "regret_deterministic",
                    "regret_deterministic_ci", "regret_random",
                    "wilcoxon_p_holm", "cliffs_delta"])
        st = {r[0]: r for r in regret["stats"]}
        for lab, wc, wt, ws in regret["profiles"]:
            r = regret["regret"][lab]
            w.writerow([lab, wc, wt, ws, regret["theta_map"][lab],
                        r["regret_dial"][0], r["regret_dial"][1],
                        r["regret_det"][0], r["regret_det"][1],
                        r["regret_random"][0],
                        f"{st[lab][2]:.6f}", f"{st[lab][3]:.3f}"])
    with open(os.path.join(outdir, "planner_parameter_guidance.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spread_delta", "profile"] + [f"theta={t}" for t in
                                                  guide["theta_grid"]]
                   + ["recommended_theta"])
        for delta in guide["spreads"]:
            g = guide["grid"][str(delta)]
            for lab in g:
                w.writerow([delta, lab] + [g[lab][t] for t in guide["theta_grid"]]
                           + [guide["recommended_theta"][str(delta)][lab]])

    meta = dict(seeds=seeds, orness_sweep=sweep, selection_stability=stab,
                planner_regret=regret, parameter_guidance=guide)
    with open(os.path.join(outdir, "planner_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"# planner decision analysis, {seeds} seeds")
    print(f"{'theta':>6} {'f1':>8} {'P95':>8} {'stability':>10} {'buffer':>8} "
          f"{'OOD%':>7}")
    for r in sweep["rows"]:
        print(f"{r['theta']:6.2f} {r['f1']:8.2f} {r['p95']:8.2f} {r['f3']:10.2f} "
              f"{r['buffer']:8.3f} {r['ood']:7.2f}")
    print(f"\ndial->buffer Spearman rho = {sweep['theta_buffer_spearman'][0]:+.3f}"
          f" (p={sweep['theta_buffer_spearman'][1]:.2e}); "
          f"knee at theta = {sweep['knee_theta']}")
    print(f"distinct selections over the dial range: "
          f"{stab['mean_distinct_selections']} +- {stab['ci_distinct']}, "
          f"mean plateau width {stab['mean_plateau_width_theta']:.3f} in theta")
    print("\nplanner regret (lower is better):")
    print(f"{'profile':30s} {'dial':>10} {'determ.':>10} {'random':>10} "
          f"{'p_holm':>10} {'delta':>7}")
    st = {r[0]: r for r in regret["stats"]}
    for lab, *_ in regret["profiles"]:
        r = regret["regret"][lab]
        print(f"{lab:30s} {r['regret_dial'][0]:10.4f} {r['regret_det'][0]:10.4f} "
              f"{r['regret_random'][0]:10.4f} {st[lab][2]:10.2e} "
              f"{st[lab][3]:+7.3f}")
    print("\nrecommended theta by uncertainty spread:")
    for delta, d in guide["recommended_theta"].items():
        print(f"  delta={delta}: " + ", ".join(f"{k.split(' ')[0]}={v}"
                                               for k, v in d.items()))
    return meta


if __name__ == "__main__":
    run(outdir="data")
