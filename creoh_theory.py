"""Analytical properties and the robustness-opportunity index (revision).

Reviewers observed that the first submission's contribution was algorithmic and
that its central empirical finding -- the fuzzy evaluator pays off on scheduling
but not on routing -- was reported as an observation rather than explained.
This module supplies the analytical half and the numerical verification of it.

Propositions proved in the manuscript and checked numerically here
-----------------------------------------------------------------
P1  *Tail premium.*  For the tail-averaging OWA with cutoff
    ``q(theta) = ceil((1-theta) M)``, ``f2 >= f1`` for every candidate, with
    equality iff the scenario costs are constant or ``q = M``.

P2  *CVaR identity.*  On the empirical distribution of ``M`` equally likely
    scenarios, ``f2^theta = CVaR_beta`` with ``beta = 1 - q(theta)/M``.

P3  *Dial monotonicity.*  ``theta -> f2^theta(h)`` is non-decreasing for every
    candidate ``h``: turning the dial towards worst-case never reports a lower
    risk.

P4  *No risk reversal.*  If ``c(h)`` dominates ``c(g)`` in the increasing convex
    order, then ``f2^theta(h) >= f2^theta(g)`` for every ``theta``.  A planner
    therefore cannot reverse, by moving the dial, a preference shared by all
    risk-averse planners.

P5  *Archive monotonicity.*  The archive update rule makes the hypervolume
    non-decreasing along the search and bounded above by the hypervolume of the
    true Pareto set, hence convergent.

P6  *Structure of the price of robustness.*  In the field-service model the
    total regular work is invariant across feasible schedules, so the nominal
    premium of a robust schedule over a nominal-optimal one is
    ``[(K_R - K_D) c_tech + rho (O_R - O_D)] / C_D``, and since a buffered
    schedule has ``O_R <= O_D`` on the modal instance the premium is bounded by
    the extra technician-opening cost alone.

P7  *Rank invariance (when robust evaluation cannot pay off).*  If the cost is
    positively homogeneous of degree one in the uncertain parameters and the
    uncertainty is single-zone common mode, then every scenario cost satisfies
    ``c_m(h) = lambda_m c_nom(h)`` with ``lambda_m`` independent of the
    candidate.  All of ``f1``, ``f2`` and ``P95`` then induce the *same* ranking
    of candidates as the nominal cost, and no evaluator that selects from the
    pool can improve the tail.  Fixed, uncertainty-free cost components and
    several zones with candidate-dependent exposure are exactly what break the
    invariance.

The Robustness Opportunity Index
--------------------------------
P7 suggests a cheap, a-priori diagnostic.  For a candidate pool ``P`` on one
instance,

    ROI(P) = [ T(h_nom) - min_h T(h) ] / T(h_nom),   h_nom = argmin_h c_nom(h),

with ``T`` the tail measure (``f2`` or ``P95``) and ``c_nom`` the cost on the
modal instance, i.e. the program the deterministic evaluator would deploy.  ROI
is the relative tail reduction obtainable *within the pool* by moving from the
nominal-optimal program to the tail-optimal one; it is an upper bound on the
reduction any selection rule can deliver on that pool (P8), it costs one extra
pass over an already-evaluated pool, and under the hypotheses of P7 it is zero.
The module computes ROI and the reduction actually realised by C-R-EoH *on the
same pools*, checks the bound instance by instance, and reports the
instance-level rank correlation between the two.
"""
from __future__ import annotations

import csv
import json
import math
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import spearmanr

from creoh_routing import (ExpectedCost, OWARisk, ScenarioStability, p95,
                           FuzzyInstanceGenerator, EndpointBuilder,
                           LatinHypercubeBuilder, StressBuilder,
                           CandidatePool as RoutingPool)
import creoh_routing as rt
import creoh_scheduling as sched
import creoh_public as pub


# ==========================================================================
# Numerical verification of the propositions
# ==========================================================================
class PropositionCheck(ABC):
    """A proposition of the manuscript, verified numerically on random instances."""

    label: str

    @abstractmethod
    def check(self, rng) -> bool: ...

    def run(self, trials: int = 500, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        failures = sum(0 if self.check(rng) else 1 for _ in range(trials))
        return dict(proposition=self.label, trials=trials, failures=failures,
                    holds=failures == 0)


class TailPremiumCheck(PropositionCheck):
    label = "P1 tail premium f2 >= f1"

    def check(self, rng):
        c = rng.uniform(1, 200, size=int(rng.integers(2, 60)))
        th = float(rng.uniform(0.01, 0.99))
        return OWARisk(th)(c) >= ExpectedCost()(c) - 1e-9


class CVaRIdentityCheck(PropositionCheck):
    label = "P2 OWA equals empirical CVaR"

    def check(self, rng):
        M = int(rng.integers(4, 80))
        c = rng.uniform(1, 200, size=M)
        th = float(rng.uniform(0.01, 0.99))
        q = max(1, int(math.ceil((1 - th) * M)))
        beta = 1 - q / M
        # the identity is between integer cutoffs; recovering q from beta in
        # floating point needs a tolerance, otherwise ceil() rounds up on the
        # representation error of (1 - beta) * M
        k = max(1, int(math.ceil((1 - beta) * M - 1e-9)))
        cvar = float(np.sort(c)[::-1][:k].mean())
        return k == q and abs(OWARisk(th)(c) - cvar) < 1e-9


class DialMonotonicityCheck(PropositionCheck):
    label = "P3 f2 non-decreasing in theta"

    def check(self, rng):
        c = rng.uniform(1, 200, size=int(rng.integers(3, 60)))
        ths = np.sort(rng.uniform(0.01, 0.99, size=8))
        vals = [OWARisk(float(t))(c) for t in ths]
        return all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


class NoRiskReversalCheck(PropositionCheck):
    label = "P4 consistency with the increasing convex order"

    def check(self, rng):
        """Draw a pair, test the icx hypothesis, and verify the conclusion.

        For two cost vectors of equal length on a uniform scenario measure,
        ``h`` dominates ``g`` in the increasing convex order exactly when every
        top-``k`` partial sum of the sorted vectors satisfies
        ``sum_k h_(i) >= sum_k g_(i)``.  A randomly perturbed vector need not
        satisfy this, so the hypothesis is *tested* rather than assumed: pairs
        that fail it are skipped (the proposition says nothing about them) and
        pairs that satisfy it must satisfy the conclusion.
        """
        M = int(rng.integers(4, 40))
        g = rng.uniform(10, 100, size=M)
        eps = rng.uniform(0, 20, size=M)
        eps = eps - eps.mean()
        h = g + eps + float(rng.uniform(0, 5))
        hs = np.sort(h)[::-1]; gs = np.sort(g)[::-1]
        icx = all(hs[:k].sum() >= gs[:k].sum() - 1e-9 for k in range(1, M + 1))
        if not icx:
            return True                        # hypothesis not met: vacuous
        for t in rng.uniform(0.01, 0.99, size=5):
            if OWARisk(float(t))(hs) < OWARisk(float(t))(gs) - 1e-9:
                return False
        return True


class ArchiveMonotonicityCheck(PropositionCheck):
    label = "P5 hypervolume non-decreasing along the search"

    def check(self, rng):
        arch = rt.ParetoArchive()
        prev = -1.0
        for _ in range(30):
            fv = rt.FitnessVector(*rng.uniform(0.0, 1.0, size=3))
            arch.add(fv, None)
            hv = arch.hypervolume((1.05, 1.05))
            if hv < prev - 1e-9:
                return False
            prev = hv
        return True


class RankInvarianceCheck(PropositionCheck):
    """P7 on a cost model satisfying its hypotheses exactly.

    The proposition is a statement about the cost model, so it is verified on
    the cost model rather than through a particular heuristic family: ``P``
    candidates with arbitrary nominal costs, a single common-mode zone, and a
    cost positively homogeneous of degree one in the uncertain parameters give
    ``c_m(h) = lambda_m c_nom(h)``.  Under these hypotheses the nominal, mean,
    OWA and P95 rankings must coincide and ROI must vanish.
    """

    label = "P7 rank invariance under single-zone homogeneous cost"

    def check(self, rng):
        P = int(rng.integers(5, 40))
        M = int(rng.integers(4, 50))
        nominal = rng.uniform(50, 500, size=P)
        lam = 1.0 + rng.uniform(-0.3, 0.9, size=M)      # common-mode factors
        C = nominal[:, None] * lam[None, :]
        f1 = C.mean(axis=1)
        f2 = np.array([OWARisk(0.7)(row) for row in C])
        tail = np.array([p95(row) for row in C])
        order = np.argsort(nominal, kind="stable")
        ok = (np.all(np.diff(f1[order]) >= -1e-9)
              and np.all(np.diff(f2[order]) >= -1e-9)
              and np.all(np.diff(tail[order]) >= -1e-9))
        roi = (f2[int(np.argmin(f1))] - f2.min()) / f2[int(np.argmin(f1))]
        return ok and roi < 1e-9


class PremiumDecompositionCheck(PropositionCheck):
    """P6 on random field-service instances and random feasible partitions."""

    label = "P6 nominal premium decomposition"

    def check(self, rng):
        n = int(rng.integers(6, 40))
        p = rng.uniform(5, 20, size=n)
        S, rho, ct = 100.0, float(rng.uniform(0.5, 3.0)), 40.0

        def cost(parts):
            L = np.array([p[m].sum() for m in parts])
            return len(parts) * ct + L.sum() + rho * np.maximum(0, L - S).sum(), \
                len(parts), np.maximum(0, L - S).sum()

        def partition():
            K = int(rng.integers(1, n + 1))
            lab = rng.integers(0, K, size=n)
            return [np.where(lab == k)[0] for k in range(K) if (lab == k).any()]

        cD, KD, OD = cost(partition())
        cR, KR, OR = cost(partition())
        return abs((cR - cD) - ((KR - KD) * ct + rho * (OR - OD))) < 1e-7


class OpportunityBoundCheck(PropositionCheck):
    """P8: no selection rule beats ROI on the pool it is computed from."""

    label = "P8 opportunity index is an upper bound"

    def check(self, rng):
        P, M = int(rng.integers(3, 40)), int(rng.integers(4, 60))
        C = rng.uniform(50, 200, size=(P, M))
        nominal = C[:, 0]
        tail = np.array([OWARisk(0.7)(row) for row in C])
        ref = tail[int(np.argmin(nominal))]
        roi = (ref - tail.min()) / ref
        j = int(rng.integers(0, P))                 # an arbitrary selection rule
        return (ref - tail[j]) / ref <= roi + 1e-12


class InvarianceBreakingStudy:
    """Corollary to P7: which hypothesis failure creates the opportunity.

    P7 needs two things: a cost positively homogeneous of degree one in the
    uncertain parameters, and a common-mode uncertainty to which all candidates
    are equally exposed.  Operationally the second fails when candidates route
    or load different congestion zones, and the first fails when the cost has a
    *kink*, as the convex overtime penalty
    ``rho * max(0, L - S)`` does.  This study varies the two independently in a
    transparent surrogate cost model -- ``K`` resources with nominal loads drawn
    around a fraction ``u`` of the shift, zone shocks, and overtime rate
    ``rho`` -- and reports the resulting ROI.  It separates the contribution of
    exposure dispersion from the contribution of convexity, and shows that the
    convexity term is the dominant one, which is precisely the difference
    between the scheduling family (convex overtime) and the routing family
    (travel cost linear in the travel times).
    """

    def __init__(self, trials: int = 1200, P: int = 60, M: int = 40,
                 K: int = 6, zones: int = 6, seed: int = 3):
        self.trials, self.P, self.M = trials, P, M
        self.K, self.zones, self.seed = K, zones, seed

    def _roi(self, rng, sigma: float, rho: float, utilisation: float) -> float:
        """Surrogate of the field-service trade-off.

        Total work ``W`` is fixed, so a candidate is characterised by how many
        resources it opens: fewer resources means a lower opening cost and a
        higher utilisation, hence more exposure to the kink; more resources
        means the opposite.  This is exactly the packing decision the heuristic
        family searches over, reduced to its essentials.
        """
        Z, P, M = self.zones, self.P, self.M
        shift, c_res = 100.0, 40.0
        k_star = max(1, int(round(self.K / max(utilisation, 1e-6))))
        Ks = rng.integers(max(1, k_star - 2), k_star + 4, size=P)
        W = self.K * shift * utilisation
        conc = np.ones(Z) * (1.0 / max(sigma, 1e-6))
        base = rng.dirichlet(np.ones(Z))
        shocks = rng.uniform(-0.3, 0.9, size=(M, Z))
        C = np.empty((P, M))
        nominal = np.empty(P)
        for h in range(P):
            K = int(Ks[h])
            loads = (W / K) * rng.uniform(0.9, 1.1, size=K)
            loads *= W / loads.sum()
            prof = (np.array([rng.dirichlet(conc) for _ in range(K)])
                    if sigma > 0 else np.tile(base, (K, 1)))
            realised = loads[:, None] * (1.0 + prof @ shocks.T)      # (K, M)
            over = np.maximum(0.0, realised - shift)
            C[h] = K * c_res + realised.sum(axis=0) + rho * over.sum(axis=0)
            nominal[h] = K * c_res + loads.sum() + rho * np.maximum(0.0, loads - shift).sum()
        f2 = np.array([OWARisk(0.7)(row) for row in C])
        ref = f2[int(np.argmin(nominal))]
        return float((ref - f2.min()) / ref) if ref > 0 else 0.0

    def run(self, exposure_spreads=(0.0, 0.25, 0.5, 1.0),
            overtime_rates=(0.0, 0.5, 1.5, 3.0),
            utilisation: float = 0.95) -> list[dict]:
        rng = np.random.default_rng(self.seed)
        out = []
        for sigma in exposure_spreads:
            for rho in overtime_rates:
                rois = [self._roi(rng, sigma, rho, utilisation)
                        for _ in range(self.trials)]
                out.append(dict(exposure_spread=sigma, overtime_rate=rho,
                                roi_mean=round(float(np.mean(rois)), 5),
                                roi_ci=round(float(1.96 * np.std(rois, ddof=1)
                                                   / math.sqrt(len(rois))), 5)))
        return out


CHECKS = [TailPremiumCheck, CVaRIdentityCheck, DialMonotonicityCheck,
          NoRiskReversalCheck, ArchiveMonotonicityCheck, PremiumDecompositionCheck,
          RankInvarianceCheck, OpportunityBoundCheck]


# ==========================================================================
# Robustness Opportunity Index
# ==========================================================================
class OpportunityMeasure:
    """ROI, rank correlation and the realised C-R-EoH reduction on one pool."""

    def __init__(self, orness: float = 0.7):
        self.orness = orness
        self.f2 = OWARisk(orness)

    def measure(self, pool, method) -> dict:
        nom = np.array([c.train[0] for c in pool])
        v2 = np.array([self.f2(c.train) for c in pool])
        vt = np.array([p95(c.train) for c in pool])
        k = int(np.argmin(nom))                     # deterministic AHD choice
        roi = float((v2[k] - v2.min()) / v2[k])
        roi_tail = float((vt[k] - vt.min()) / vt[k])
        sel, _ = method.select(pool)
        realised = float((vt[k] - p95(sel.train)) / vt[k])
        rho = float(spearmanr(nom, v2).statistic) if len(set(np.round(nom, 9))) > 2 \
            else 1.0
        return dict(roi=roi, roi_p95=roi_tail, realised_p95=realised,
                    rank_corr=rho, pool_size=len(pool),
                    bound_ok=realised <= roi_tail + 1e-12)


class FamilyProbe(ABC):
    """Computes the opportunity index for one instance family."""

    label: str

    def __init__(self, seeds: int = 15, orness: float = 0.7):
        self.seeds, self.orness = seeds, orness
        self.measure = OpportunityMeasure(orness)

    @abstractmethod
    def pools(self): ...

    @abstractmethod
    def method(self): ...

    def run(self) -> dict:
        vals = [self.measure.measure(p, self.method()) for p in self.pools()]
        agg = {}
        for key in ("roi", "roi_p95", "realised_p95", "rank_corr"):
            arr = np.array([v[key] for v in vals], float)
            agg[key] = [float(arr.mean()),
                        float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))]
        agg["units"] = len(vals)
        agg["family"] = self.label
        agg["bound_violations"] = int(sum(not v["bound_ok"] for v in vals))
        agg["instances"] = [[round(v["roi_p95"], 6), round(v["realised_p95"], 6)]
                            for v in vals]
        return agg


class SyntheticSchedulingProbe(FamilyProbe):
    label = "Synthetic scheduling (primary)"

    def pools(self):
        gen = sched.ScheduleGenerator(n=40, spread=0.15)
        eb, lb, sb = (sched.EndpointBuilder(), sched.LatinHypercubeBuilder(),
                      sched.StressBuilder())
        for seed in range(self.seeds):
            inst = gen.generate(seed)
            tr = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
            yield sched.CandidatePool(120).grow(inst, tr, sb.build(inst, seed), seed)

    def method(self):
        return sched.CREoHMethod(self.orness)


class SyntheticRoutingProbe(FamilyProbe):
    label = "Synthetic routing (contrast)"

    def pools(self):
        gen = FuzzyInstanceGenerator(n=25, spread=0.15, n_regions=6)
        eb, lb, sb = (EndpointBuilder(reps=3), LatinHypercubeBuilder(m=24),
                      StressBuilder(m=12, extra=1.6))
        for seed in range(self.seeds):
            inst = gen.generate(seed)
            tr = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
            yield RoutingPool(80).grow(inst, tr, sb.build(inst, seed), seed)

    def method(self):
        return rt.CREoHMethod(self.orness)


class PublicRoutingProbe(FamilyProbe):
    def __init__(self, reader, label, limit=15, **kw):
        super().__init__(**kw)
        self.reader, self.label, self.limit = reader, label, limit
        self.inj = pub.RoutingFuzzyInjector(0.15)
        self.eb, self.lb, self.sb = (EndpointBuilder(reps=3),
                                     LatinHypercubeBuilder(m=24),
                                     StressBuilder(m=12, extra=1.6))

    def pools(self):
        raws = pub.unique_routing_instances(self.reader.read_all())
        for raw in raws[:self.limit]:
            inst = self.inj.inject(raw, 0)
            tr = np.concatenate([self.eb.build(inst, 0), self.lb.build(inst, 0)])
            yield RoutingPool(60).grow(inst, tr, self.sb.build(inst, 0), 0)

    def method(self):
        return rt.CREoHMethod(self.orness)


class PublicSchedulingProbe(FamilyProbe):
    label = "Solomon-derived scheduling (public)"

    def __init__(self, limit=20, **kw):
        super().__init__(**kw)
        self.limit = limit
        self.inj = pub.SolomonSchedulingInjector(0.15)
        self.eb, self.lb, self.sb = (sched.EndpointBuilder(),
                                     sched.LatinHypercubeBuilder(),
                                     sched.StressBuilder())

    def pools(self):
        raws = pub.unique_scheduling_instances(pub.SolomonReader().read_all())
        for raw in raws[:self.limit]:
            inst = self.inj.inject(raw, 0)
            tr = np.concatenate([self.eb.build(inst, 0), self.lb.build(inst, 0)])
            yield sched.CandidatePool(120).grow(inst, tr, self.sb.build(inst, 0), 0)

    def method(self):
        return sched.CREoHMethod(self.orness)


# ==========================================================================
# Driver
# ==========================================================================
def run(outdir: str = "data", trials: int = 500) -> dict:
    os.makedirs(outdir, exist_ok=True)
    props = [C().run(trials=trials) for C in CHECKS]
    breaking = InvarianceBreakingStudy().run()

    probes = [SyntheticSchedulingProbe(),
              SyntheticRoutingProbe(),
              PublicSchedulingProbe(),
              PublicRoutingProbe(pub.CVRPLIBReader(), "CVRPLIB X-set (public)"),
              PublicRoutingProbe(pub.SolomonReader(), "Solomon routing (public)")]
    fam = [p.run() for p in probes]

    # instance-level association between ROI and the realised reduction
    pts = [pt for f in fam for pt in f["instances"]]
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    rho, pv = spearmanr(x, y)
    fam_x = np.array([f["roi_p95"][0] for f in fam])
    fam_y = np.array([f["realised_p95"][0] for f in fam])
    assoc = dict(instance_spearman=round(float(rho), 4), p_value=float(pv),
                 units=len(pts), n_families=len(fam),
                 bound_violations=int(sum(f["bound_violations"] for f in fam)),
                 captured_share=[round(float(a / b), 4) if b > 0 else None
                                 for a, b in zip(fam_y, fam_x)])

    with open(os.path.join(outdir, "theory_propositions.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(props[0].keys()))
        w.writeheader(); w.writerows(props)
    with open(os.path.join(outdir, "theory_invariance_breaking.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(breaking[0].keys()))
        w.writeheader(); w.writerows(breaking)
    with open(os.path.join(outdir, "theory_opportunity_index.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "units", "roi_mean", "roi_ci", "roi_p95_mean",
                    "roi_p95_ci", "realised_p95_mean", "realised_p95_ci",
                    "rank_corr_mean", "rank_corr_ci", "bound_violations"])
        for f in fam:
            w.writerow([f["family"], f["units"], f"{f['roi'][0]:.4f}",
                        f"{f['roi'][1]:.4f}", f"{f['roi_p95'][0]:.4f}",
                        f"{f['roi_p95'][1]:.4f}", f"{f['realised_p95'][0]:.4f}",
                        f"{f['realised_p95'][1]:.4f}", f"{f['rank_corr'][0]:.4f}",
                        f"{f['rank_corr'][1]:.4f}", f["bound_violations"]])
    meta = dict(propositions=props, invariance_breaking=breaking,
                families=fam, association=assoc)
    with open(os.path.join(outdir, "theory_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("# analytical properties, numerically verified")
    for p in props:
        print(f"  {p['proposition']:52s} {p['trials']:4d} trials, "
              f"{p['failures']} failures -> {'holds' if p['holds'] else 'FAILS'}")
    print("\n# what breaks rank invariance: ROI (%) by exposure spread "
          "and convexity")
    rhos = sorted({r["overtime_rate"] for r in breaking})
    print(f"{'sigma':>8}  " + "".join(f"rho={r:<10.1f}" for r in rhos))
    for sg in sorted({r["exposure_spread"] for r in breaking}):
        row = {r["overtime_rate"]: r["roi_mean"] for r in breaking
               if r["exposure_spread"] == sg}
        print(f"{sg:8.2f}  " + "".join(f"{100*row[r]:<14.2f}" for r in rhos))
    print("\n# robustness opportunity index by instance family")
    print(f"{'family':38s} {'units':>6} {'ROI(OWA)':>10} {'ROI(P95)':>10} "
          f"{'realised':>10} {'rank corr':>10} {'viol.':>6}")
    for f in fam:
        print(f"{f['family']:38s} {f['units']:6d} "
              f"{100*f['roi'][0]:9.2f}% {100*f['roi_p95'][0]:9.2f}% "
              f"{100*f['realised_p95'][0]:9.2f}% {f['rank_corr'][0]:10.3f} "
              f"{f['bound_violations']:6d}")
    print(f"\ninstance-level Spearman(ROI, realised) = "
          f"{assoc['instance_spearman']:+.3f} (p = {assoc['p_value']:.3g}, "
          f"{assoc['units']} instances)")
    return meta


if __name__ == "__main__":
    run(outdir="data")
