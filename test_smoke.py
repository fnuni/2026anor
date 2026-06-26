"""Minimal smoke tests: run with `python -m pytest test_smoke.py` or `python test_smoke.py`."""
import numpy as np
import creoh_routing as R
import creoh_scheduling as S


def test_owa_tail_premium():
    # Proposition 1: upper-tail OWA (f2) >= mean (f1) for any cost vector.
    rng = np.random.default_rng(0)
    f1, f2 = R.ExpectedCost(), R.OWARisk(0.7)
    for _ in range(200):
        c = rng.uniform(1, 100, size=rng.integers(2, 40))
        assert f2(c) >= f1(c) - 1e-9


def test_pareto_dominance():
    a = R.ParetoArchive()
    a.add(R.FitnessVector(1.0, 1.0, 1.0), "x")
    a.add(R.FitnessVector(2.0, 2.0, 2.0), "dominated")  # should be rejected
    a.add(R.FitnessVector(0.5, 3.0, 1.0), "nd")          # non-dominated, kept
    payloads = {p for _, p in a.items}
    assert "dominated" not in payloads and "x" in payloads and "nd" in payloads


def test_scheduling_robust_beats_nominal_on_tail():
    # Quick 3-seed run: C-R-EoH must lower P95 vs deterministic AHD.
    h = S.run(seeds=3, n=30, budget=60, outdir="/tmp/_creoh_smoke")
    assert h["p95_reduction_pct"] > 3.0
    assert h["friedman_p"] < 0.05


if __name__ == "__main__":
    test_owa_tail_premium(); test_pareto_dominance(); test_scheduling_robust_beats_nominal_on_tail()
    print("all smoke tests passed")
