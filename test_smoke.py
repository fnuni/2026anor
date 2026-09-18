"""Minimal smoke tests: run with `python -m pytest test_smoke.py` or `python test_smoke.py`."""
import numpy as np
import creoh_routing as R
import creoh_scheduling as S
import creoh_llm_pilot as L


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


def test_hypervolume_matches_monte_carlo():
    """The 2-D archive hypervolume must agree with a Monte Carlo estimate.

    This guards the sweep corrected in the revision (see
    ``ParetoArchive.hypervolume``): the previous implementation returned the
    rectangle of the lowest-cost archive member only.
    """
    rng = np.random.default_rng(7)
    ref = (1.05, 1.05)
    for _ in range(20):
        pts = rng.uniform(0.0, 1.0, size=(rng.integers(2, 12), 2))
        arch = R.ParetoArchive()
        for x, y in pts:
            arch.add(R.FitnessVector(float(x), float(y), 0.0), None)
        hv = arch.hypervolume(ref)
        keep = [(fv.f1, fv.f2) for fv, _ in arch.items]
        sample = rng.uniform([0, 0], list(ref), size=(200_000, 2))
        dom = np.zeros(len(sample), dtype=bool)
        for x, y in keep:
            dom |= (sample[:, 0] >= x) & (sample[:, 1] >= y)
        mc = dom.mean() * ref[0] * ref[1]
        assert abs(hv - mc) < 0.01, (hv, mc)


def test_sandbox_rejects_policy_probes():
    """The static policy, the timeout and the repair policy must all fire.

    The probes live in ``llm_candidates/policy_probes/`` and are author-written
    adversarial inputs, not language-model output; they exist so that the
    rejection and repair paths exercised by the pilot are demonstrably live.
    """
    import os
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "llm_candidates", "policy_probes")
    sb = L.Sandbox(timeout_s=0.5)
    expected_rejection = ["probe_forbidden_import.py", "probe_file_access.py",
                          "probe_dunder_attribute.py", "probe_syntax_error.py"]
    for fname in expected_rejection:
        src = open(os.path.join(root, fname)).read()
        try:
            sb.load(src, fname)
        except L.SandboxViolation:
            continue
        raise AssertionError(f"{fname} was not rejected by the static policy")

    # non-terminating code must be stopped by the execution timeout
    fn = sb.load(open(os.path.join(root, "probe_nonterminating.py")).read(), "nt")
    try:
        sb.call(fn, {"n": 4}, {})
    except L._Timeout:
        pass
    else:                                            # pragma: no cover
        raise AssertionError("non-terminating program was not interrupted")

    # duplicated and omitted tasks are repaired; out-of-range ids are rejected
    rp = L.RepairPolicy()
    fn = sb.load(open(os.path.join(root,
                                   "probe_duplicate_and_missing.py")).read(), "dm")
    out = rp.apply(sb.call(fn, {"n": 5}, {}), 5)
    assert out.status == "ok" and out.repaired
    flat = sorted(j for m in out.machines for j in m)
    assert flat == list(range(5))
    fn = sb.load(open(os.path.join(root, "probe_out_of_range.py")).read(), "oor")
    assert rp.apply(sb.call(fn, {"n": 5}, {}), 5).status == "infeasible"


def test_llm_pilot_programs_are_admitted():
    """Every released LLM response must load under the static policy."""
    progs = L.LLMProposerPilot(seeds=1).admit(L.ProgramLoader().load())
    assert progs, "no generated programs found"
    bad = [(p.name, p.detail) for p in progs if p.outcome != "loaded"]
    assert not bad, bad


def test_owa_tail_count_is_exact():
    # (1 - 0.7) * 40 = 12.000000000000002 in floating point; q must be 12
    assert R.tail_count(0.7, 40) == 12 and R.tail_count(0.7, 20) == 6


def test_balance_gene_changes_assignment():
    inst = S.ScheduleGenerator().generate(0)
    tight = S.DispatchHeuristic(S.Genome(1.0, 0.0, 0.0)).assign(inst)
    balanced = S.DispatchHeuristic(S.Genome(1.0, 0.0, 1.0)).assign(inst)
    assert tight != balanced


def test_ablation_full_is_the_reported_method():
    import numpy as np
    import creoh_ablation as A
    gen = S.ScheduleGenerator()
    eb, lb, sb = S.EndpointBuilder(), S.LatinHypercubeBuilder(), S.StressBuilder()
    for seed in range(3):
        inst = gen.generate(seed)
        tr = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
        pool = S.CandidatePool(60).grow(inst, tr, sb.build(inst, seed), seed)
        main, _ = S.CREoHMethod(0.7).select(pool)
        full, _ = A.FullCREoH(0.7).select(pool)
        assert main is full


def test_scheduling_robust_beats_nominal_on_tail():
    # Quick 3-seed run: C-R-EoH must lower P95 vs deterministic AHD.
    h = S.run(seeds=3, n=30, budget=60, outdir="/tmp/_creoh_smoke")
    assert h["p95_reduction_pct"] > 2.0
    assert h["friedman_p"] < 0.05


def test_repair_rejects_coercion_and_records_empty_technicians():
    rp = L.RepairPolicy()
    for task_id in (0.5, 0.0, "0", True):
        assert rp.apply([[task_id]], 1).status == "infeasible"
    out = rp.apply([[0], []], 1)
    assert out.status == "ok" and out.repaired and out.machines == [[0]]
    # Task count prefers machine 0; actual nominal load correctly prefers 1.
    out = rp.apply([[0], [1, 2]], 4, [10, 1, 1, 3])
    assert out.machines == [[0], [1, 2, 3]]


def test_timeout_is_not_swallowed_by_generated_exception_handler():
    sb = L.Sandbox(timeout_s=0.05)
    fn = sb.load("def solve(instance, params):\n    try:\n        while True: pass\n    except Exception:\n        return [[0]]\n", "catch_exception")
    try:
        sb.call(fn, {}, {})
    except L._Timeout:
        pass
    else:
        raise AssertionError("generated handler swallowed timeout")
    try:
        sb.load("def solve(instance, params):\n    try: return [[0]]\n    except: return [[0]]\n", "bare_except")
    except L.SandboxViolation:
        pass
    else:
        raise AssertionError("bare handler admitted")


if __name__ == "__main__":
    test_owa_tail_premium(); test_pareto_dominance()
    test_hypervolume_matches_monte_carlo()
    test_sandbox_rejects_policy_probes()
    test_llm_pilot_programs_are_admitted()
    test_owa_tail_count_is_exact(); test_balance_gene_changes_assignment()
    test_ablation_full_is_the_reported_method()
    test_scheduling_robust_beats_nominal_on_tail()
    test_repair_rejects_coercion_and_records_empty_technicians()
    test_timeout_is_not_swallowed_by_generated_exception_handler()
    print("all smoke tests passed")
