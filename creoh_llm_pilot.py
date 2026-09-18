"""Real-LLM proposer pilot (revision).

The controlled experiments of the first submission replaced the language-model
proposer by a reproducible seeded search, to isolate the evaluator effect from
the backbone.  Reviewers asked, reasonably, whether the evaluator effect
survives when the candidate programs are actually written by a language model:
LLM output is noisier, sometimes infeasible, and structurally different from a
parametric mutation family.

This module runs that pilot end to end.  It is a *pilot*, not a backbone
comparison: one backbone, one problem family, one reflection loop, disclosed
budget.  What it establishes is (i) that the deterministic evaluator accepts,
rejects and repairs generated code as specified, (ii) how large the feasibility
and repair rates are for a current model, and (iii) whether selecting from the
LLM-written pool with the fuzzy multi-objective evaluator still beats selecting
from the *same* pool with a nominal scalar evaluator.

Protocol, fully disclosed
-------------------------
* **Backbone and interface.** Anthropic Claude Opus 5 (``claude-opus-5``),
  used in September 2026 through an interactive chat session, not through the
  API.  No fine-tuning.  Sampling parameters, tool availability and the
  session context were not under the authors' control and are not logged;
  no API request logs exist.
* **Prompt.** The prompt skeleton of the manuscript appendix, instantiated for the fuzzy field-service
  scheduling problem.  The two reflection blocks were *written by the authors*
  after inspecting the previous generation; they contain qualitative design
  guidance and were not generated automatically by the evaluator.  The
  generation-2 block states that two candidates required repair, which is not
  consistent with the execution log (no repair fired); the prompt is released
  unchanged and the discrepancy is disclosed in the manuscript.  Every prompt
  used is released verbatim in ``llm_candidates/prompts/``.
* **Generations.** Three: an initial generation from the bare interface, then
  two generations prompted with the author-written reflection blocks.
* **Output.** Each response is stored as one ``.py`` file in
  ``llm_candidates/``; the program text is not edited by the authors.
* **Evaluation.** The evaluator, repair policy, archive, selection rule and
  statistics are byte-identical to the ones used for the reproducible-search
  experiments.  Only the proposer changed.

Because provider-side model updates are outside the authors' control, the
pilot is reported as evidence about the *evaluator under noisy proposals*, and
the reproducible-search experiments remain the backbone-independent evidence.
"""
from __future__ import annotations

import ast
import csv
import glob
import json
import math
import os
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import wilcoxon

from creoh_routing import (ExpectedCost, OWARisk, ScenarioStability, FitnessVector,
                           ParetoArchive, p95, cliffs_delta, holm, dial_scores)
import creoh_scheduling as sched

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_DIR = os.path.join(HERE, "llm_candidates")

BACKBONE = dict(provider="Anthropic", model="claude-opus-5",
                queried="2026-09", interface="interactive chat session",
                sampling_parameters="not controlled", tools="not controlled",
                api_logs=False, reflection_blocks="written by the authors",
                fine_tuned=False)


# ==========================================================================
# Sandbox: static policy + bounded execution
# ==========================================================================
class SandboxViolation(Exception):
    """Raised when generated code breaks the static code-generation policy."""


class _Timeout(Exception):
    pass


class StaticPolicy:
    """AST-level enforcement of the prompt's 'allowed operations' clause.

    The policy is deliberately strict and is applied *before* execution: the
    generated program may not import anything beyond ``math``, may not touch
    attributes whose name begins with an underscore, and may not reference any
    name associated with file, network, process or introspection access.  A
    violation is a rejection, recorded as such, never a silent repair.
    """

    ALLOWED_IMPORTS = {"math"}
    BANNED_NAMES = {
        "open", "eval", "exec", "compile", "__import__", "input", "globals",
        "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
        "breakpoint", "memoryview", "object", "super", "classmethod",
        "staticmethod", "property", "type",
    }
    BANNED_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "pathlib",
                      "importlib", "builtins", "ctypes", "pickle", "requests",
                      "urllib", "threading", "multiprocessing", "signal"}

    def check(self, source: str) -> ast.Module:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise SandboxViolation(f"syntax error: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = ([a.name.split(".")[0] for a in node.names]
                        if isinstance(node, ast.Import)
                        else [(node.module or "").split(".")[0]])
                for m in mods:
                    if m in self.BANNED_MODULES or m not in self.ALLOWED_IMPORTS:
                        raise SandboxViolation(f"forbidden import: {m}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise SandboxViolation(f"forbidden attribute: {node.attr}")
            if isinstance(node, ast.Name) and node.id in self.BANNED_NAMES:
                raise SandboxViolation(f"forbidden name: {node.id}")
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise SandboxViolation("hidden state across evaluations")
        return tree


SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "len": len, "list": list, "map": map, "max": max, "min": min,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "sum": sum, "tuple": tuple, "zip": zip,
    "ValueError": ValueError, "IndexError": IndexError, "TypeError": TypeError,
    "KeyError": KeyError, "ZeroDivisionError": ZeroDivisionError,
    "Exception": Exception, "print": lambda *a, **k: None,
}


def _guarded_import(name, *_args, **_kwargs):
    """The only import path available to generated code.

    The static policy has already rejected every import outside
    ``StaticPolicy.ALLOWED_IMPORTS``; this guard makes the restriction hold at
    run time as well, so that a program cannot reach a module through a
    dynamically constructed import statement.
    """
    if name not in StaticPolicy.ALLOWED_IMPORTS:
        raise SandboxViolation(f"forbidden import at run time: {name}")
    return math


SAFE_BUILTINS["__import__"] = _guarded_import


class Sandbox:
    """Compiles and runs one generated program under policy and time limits."""

    def __init__(self, timeout_s: float = 2.0):
        self.timeout_s = timeout_s
        self.policy = StaticPolicy()

    def load(self, source: str, name: str):
        tree = self.policy.check(source)
        env = {"__builtins__": dict(SAFE_BUILTINS), "math": math}
        code = compile(tree, filename=f"<llm:{name}>", mode="exec")
        with self._limit():
            exec(code, env)                    # noqa: S102 - sandboxed by policy
        fn = env.get("solve")
        if not callable(fn):
            raise SandboxViolation("no callable solve(instance, params)")
        return fn

    def call(self, fn, instance: dict, params: dict):
        with self._limit():
            return fn(instance, params)

    def _limit(self):
        sandbox = self

        class _Ctx:
            def __enter__(self):
                def handler(signum, frame):
                    raise _Timeout("execution timeout")
                try:
                    self.prev = signal.signal(signal.SIGALRM, handler)
                    signal.setitimer(signal.ITIMER_REAL, sandbox.timeout_s)
                    self.armed = True
                except ValueError:             # not in the main thread
                    self.armed = False
                return self

            def __exit__(self, *exc):
                if getattr(self, "armed", False):
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, self.prev)
                return False

        return _Ctx()


# ==========================================================================
# Feasibility and the deterministic repair policy
# ==========================================================================
@dataclass
class EvaluationOutcome:
    status: str                    # ok | rejected | error | timeout | infeasible
    detail: str = ""
    repaired: bool = False
    machines: list | None = None


class RepairPolicy:
    """Documented, deterministic repair of a generated assignment.

    Exactly three defects are repairable, in this order, and each repair is
    logged so that pre- and post-repair feasibility can be reported separately:

    1. **duplicates** -- a task appearing on more than one technician is kept on
       the first and removed from the others;
    2. **omissions** -- a task appearing nowhere is appended to the technician
       with the smallest nominal load;
    3. **empty technicians** -- removed.

    Anything else (out-of-range indices, wrong output type, non-integer task
    identifiers) is *not* repaired: the candidate is rejected with a dominated
    penalty vector, as in the reproducible-search experiments.
    """

    def apply(self, machines, n: int) -> EvaluationOutcome:
        if not isinstance(machines, (list, tuple)):
            return EvaluationOutcome("infeasible", "output is not a list")
        clean, seen, repaired = [], set(), False
        for mach in machines:
            if not isinstance(mach, (list, tuple)):
                return EvaluationOutcome("infeasible", "technician is not a list")
            row = []
            for j in mach:
                try:
                    j = int(j)
                except (TypeError, ValueError):
                    return EvaluationOutcome("infeasible", "non-integer task id")
                if not (0 <= j < n):
                    return EvaluationOutcome("infeasible", f"task id {j} out of range")
                if j in seen:
                    repaired = True
                    continue
                seen.add(j); row.append(j)
            if row:
                clean.append(row)
            elif mach:
                repaired = True
        missing = [j for j in range(n) if j not in seen]
        if missing:
            repaired = True
            if not clean:
                clean.append([])
            for j in missing:
                k = int(np.argmin([len(m) for m in clean]))
                clean[k].append(j)
        if not clean:
            return EvaluationOutcome("infeasible", "empty assignment")
        return EvaluationOutcome("ok", "", repaired, clean)


# ==========================================================================
# Generated candidate program
# ==========================================================================
@dataclass
class GeneratedProgram:
    name: str
    generation: int
    source: str
    fn: object | None = None
    outcome: str = "untested"
    detail: str = ""

    @property
    def loc(self) -> int:
        return len([l for l in self.source.splitlines() if l.strip()
                    and not l.strip().startswith("#")])


class ProgramLoader:
    """Reads the released LLM responses from ``llm_candidates/``."""

    def load(self) -> list[GeneratedProgram]:
        progs = []
        for path in sorted(glob.glob(os.path.join(CANDIDATE_DIR, "gen*_*.py"))):
            base = os.path.basename(path)
            gen = int(base.split("_")[0].replace("gen", ""))
            progs.append(GeneratedProgram(base[:-3], gen, open(path).read()))
        return progs


# ==========================================================================
# Pilot experiment
# ==========================================================================
@dataclass
class LLMCandidate:
    """A generated program with the cost vectors the evaluator computed."""
    program: str
    generation: int
    train: np.ndarray
    ood: np.ndarray
    repaired: bool
    machines: list

    genome = sched.Genome()            # unused; kept for interface compatibility

    def fitness(self, f1, f2, f3) -> FitnessVector:
        return FitnessVector(f1(self.train), f2(self.train), f3(self.train))


class LLMProposerPilot:
    """Evaluates the released LLM programs with the unchanged evaluator."""

    def __init__(self, seeds: int = 30, n: int = 40, orness: float = 0.7,
                 spread: float = 0.15, timeout_s: float = 2.0):
        self.seeds, self.n, self.orness, self.spread = seeds, n, orness, spread
        self.sandbox = Sandbox(timeout_s)
        self.repair = RepairPolicy()
        self.f1, self.f2, self.f3 = ExpectedCost(), OWARisk(orness), ScenarioStability()

    # -- static admission -------------------------------------------------
    def admit(self, progs: list[GeneratedProgram]) -> list[GeneratedProgram]:
        for p in progs:
            try:
                p.fn = self.sandbox.load(p.source, p.name)
                p.outcome = "loaded"
            except SandboxViolation as exc:
                p.outcome, p.detail = "rejected", str(exc)
            except _Timeout:
                p.outcome, p.detail = "timeout", "import-time timeout"
            except Exception as exc:                        # noqa: BLE001
                p.outcome, p.detail = "error", f"{type(exc).__name__}: {exc}"
        return progs

    # -- one instance -----------------------------------------------------
    @staticmethod
    def _instance_view(inst: sched.FuzzyScheduleInstance) -> dict:
        """Read-only, plain-Python view handed to generated code."""
        return dict(n=int(inst.n),
                    proc=[float(x) for x in inst.proc],
                    volatility=[float(x) for x in inst.vol],
                    zone=[int(x) for x in inst.zone],
                    shift=float(inst.shift),
                    overtime_rate=float(inst.overtime_rate),
                    technician_cost=float(inst.tech_cost))

    def evaluate_on(self, prog: GeneratedProgram, inst, train, ood,
                    params: dict) -> tuple[EvaluationOutcome, float]:
        view = self._instance_view(inst)
        t0 = time.perf_counter()
        try:
            raw = self.sandbox.call(prog.fn, view, dict(params))
        except _Timeout:
            return EvaluationOutcome("timeout", "execution timeout"), 0.0
        except Exception as exc:                            # noqa: BLE001
            return EvaluationOutcome("error",
                                     f"{type(exc).__name__}: {exc}"), 0.0
        dt = time.perf_counter() - t0
        out = self.repair.apply(raw, inst.n)
        return out, dt

    @staticmethod
    def _cost_vector(inst, machines, scenarios) -> np.ndarray:
        vals = np.empty(len(scenarios))
        for s, pt in enumerate(scenarios):
            tot = 0.0
            for mach in machines:
                load = float(pt[mach].sum())
                tot += (inst.tech_cost + load
                        + inst.overtime_rate * max(0.0, load - inst.shift))
            vals[s] = tot
        return vals

    # -- driver -----------------------------------------------------------
    def run(self, outdir: str = "data") -> dict:
        progs = self.admit(ProgramLoader().load())
        loaded = [p for p in progs if p.outcome == "loaded"]
        gen = sched.ScheduleGenerator(n=self.n, spread=self.spread)
        eb, lb, sb = (sched.EndpointBuilder(), sched.LatinHypercubeBuilder(),
                      sched.StressBuilder())
        params = dict(orness=self.orness, alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
                      train_spread=self.spread)

        per = {k: {m: [] for m in ("f1", "f2", "f3", "p95", "hv", "ood")}
               for k in ("LLM pool + nominal evaluator",
                         "LLM pool + fuzzy MO evaluator (C-R-EoH)")}
        exec_rows, attempts = [], 0
        per_gen: dict[str, dict[str, list]] = {}
        feas_pre = feas_post = repaired_n = 0
        runtimes = []
        pool_sizes = []

        for seed in range(self.seeds):
            inst = gen.generate(seed)
            train = np.concatenate([eb.build(inst, seed), lb.build(inst, seed)])
            ood = sb.build(inst, seed)
            cands = []
            for p in loaded:
                attempts += 1
                out, dt = self.evaluate_on(p, inst, train, ood, params)
                runtimes.append(dt)
                if out.status == "ok":
                    feas_post += 1
                    if out.repaired:
                        repaired_n += 1
                    else:
                        feas_pre += 1
                    cands.append(LLMCandidate(
                        p.name, p.generation,
                        self._cost_vector(inst, out.machines, train),
                        self._cost_vector(inst, out.machines, ood),
                        out.repaired, out.machines))
                exec_rows.append(dict(seed=seed, program=p.name,
                                      generation=p.generation,
                                      status=out.status, detail=out.detail,
                                      repaired=out.repaired,
                                      runtime_ms=round(1e3 * dt, 3)))
            if len(cands) < 2:
                continue
            pool_sizes.append(len(cands))
            for g in sorted({c.generation for c in cands}):
                sub = [c for c in cands if c.generation == g]
                gk = f"generation {g}"
                dg = per_gen.setdefault(gk, {m: [] for m in
                                             ("f1", "f2", "f3", "p95", "ood")})
                best = min(sub, key=lambda c: self.f2(c.train))
                dg["f1"].append(self.f1(best.train))
                dg["f2"].append(self.f2(best.train))
                dg["f3"].append(self.f3(best.train))
                dg["p95"].append(p95(best.train))
                dg["ood"].append(100.0 * (best.ood.mean() - best.train.mean())
                                 / best.train.mean())
            a1 = np.array([self.f1(c.train) for c in cands])
            a2 = np.array([self.f2(c.train) for c in cands])
            lo1, hi1, lo2, hi2 = a1.min(), a1.max(), a2.min(), a2.max()

            def norm(fv):
                return FitnessVector((fv.f1 - lo1) / (hi1 - lo1 + 1e-9),
                                     (fv.f2 - lo2) / (hi2 - lo2 + 1e-9), fv.f3)

            # (a) nominal scalar evaluator on the SAME LLM pool
            nom = min(cands, key=lambda c: c.train[0])
            na = ParetoArchive(); na.add(norm(nom.fitness(self.f1, self.f2, self.f3)), nom)
            # (b) fuzzy multi-objective evaluator on the SAME LLM pool
            arch = ParetoArchive()
            for c in cands:
                arch.add(c.fitness(self.f1, self.f2, self.f3), c)
            fvs = [fv for fv, _ in arch.items]
            knee = arch.items[int(np.argmin(dial_scores(fvs, self.orness)))][1]
            ca = ParetoArchive()
            for _fv, c in arch.items:
                ca.add(norm(c.fitness(self.f1, self.f2, self.f3)), c)

            for label, sel, archive in (
                    ("LLM pool + nominal evaluator", nom, na),
                    ("LLM pool + fuzzy MO evaluator (C-R-EoH)", knee, ca)):
                d = per[label]
                d["f1"].append(self.f1(sel.train)); d["f2"].append(self.f2(sel.train))
                d["f3"].append(self.f3(sel.train)); d["p95"].append(p95(sel.train))
                d["hv"].append(archive.hypervolume((1.05, 1.05)))
                d["ood"].append(100.0 * (sel.ood.mean() - sel.train.mean())
                                / sel.train.mean())

        return self._report(progs, loaded, per, exec_rows, attempts, feas_pre,
                            feas_post, repaired_n, runtimes, pool_sizes, outdir,
                            per_gen)

    # -- reporting --------------------------------------------------------
    def _report(self, progs, loaded, per, exec_rows, attempts, feas_pre,
                feas_post, repaired_n, runtimes, pool_sizes, outdir,
                per_gen=None) -> dict:
        def ci(x):
            x = np.asarray(x, float)
            return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0

        nom = per["LLM pool + nominal evaluator"]
        cr = per["LLM pool + fuzzy MO evaluator (C-R-EoH)"]
        base = np.mean(nom["f1"]) if nom["f1"] else 1.0
        scale = 100.0 / base
        summary = {}
        for label, d in per.items():
            summary[label] = {k: (float(np.mean(np.array(v) * (
                                  1.0 if k in ("hv", "ood") else scale))),
                                  float(ci(np.array(v) * (
                                  1.0 if k in ("hv", "ood") else scale))))
                              for k, v in d.items() if v}
        raw_p, stats_rows = {}, []
        for k in ("p95", "f2", "f3", "hv", "f1"):
            if len(cr.get(k, [])) < 2:
                raw_p[k] = 1.0
                stats_rows.append([k, 1.0, 0.0])
                continue
            try:
                _, p = wilcoxon(cr[k], nom[k])
            except ValueError:
                p = 1.0
            raw_p[k] = float(p)
            stats_rows.append([k, float(p), cliffs_delta(cr[k], nom[k])])
        adj = holm(raw_p)
        for r in stats_rows:
            r.insert(2, adj[r[0]])

        status_counts: dict[str, int] = {}
        for r in exec_rows:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        by_gen: dict[int, dict] = {}
        for p in progs:
            g = by_gen.setdefault(p.generation, dict(generated=0, admitted=0,
                                                     loc=[]))
            g["generated"] += 1
            g["loc"].append(p.loc)
            if p.outcome == "loaded":
                g["admitted"] += 1
        for g in by_gen.values():
            g["mean_loc"] = round(float(np.mean(g["loc"])), 1)
            g.pop("loc")

        meta = dict(
            backbone=BACKBONE, seeds=self.seeds,
            programs_generated=len(progs),
            programs_admitted=len(loaded),
            static_rejections=[[p.name, p.detail] for p in progs
                               if p.outcome != "loaded"],
            by_generation=by_gen,
            executions=attempts,
            execution_status=status_counts,
            feasible_before_repair_rate=round(feas_pre / max(1, attempts), 4),
            feasible_after_repair_rate=round(feas_post / max(1, attempts), 4),
            repair_rate=round(repaired_n / max(1, attempts), 4),
            mean_program_runtime_ms=round(1e3 * float(np.mean(runtimes)), 3),
            mean_pool_size=round(float(np.mean(pool_sizes)), 2)
            if pool_sizes else 0,
            summary={k: {m: [round(v[0], 3), round(v[1], 3)]
                         for m, v in d.items()} for k, d in summary.items()},
            stats=[[r[0], r[1], r[2], round(r[3], 3)] for r in stats_rows],
            per_generation_best=({
                g: {m: [round(float(np.mean(np.array(v) * scale)), 3),
                        round(float(ci(np.array(v) * scale)), 3)]
                    for m, v in d.items() if m != "ood"}
                   | {"ood": [round(float(np.mean(d["ood"])), 3),
                              round(float(ci(d["ood"])), 3)]}
                for g, d in sorted((per_gen or {}).items())}),
            tail_reduction_pct=round(100 * (summary["LLM pool + nominal evaluator"]
                                            ["p95"][0]
                                            - summary["LLM pool + fuzzy MO "
                                                      "evaluator (C-R-EoH)"]
                                            ["p95"][0])
                                     / summary["LLM pool + nominal evaluator"]
                                     ["p95"][0], 2))

        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "llm_pilot_executions.csv"), "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(exec_rows[0].keys()))
            w.writeheader(); w.writerows(exec_rows)
        with open(os.path.join(outdir, "llm_pilot_main.csv"), "w",
                  newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["evaluator_on_llm_pool", "f1_mean", "f1_ci", "f2_mean",
                        "f2_ci", "f3_mean", "f3_ci", "p95_mean", "p95_ci",
                        "hv_mean", "hv_ci", "ood_mean", "ood_ci"])
            for k, d in summary.items():
                w.writerow([k] + [f"{v:.3f}" for m in
                                  ("f1", "f2", "f3", "p95", "hv", "ood")
                                  for v in d[m]])
        with open(os.path.join(outdir, "llm_pilot_metadata.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        print(f"# LLM proposer pilot -- backbone {BACKBONE['model']}, "
              f"{self.seeds} instances")
        print(f"programs generated {meta['programs_generated']}, "
              f"admitted by the static policy {meta['programs_admitted']}")
        print(f"execution status: {status_counts}")
        print(f"feasible before repair {100*meta['feasible_before_repair_rate']:.1f}%"
              f", after repair {100*meta['feasible_after_repair_rate']:.1f}%"
              f", repair rate {100*meta['repair_rate']:.1f}%")
        print(f"mean candidate pool per instance: {meta['mean_pool_size']}")
        print(f"\n{'evaluator on the same LLM pool':42s} {'f1':>8} {'f2':>8} "
              f"{'f3':>7} {'P95':>8} {'HV':>6} {'OOD%':>7}")
        for k, d in summary.items():
            print(f"{k:42s} {d['f1'][0]:8.2f} {d['f2'][0]:8.2f} {d['f3'][0]:7.2f} "
                  f"{d['p95'][0]:8.2f} {d['hv'][0]:6.3f} {d['ood'][0]:7.2f}")
        print("\nbest candidate per generation (fuzzy tail objective):")
        print(f"{'generation':14s} {'f1':>8} {'f2':>8} {'f3':>7} {'P95':>8} "
              f"{'OOD%':>7}")
        for g, d in meta["per_generation_best"].items():
            print(f"{g:14s} {d['f1'][0]:8.2f} {d['f2'][0]:8.2f} {d['f3'][0]:7.2f} "
                  f"{d['p95'][0]:8.2f} {d['ood'][0]:7.2f}")
        print(f"\ntail reduction on the LLM pool: {meta['tail_reduction_pct']}%")
        for r in stats_rows:
            print(f"  {r[0]:5s} p_holm={r[2]:.3e} delta={r[3]:+.3f}")
        return meta


def run(seeds: int = 30, outdir: str = "data") -> dict:
    return LLMProposerPilot(seeds=seeds).run(outdir=outdir)


if __name__ == "__main__":
    run(outdir="data")
