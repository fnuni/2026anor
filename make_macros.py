"""Emit ``results_macros.tex`` from the computed artefacts in ``data/``.

Every numeric value that appears in the manuscript is defined by exactly one
LaTeX macro generated here from a JSON/CSV artefact written by the pipelines.
No number is typed by hand into the manuscript, so a re-run of ``run_all.py``
followed by ``make_macros.py`` regenerates the paper's numbers end to end and
any drift between text and data is impossible by construction.
"""
from __future__ import annotations

import csv
import json
import os
from abc import ABC, abstractmethod

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _load(name: str) -> dict:
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


def _pm(pair, fmt="{:.1f}") -> str:
    return f"${fmt.format(pair[0])}\\pm{fmt.format(pair[1])}$"


def _pct(x, fmt="{:.1f}") -> str:
    return f"${fmt.format(x)}\\%$"


def _p(value) -> str:
    """A p-value carrying its own relation symbol.

    Values below the reporting resolution are written ``<10^{-3}`` and the rest
    ``=0.043``, so that a macro can follow ``$p_{\\mathrm{Holm}}$`` in running
    text and stand alone under a ``p`` column header without producing ``=<``.
    """
    if value is None:
        return "--"
    if value < 1e-3:
        return "${<}10^{-3}$"
    if value < 1e-2:
        return f"${{=}}{value:.4f}$"
    return f"${{=}}{value:.3f}$"


def _sci(value: float, digits: int = 1) -> str:
    """Scientific notation for small p-values, with its own relation symbol."""
    if value >= 1e-3:
        return f"${{=}}{value:.3f}$"
    m, e = f"{value:.{digits}e}".split("e")
    return f"${{=}}{m}\\times10^{{{int(e)}}}$"


_DIGIT_WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
               "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def _texname(name: str) -> str:
    """LaTeX control sequences may not contain digits: spell them out."""
    return "".join(_DIGIT_WORD.get(ch, ch) for ch in name)


class MacroBlock(ABC):
    """One thematic group of macros, emitted from one artefact."""

    title: str

    @abstractmethod
    def macros(self) -> dict[str, str]: ...

    @staticmethod
    def _mathsafe(value: str) -> str:
        """Wrap a numeric value so it typesets inside and outside math mode.

        A macro defined as ``$x$`` breaks when the manuscript writes
        ``$\\delta=\\Macro$``; ``\\ensuremath{x}`` is correct in both
        positions, which lets the text read naturally without the author having
        to remember which macros carry their own math delimiters.
        """
        if len(value) > 1 and value.startswith("$") and value.endswith("$"):
            return "\\ensuremath{" + value[1:-1] + "}"
        return value

    def emit(self) -> list[str]:
        lines = [f"% ---- {self.title} " + "-" * max(0, 60 - len(self.title))]
        for k, v in self.macros().items():
            lines.append(f"\\newcommand{{\\{_texname(k)}}}"
                         f"{{{self._mathsafe(str(v))}}}")
        return lines


class SchedulingBlock(MacroBlock):
    title = "Primary synthetic scheduling benchmark"

    def macros(self):
        h = _load("scheduling_metadata.json")
        rows = {}
        with open(os.path.join(DATA, "scheduling_main.csv")) as fh:
            for r in csv.DictReader(fh):
                rows[r["method"]] = r
        key = {"Classical heuristic": "SchedClassical",
               "Deterministic AHD": "SchedDet",
               "MO AHD (no fuzzy risk)": "SchedMo",
               "C-R-EoH": "SchedCreoh"}
        col = {"FOne": ("f1_mean", "f1_ci", "{:.1f}"),
               "FTwo": ("f2_mean", "f2_ci", "{:.1f}"),
               "FThree": ("f3_mean", "f3_ci", "{:.1f}"),
               "PNinetyFive": ("p95_mean", "p95_ci", "{:.1f}"),
               "Hv": ("hv_mean", "hv_ci", "{:.2f}"),
               "Ood": ("ood_mean", "ood_ci", "{:.1f}")}
        out = {}
        for meth, prefix in key.items():
            for suffix, (m, c, fmt) in col.items():
                out[prefix + suffix] = _pm([float(rows[meth][m]),
                                            float(rows[meth][c])], fmt)
        out.update(
            SchedTailReduction=_pct(h["p95_reduction_pct"]),
            SchedOwaReduction=_pct(h["owa_reduction_pct"]),
            SchedStabilityReduction=_pct(h["stability_reduction_pct"]),
            TailCountMForty=str(h["tail_count_M40"]),
            SchedNominalPremium=_pct(h["nominal_premium_pct"]),
            SchedExpectedChange=_pct(h["expected_cost_change_pct"]),
            SchedHvGain=_pct(h["hv_gain_pct"], "{:.0f}"),
            SchedDetOodMean=_pct(float(rows["Deterministic AHD"]["ood_mean"])),
            SchedCreohOodMean=_pct(float(rows["C-R-EoH"]["ood_mean"])),
            SchedSeeds=str(h["seeds"]), SchedTasks=str(h["n_tasks"]),
            SchedBudget=str(h["budget"]), SchedOrness=f"{h['orness']:.2f}",
            SchedSpread=f"{h['spread']:.2f}",
            FriedmanChi=f"${h['friedman_chi2']:.1f}$",
            FriedmanP=_sci(h["friedman_p"]))
        return out


class SchedulingStatsBlock(MacroBlock):
    title = "Scheduling statistics"

    def macros(self):
        rows = {}
        with open(os.path.join(DATA, "scheduling_stats.csv")) as fh:
            for r in csv.DictReader(fh):
                rows[(r["comparison"], r["metric"])] = r
        name = {("C-R-EoH vs deterministic AHD", "P95"): "StatTail",
                ("C-R-EoH vs deterministic AHD", "OOD degradation"): "StatOod",
                ("C-R-EoH vs MO AHD", "Hypervolume"): "StatHv",
                ("C-R-EoH vs deterministic AHD", "Mean cost"): "StatMean"}
        out = {}
        for k, prefix in name.items():
            r = rows[k]
            out[prefix + "Pvalue"] = _p(float(r["wilcoxon_p_holm"]))
            out[prefix + "Delta"] = f"${float(r['cliffs_delta']):+.2f}$"
        return out


class RoutingBlock(MacroBlock):
    title = "Synthetic routing contrast benchmark"

    def macros(self):
        m = _load("routing_metadata.json")
        rows = {}
        with open(os.path.join(DATA, "routing_contrast.csv")) as fh:
            for r in csv.DictReader(fh):
                rows[r["method"]] = r
        st = {}
        with open(os.path.join(DATA, "routing_stats.csv")) as fh:
            for r in csv.DictReader(fh):
                st[(r["comparison"], r["metric"])] = r
        det, cr = rows["Deterministic AHD"], rows["C-R-EoH"]
        tail = 100 * (float(det["p95_mean"]) - float(cr["p95_mean"])) \
            / float(det["p95_mean"])
        t = st[("C-R-EoH vs deterministic AHD", "P95")]
        return dict(
            RoutClassicalPNinetyFive=_pm([float(rows["Classical heuristic"]
                                                ["p95_mean"]),
                                          float(rows["Classical heuristic"]
                                                ["p95_ci"])]),
            RoutClassicalHv=_pm([float(rows["Classical heuristic"]["hv_mean"]),
                                 float(rows["Classical heuristic"]["hv_ci"])],
                                "{:.2f}"),
            RoutDetFOne=_pm([float(det["f1_mean"]), float(det["f1_ci"])]),
            RoutDetPNinetyFive=_pm([float(det["p95_mean"]), float(det["p95_ci"])]),
            RoutDetHv=_pm([float(det["hv_mean"]), float(det["hv_ci"])], "{:.2f}"),
            RoutCreohFOne=_pm([float(cr["f1_mean"]), float(cr["f1_ci"])]),
            RoutCreohPNinetyFive=_pm([float(cr["p95_mean"]), float(cr["p95_ci"])]),
            RoutCreohHv=_pm([float(cr["hv_mean"]), float(cr["hv_ci"])], "{:.2f}"),
            RoutTailReduction=_pct(tail),
            RoutTailPvalue=_p(float(t['wilcoxon_p_holm'])),
            RoutHvPvalue=_p(float(st[("C-R-EoH vs MO AHD", "Hypervolume")]
                                  ["wilcoxon_p_holm"])),
            RoutHvDelta=f"${float(st[('C-R-EoH vs MO AHD', 'Hypervolume')]['cliffs_delta']):+.2f}$",
            RoutTailDelta=f"${float(t['cliffs_delta']):+.2f}$",
            RoutFriedmanChi=f"${m['friedman_stat']:.1f}$")


class PublicBlock(MacroBlock):
    title = "Public-benchmark validation"

    def macros(self):
        pb = _load("public_benchmarks_metadata.json")
        out = {}
        prefix = {"cvrplib": "PubCvrp", "solomon_routing": "PubSolR",
                  "solomon_scheduling": "PubSolS"}
        for tag, pre in prefix.items():
            d = pb[tag]
            out[pre + "Instances"] = str(d["n_instances"])
            out[pre + "SourceFiles"] = str(d.get("source_file_count",
                                                   d["n_instances"]))
            out[pre + "TailReduction"] = _pct(d["p95_reduction_pct"], "{:.1f}")
            out[pre + "OwaReduction"] = _pct(d["owa_reduction_pct"], "{:.1f}")
            out[pre + "StabilityReduction"] = _pct(d["stability_reduction_pct"],
                                                   "{:.1f}")
            out[pre + "ExpectedChange"] = _pct(d["expected_cost_change_pct"],
                                               "{:.1f}")
            out[pre + "OodDet"] = _pct(d["ood_det"], "{:.1f}")
            out[pre + "OodCreoh"] = _pct(d["ood_creoh"], "{:.1f}")
            out[pre + "HvDet"] = f"${d['hv_det']:.2f}$"
            out[pre + "HvCreoh"] = f"${d['hv_creoh']:.2f}$"
            for meth, mp in (("Deterministic AHD", "Det"),
                             ("C-R-EoH", "Creoh"),
                             ("Classical heuristic", "Classical"),
                             ("MO AHD (no fuzzy risk)", "Mo")):
                s = d["summary"][meth]
                for k, mk, fmt in (("FOne", "f1", "{:.1f}"),
                                   ("FTwo", "f2", "{:.1f}"),
                                   ("FThree", "f3", "{:.1f}"),
                                   ("PNinetyFive", "p95", "{:.1f}"),
                                   ("Hv", "hv", "{:.2f}"),
                                   ("Ood", "ood", "{:.1f}")):
                    out[pre + mp + k] = _pm(s[mk], fmt)
            for r in d["stats"]:
                if r[1] == "P95":
                    out[pre + "TailPvalue"] = _p(r[3])
                    out[pre + "TailDelta"] = f"${r[4]:+.2f}$"
                if r[1] == "OOD degradation":
                    out[pre + "OodPvalue"] = _p(r[3])
                    out[pre + "OodDelta"] = f"${r[4]:+.2f}$"
                if r[1] == "Hypervolume":
                    out[pre + "HvPvalue"] = _p(r[3])
                    out[pre + "HvDelta"] = f"${r[4]:+.2f}$"
                if r[1] == "Expected cost":
                    out[pre + "MeanPvalue"] = _p(r[3])
                    out[pre + "MeanDelta"] = f"${r[4]:+.2f}$"
                if r[1] == "Scenario stability":
                    out[pre + "StabPvalue"] = _p(r[3])
                    out[pre + "StabDelta"] = f"${r[4]:+.2f}$"
        return out


class BaselineBlock(MacroBlock):
    title = "Strong OR baselines (held-out scenarios)"

    def macros(self):
        b = _load("baselines_metadata.json")
        short = {"Deterministic AHD (nominal)": "Nominal",
                 "Robust min-max (Wald)": "Wald",
                 "Min-max regret (Savage)": "Regret",
                 "Budgeted robust ($\\Gamma$=2 zones)": "Budgeted",
                 "SAA mean-CVaR ($\\beta$=0.90)": "Saa",
                 "Iterated racing (irace-style)": "Irace",
                 "Choice-function hyper-heuristic": "Hyper",
                 "C-R-EoH (fuzzy MO evaluator)": "Creoh"}
        out = {}
        for name, pre in short.items():
            s = b["summary"][name]
            for k, mk, fmt in (("FOne", "f1", "{:.1f}"), ("FTwo", "f2", "{:.1f}"),
                               ("FThree", "f3", "{:.1f}"),
                               ("PNinetyFive", "p95", "{:.1f}"),
                               ("Nominal", "nominal", "{:.1f}"),
                               ("Hv", "hv", "{:.2f}"), ("Ood", "ood", "{:.1f}")):
                out["Base" + pre + k] = _pm(s[mk], fmt)
        for r in b["stats"]:
            pre = short.get(r[0])
            if pre is None:
                continue
            mk = {"p95": "Tail", "f2": "Owa", "f3": "Stab",
                  "nominal": "Nom", "hv": "Hv"}.get(r[1])
            if mk:
                out[f"Base{pre}{mk}P"] = _p(r[3])
                out[f"Base{pre}{mk}D"] = f"${r[4]:+.2f}$"
        out["BaseFriedmanChi"] = f"${b['friedman_chi2']:.1f}$"
        nomp, crp = (b["summary"]["Deterministic AHD (nominal)"]["p95"][0],
                     b["summary"]["C-R-EoH (fuzzy MO evaluator)"]["p95"][0])
        out["BaseHeldOutTailReduction"] = _pct(100 * (nomp - crp) / nomp)
        out["BaseGamma"] = str(b["budget_gamma_zones"])
        out["BaseHeldOut"] = f"${100*b['held_out_fraction']:.0f}\\%$"
        return out


class AblationBlock(MacroBlock):
    title = "Ablation with confidence intervals"

    def macros(self):
        a = _load("ablation_metadata.json")
        short = {"Full C-R-EoH": "Full", "No OWA risk objective": "NoOwa",
                 "No stability objective": "NoStab",
                 "No robustness buffer": "NoBuffer",
                 "No Pareto archive (scalarised)": "NoArch",
                 "No alpha-cut ensemble (modal only)": "NoEns"}
        out = {"AblIdentical": a["full_identical_to_main_method"]}
        for name, pre in short.items():
            s = a["summary"][name]
            out[f"Abl{pre}FOne"] = _pm(s["f1"], "{:.1f}")
            out[f"Abl{pre}PNinetyFive"] = _pm(s["p95"], "{:.1f}")
            out[f"Abl{pre}Stability"] = _pm(s["f3"], "{:.1f}")
            out[f"Abl{pre}Hv"] = _pm(s["hv"], "{:.2f}")
            out[f"Abl{pre}Portfolio"] = f"${s['portfolio'][0]:.1f}$"
        for r in a["stats"]:
            pre = short.get(r[0])
            mk = {"p95": "Tail", "f2": "Owa", "f3": "Stab", "hv": "Hv"}.get(r[1])
            if pre and mk:
                out[f"Abl{pre}{mk}P"] = _p(r[3])
                out[f"Abl{pre}{mk}D"] = f"${r[4]:+.2f}$"
        return out


class MechanismBlock(MacroBlock):
    title = "Mechanism analysis"

    @staticmethod
    def _num(v, fmt):
        return "--" if v is None else f"${fmt.format(v)}$"

    def macros(self):
        m = _load("mechanism_metadata.json")
        short = {"Classical heuristic": "Classical",
                 "Deterministic AHD": "Det",
                 "MO AHD (no fuzzy risk)": "Mo", "C-R-EoH": "Creoh"}
        mant, ex = f"{m['work_identity_max_abs_diff']:.1e}".split("e")
        out = {"MechWorkIdentity": f"${mant}\\times10^{{{int(ex)}}}$"}
        for name, pre in short.items():
            s = m["summary"][name]
            out[f"Mech{pre}Techs"] = self._num(s['technicians'][0], "{:.2f}")
            out[f"Mech{pre}Work"] = self._num(s['work_mean'][0], "{:.1f}")
            out[f"Mech{pre}OtMean"] = self._num(s['overtime_mean'][0], "{:.1f}")
            out[f"Mech{pre}OtPNinetyFive"] = self._num(s['overtime_p95'][0], "{:.1f}")
            out[f"Mech{pre}Crossing"] = _pct(100 * s["crossing_frac"][0], "{:.0f}")
            out[f"Mech{pre}WorkloadGini"] = self._num(s['workload_gini'][0], "{:.3f}")
            out[f"Mech{pre}OvertimeGini"] = self._num(s['overtime_gini'][0], "{:.3f}")
            out[f"Mech{pre}MaxOt"] = self._num(s['max_tech_overtime'][0], "{:.1f}")
            out[f"Mech{pre}ArchSize"] = self._num(s['arch_size'][0], "{:.1f}")
            out[f"Mech{pre}Coll"] = self._num(s['arch_collinearity'][0], "{:.2f}")
            out[f"Mech{pre}SpreadTwo"] = self._num(s['arch_spread_f2'][0], "{:.2f}")
            out[f"Mech{pre}CollUnits"] = str(m["collinearity_defined_units"][name])
        for k, v in m["correlations"].items():
            nm = "".join(w.capitalize() for w in k.split("_"))
            out[f"Corr{nm}"] = f"${v[0]:+.2f}$"
            out[f"Corr{nm}P"] = _p(v[1])
        for r in m["stats"]:
            nm = "".join(w.capitalize() for w in r[0].split("_"))
            out[f"MechStat{nm}P"] = _p(r[2])
            out[f"MechStat{nm}D"] = f"${r[3]:+.2f}$"
        return out


class RuntimeBlock(MacroBlock):
    title = "Runtime and scalability"

    def macros(self):
        r = _load("runtime_metadata.json")
        out = dict(
            RunMachine=r["machine"]["processor"].replace("_", "\\_"),
            RunCpuModel=r["machine"].get("cpu_model", "").replace("_", "\\_")
            .replace("(R)", "").replace("(TM)", ""),
            RunPlatform="Linux" if "Linux" in r["machine"]["platform"]
            else r["machine"]["platform"].split("-")[0],
            RunMaxPredError=_pct(100 * r["max_relative_prediction_error"], "{:.0f}"),
            RunCostBenefitSeeds=str(r["cost_benefit"]["seeds"]),
            RunCpus=str(r["machine"]["logical_cpus"]),
            RunEnsemble=str(r["theoretical_overhead_factor"]),
            RunMedianOverhead=f"${r['measured_overhead_factor']:.1f}\\times$",
            RunExponentDet=f"${r['power_law']['deterministic']['exponent']:.2f}$",
            RunExponentFuzzy=f"${r['power_law']['fuzzy']['exponent']:.2f}$",
            RunExtraSeconds=f"${r['cost_benefit']['extra_seconds_per_run']:.2f}$",
            RunTailReduction=_pct(r["cost_benefit"]["p95_reduction_pct"]),
            RunSecondsPerPoint=f"${r['cost_benefit']['seconds_per_pct_tail_reduction']:.3f}$",
            RunDetSearch=f"${r['cost_benefit']['t_deterministic_search_s']:.2f}$",
            RunFuzzySearch=f"${r['cost_benefit']['t_fuzzy_search_s']:.2f}$")
        word = {20: "TwentyN", 40: "FortyN", 80: "EightyN", 160: "OneSixtyN",
                320: "ThreeTwentyN", 640: "SixFortyN"}
        for row in r["decomposition"]:
            w = word.get(row["n"])
            if w is None:
                continue
            out[f"Run{w}Struct"] = f"{row['t_struct_ms']:.3f}"
            out[f"Run{w}Scen"] = f"{row['t_scenario_ms']:.3f}"
            out[f"Run{w}Predicted"] = f"{row['predicted_factor']:.2f}"
            out[f"Run{w}Observed"] = f"{row['observed_factor']:.2f}"
            out[f"Run{w}Parallel"] = f"{row['parallel_bound_p8']:.2f}"
        first, last = r["decomposition"][0], r["decomposition"][-1]
        out.update(
            RunSmallN=str(first["n"]), RunLargeN=str(last["n"]),
            RunSmallFactor=f"${first['observed_factor']:.1f}\\times$",
            RunLargeFactor=f"${last['observed_factor']:.1f}\\times$",
            RunSmallPredicted=f"${first['predicted_factor']:.1f}\\times$",
            RunLargePredicted=f"${last['predicted_factor']:.1f}\\times$",
            RunLargeParallelBound=f"${last['parallel_bound_p8']:.2f}\\times$")
        return out


class PlannerBlock(MacroBlock):
    title = "Planner decision analysis"

    def macros(self):
        p = _load("planner_metadata.json")
        sw, st, rg, gd = (p["orness_sweep"], p["selection_stability"],
                          p["planner_regret"], p["parameter_guidance"])
        out = dict(
            PlanKnee=f"${sw['knee_theta']:.2f}$",
            PlanBufferRho=f"${sw['theta_buffer_spearman'][0]:+.2f}$",
            PlanBufferRhoP=_p(sw["theta_buffer_spearman"][1]),
            PlanDistinct=f"${st['mean_distinct_selections']:.2f}\\pm"
                         f"{st['ci_distinct']:.2f}$",
            PlanPlateau=f"${st['mean_plateau_width_theta']:.2f}$")
        rows = {r["theta"]: r for r in sw["rows"]}
        for th in (0.05, 0.50, 0.70, 0.95):
            key = f"PlanTheta{str(th).replace('.', '')}"
            r = rows.get(th)
            if r:
                out[key + "FOne"] = f"${r['f1']:.1f}$"
                out[key + "PNinetyFive"] = f"${r['p95']:.1f}$"
                out[key + "Stability"] = f"${r['f3']:.1f}$"
                out[key + "Ood"] = _pct(r["ood"], "{:.1f}")
        short = {"Cost-driven (budget-bound)": "Cost", "Balanced": "Balanced",
                 "Service-level (SLA-bound)": "Sla",
                 "Overtime-averse": "Overtime",
                 "Stability-driven": "Stability",
                 "Worst-case (regulatory)": "Worst"}
        stats = {r[0]: r for r in rg["stats"]}
        for lab, pre in short.items():
            d = rg["regret"][lab]
            out[f"Reg{pre}Dial"] = f"${d['regret_dial'][0]:.3f}$"
            out[f"Reg{pre}Det"] = f"${d['regret_det'][0]:.3f}$"
            out[f"Reg{pre}Random"] = f"${d['regret_random'][0]:.3f}$"
            out[f"Reg{pre}Theta"] = f"${rg['theta_map'][lab]:.2f}$"
            out[f"Reg{pre}P"] = _p(stats[lab][2])
            out[f"Reg{pre}D"] = f"${stats[lab][3]:+.2f}$"
        ratios = [rg["regret"][l]["regret_det"][0]
                  / max(1e-9, rg["regret"][l]["regret_dial"][0]) for l in short]
        out["RegMinRatio"] = f"${min(ratios):.0f}\\times$"
        out["RegMaxRatio"] = f"${max(ratios):.0f}\\times$"
        noncost = []
        for delta, rec in gd["recommended_theta"].items():
            tag = str(delta).replace("0.", "")
            for lab, pre in short.items():
                out[f"GuideDelta{tag}{pre}"] = f"${rec[lab]:.1f}$"
                if pre != "Cost":
                    noncost.append(rec[lab])
        out["GuideNonCostMin"] = f"${min(noncost):.1f}$"
        out["GuideNonCostMax"] = f"${max(noncost):.1f}$"
        steps = []
        grid = gd["theta_grid"]
        for lab in short:
            recs = [gd["recommended_theta"][str(dl)][lab] for dl in gd["spreads"]]
            idx = [grid.index(x) for x in recs]
            steps.append(max(idx) - min(idx))
        out["GuideMaxSteps"] = str(max(steps))
        det = [rg["regret"][l]["regret_det"][0] for l in short]
        out["RegDetMin"] = f"${min(det):.3f}$"
        out["RegDetMax"] = f"${max(det):.3f}$"
        return out


class LLMPilotBlock(MacroBlock):
    title = "Real-LLM proposer pilot"

    def macros(self):
        d = _load("llm_pilot_metadata.json")
        s = d["summary"]
        nom = s["LLM pool + nominal evaluator"]
        cr = s["LLM pool + fuzzy MO evaluator (C-R-EoH)"]
        out = dict(
            LlmBackbone=d["backbone"]["model"].replace("_", "\\_"),
            LlmProvider=d["backbone"]["provider"],
            LlmQueried=d["backbone"]["queried"],
            LlmInterface=d["backbone"]["interface"],
            LlmPrograms=str(d["programs_generated"]),
            LlmAdmitted=str(d["programs_admitted"]),
            LlmExecutions=str(d["executions"]),
            LlmFeasPre=_pct(100 * d["feasible_before_repair_rate"], "{:.1f}"),
            LlmFeasPost=_pct(100 * d["feasible_after_repair_rate"], "{:.1f}"),
            LlmRepair=_pct(100 * d["repair_rate"], "{:.1f}"),
            LlmRuntime=f"${d['mean_program_runtime_ms']:.2f}$",
            LlmPool=f"{d['mean_pool_size']:.1f}".rstrip("0").rstrip("."),
            LlmTailReduction=_pct(d["tail_reduction_pct"], "{:.1f}"))
        for k, mk, fmt in (("FOne", "f1", "{:.1f}"), ("FTwo", "f2", "{:.1f}"),
                           ("FThree", "f3", "{:.1f}"),
                           ("PNinetyFive", "p95", "{:.1f}"),
                           ("Hv", "hv", "{:.2f}"), ("Ood", "ood", "{:.1f}")):
            out["LlmNom" + k] = _pm(nom[mk], fmt)
            out["LlmCreoh" + k] = _pm(cr[mk], fmt)
        for r in d["stats"]:
            mk = {"p95": "Tail", "f2": "Owa", "f3": "Stab", "hv": "Hv",
                  "f1": "Mean"}.get(r[0])
            if mk:
                out[f"Llm{mk}P"] = _p(r[2])
                out[f"Llm{mk}D"] = f"${r[3]:+.2f}$"
        for g, v in d["per_generation_best"].items():
            tag = "Gen" + g.split()[-1]
            out[f"Llm{tag}PNinetyFive"] = f"${v['p95'][0]:.1f}$"
            out[f"Llm{tag}Owa"] = f"${v['f2'][0]:.1f}$"
            out[f"Llm{tag}Stability"] = f"${v['f3'][0]:.1f}$"
            out[f"Llm{tag}Ood"] = _pct(v["ood"][0], "{:.1f}")
        for g, v in d["by_generation"].items():
            out[f"LlmGen{g}Loc"] = f"${v['mean_loc']:.0f}$"
        return out


class TheoryBlock(MacroBlock):
    title = "Analytical properties and the opportunity index"

    def macros(self):
        t = _load("theory_metadata.json")
        out = {}
        for p in t["propositions"]:
            tag = p["proposition"].split()[0]
            out[f"Prop{tag}Trials"] = str(p["trials"])
            out[f"Prop{tag}Failures"] = str(p["failures"])
        out["PropTrials"] = str(t["propositions"][0]["trials"])
        out["PropChecks"] = str(len(t["propositions"]))
        short = {"Synthetic scheduling (primary)": "SchedSyn",
                 "Synthetic routing (contrast)": "RoutSyn",
                 "Solomon-derived scheduling (public)": "SchedPub",
                 "CVRPLIB X-set (public)": "CvrpPub",
                 "Solomon routing (public)": "RoutPub"}
        for f in t["families"]:
            pre = short.get(f["family"])
            if pre is None:
                continue
            out[f"Roi{pre}"] = _pct(100 * f["roi"][0], "{:.2f}")
            out[f"Roi{pre}Tail"] = _pct(100 * f["roi_p95"][0], "{:.2f}")
            out[f"Roi{pre}Realised"] = _pct(100 * f["realised_p95"][0], "{:.2f}")
            out[f"Roi{pre}Corr"] = f"${f['rank_corr'][0]:.2f}$"
            out[f"Roi{pre}Units"] = str(f["units"])
        a = t["association"]
        out["RoiInstRho"] = f"${a['instance_spearman']:+.2f}$"
        out["RoiInstP"] = _p(a["p_value"])
        out["RoiInstUnits"] = str(a["units"])
        out["RoiFamilies"] = str(a["n_families"])
        out["RoiBoundViolations"] = str(a["bound_violations"])
        brk = t["invariance_breaking"]
        get = {(b["exposure_spread"], b["overtime_rate"]): b["roi_mean"] for b in brk}
        out["BreakBaseline"] = _pct(100 * get[(0.0, 0.0)], "{:.2f}")
        out["BreakExposureOnly"] = _pct(100 * get[(1.0, 0.0)], "{:.2f}")
        out["BreakKinkOnly"] = _pct(100 * get[(0.0, 1.5)], "{:.2f}")
        out["BreakBoth"] = _pct(100 * get[(1.0, 1.5)], "{:.2f}")
        hi = max(brk, key=lambda b: b["roi_mean"])
        out["BreakMax"] = _pct(100 * hi["roi_mean"], "{:.2f}")
        out["BreakMaxSigma"] = f"${hi['exposure_spread']:.2f}$"
        out["BreakMaxRho"] = f"${hi['overtime_rate']:.1f}$"
        return out


class PlotBlock(MacroBlock):
    r"""Bare numbers for pgfplots coordinates.

    Figures must not carry hand-typed values either: these macros expand to a
    plain number with no math delimiters, so a plot coordinate reads
    ``(C-R-EoH,\PlotSchedCreohPNinetyFive)`` and cannot drift from the table it
    illustrates.
    """

    title = "Plot coordinates (bare numbers, no math delimiters)"

    def macros(self):
        out = {}
        rows = {}
        with open(os.path.join(DATA, "scheduling_main.csv")) as fh:
            for r in csv.DictReader(fh):
                rows[r["method"]] = r
        pre = {"Classical heuristic": "Classical", "Deterministic AHD": "Det",
               "MO AHD (no fuzzy risk)": "Mo", "C-R-EoH": "Creoh"}
        for meth, p in pre.items():
            out[f"PlotSched{p}PNinetyFive"] = f"{float(rows[meth]['p95_mean']):.2f}"
            out[f"PlotSched{p}Hv"] = f"{float(rows[meth]['hv_mean']):.3f}"
        st = _load("stress_metadata.json")
        for meth, p in pre.items():
            for lvl, idx in (("mild", "Mild"), ("medium", "Medium"),
                             ("severe", "Severe")):
                out[f"PlotOod{p}{idx}"] = f"{st['summary'][meth][lvl][0]:.1f}"
        return out


BLOCKS = [SchedulingBlock, SchedulingStatsBlock, RoutingBlock, PublicBlock,
          BaselineBlock, AblationBlock, MechanismBlock, RuntimeBlock,
          PlannerBlock, LLMPilotBlock, TheoryBlock, PlotBlock]


def main(out_path: str | None = None) -> str:
    out_path = out_path or os.path.join(HERE, "results_macros.tex")
    lines = ["% =====================================================",
             "% AUTO-GENERATED by make_macros.py -- do not edit.",
             "% Every number in the manuscript is defined here from the",
             "% artefacts in data/, which are produced by run_all.py.",
             "% ====================================================="]
    total = 0
    for block in BLOCKS:
        b = block()
        try:
            emitted = b.emit()
        except (FileNotFoundError, KeyError) as exc:
            lines.append(f"% SKIPPED {b.title}: {exc}")
            continue
        lines.extend(emitted)
        total += len(emitted) - 1
        lines.append("")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {total} macros to {out_path}")
    import make_ollama_macros
    make_ollama_macros.main()
    import make_qwen27_macros
    make_qwen27_macros.main()
    return out_path


if __name__ == "__main__":
    main()
