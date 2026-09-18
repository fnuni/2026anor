"""Scripted prototype rehearsal (feasibility dry-run) of the planner study.

Every record is produced by this fixed-seed script from the per-profile
assumptions stated below. No human participant is involved: the records are
not observations of people or expert opinions, and they are released together
with the seed and the assumptions so that they cannot be mistaken for
human-subject evidence. Their use is to illustrate assumption-dependent thresholds and candidate
interface variants. The script does not execute an interface, select programs,
measure elapsed decision time, or administer a SUS questionnaire.
"""
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "planner_rehearsal_records.csv"
SUMMARY = ROOT / "data" / "planner_rehearsal_summary.json"
SEED = 20260915
RNG = random.Random(SEED)

# Profile weights and initial dial settings are those of the archive-based
# regret analysis (creoh_planner.PROFILES and PlannerRegret.theta_map), so the
# rehearsal and the regret table use the same stated preferences.
profiles = [
    ("cost-driven", 0.80, 0.15, 0.05, 0.30),
    ("balanced", 0.40, 0.40, 0.20, 0.50),
    ("service-level", 0.20, 0.65, 0.15, 0.70),
    ("overtime-averse", 0.25, 0.50, 0.25, 0.70),
    ("stability-driven", 0.25, 0.25, 0.50, 0.60),
    ("worst-case", 0.05, 0.85, 0.10, 0.90),
]

# Scenario assumptions, not fitted human effects. Cost-oriented users are
# assumed to gain little from the portfolio, whereas tail-oriented users can
# gain more. This tests whether the interface should be offered selectively.
base_regret = {
    "cost-driven": 0.080, "balanced": 0.250, "service-level": 0.330,
    "overtime-averse": 0.320, "stability-driven": 0.300,
    "worst-case": 0.380,
}
portfolio_regret_factor = {
    "cost-driven": 0.92, "balanced": 0.69, "service-level": 0.58,
    "overtime-averse": 0.63, "stability-driven": 0.68,
    "worst-case": 0.55,
}

rows = []
for role in range(1, 17):
    profile, wc, wt, ws, recommended = profiles[(role - 1) % 6]
    experience = round(RNG.uniform(2.2, 15.7), 1)
    role_shift = RNG.gauss(0, 0.05)
    order = "AB" if role % 2 else "BA"
    sus = max(35, min(95, round(RNG.gauss(73, 11))))
    for episode in range(1, 5):
        common = RNG.gauss(0, 0.04)
        # A and B use the same latent episode difficulty and preference.
        nominal_regret = max(0.005, base_regret[profile] + role_shift + common)
        for condition in ("A_nominal", "B_portfolio"):
            is_b = condition == "B_portfolio"
            regret = max(0.005, nominal_regret * portfolio_regret_factor[profile] + RNG.gauss(0, 0.025)) if is_b else nominal_regret
            time_s = max(20, round(72 + 8 * episode + RNG.gauss(0, 12) + (26 + 8 * wt + RNG.gauss(0, 7) if is_b else 0)))
            confidence = max(1, min(7, round(RNG.gauss(5.1 if is_b else 4.6, 1.0))))
            tail_burden = int(RNG.random() < min(0.93, (0.48 + 0.45 * wt if is_b else 0.15 + 0.45 * wt)))
            dial_moves = RNG.randint(1, 4) if is_b else 0
            selected_theta = round(max(0.05, min(0.95, recommended + RNG.gauss(0, 0.08))), 2) if is_b else ""
            rows.append({
                "data_status": "SCRIPTED_REHEARSAL_NOT_HUMAN_DATA",
                "role_id": f"ROLE{role:02d}", "profile": profile,
                "experience_years_assumed": experience, "condition_order": order,
                "episode": episode, "condition": condition,
                "w_cost_assumed": wc, "w_tail_assumed": wt, "w_stability_assumed": ws,
                "regret_normalized_scripted": round(regret, 4),
                "decision_time_s_scripted": time_s,
                "confidence_1_7_scripted": confidence,
                "cites_tail_or_workforce_scripted": tail_burden,
                "dial_moves_scripted": dial_moves, "selected_theta_scripted": selected_theta,
                "sus_0_100_scripted": sus if is_b else "",
            })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def role_means(key, condition):
    return [mean(float(r[key]) for r in rows if r["role_id"] == f"ROLE{i:02d}" and r["condition"] == condition) for i in range(1, 17)]

a_reg = role_means("regret_normalized_scripted", "A_nominal")
b_reg = role_means("regret_normalized_scripted", "B_portfolio")
a_time = role_means("decision_time_s_scripted", "A_nominal")
b_time = role_means("decision_time_s_scripted", "B_portfolio")

def group_summary(profiles_in_group):
    a = [r for r in rows if r["profile"] in profiles_in_group and r["condition"] == "A_nominal"]
    b = [r for r in rows if r["profile"] in profiles_in_group and r["condition"] == "B_portfolio"]
    avg = lambda group, key: mean(float(r[key]) for r in group)
    return {
        "n": len(a) // 4,
        "regret_A": round(avg(a, "regret_normalized_scripted"), 3),
        "regret_B": round(avg(b, "regret_normalized_scripted"), 3),
        "extra_time_s": round(avg(b, "decision_time_s_scripted") - avg(a, "decision_time_s_scripted"), 1),
        "risk_mention_rate_A": round(avg(a, "cites_tail_or_workforce_scripted"), 3),
        "risk_mention_rate_B": round(avg(b, "cites_tail_or_workforce_scripted"), 3),
    }

grouped = {
    "cost_driven": group_summary(["cost-driven"]),
    "balanced": group_summary(["balanced"]),
    "risk_sensitive": group_summary(["service-level", "overtime-averse", "stability-driven", "worst-case"]),
}
mean_regret_a, mean_regret_b = mean(a_reg), mean(b_reg)
mean_time_a, mean_time_b = mean(a_time), mean(b_time)
summary = {
    "status": "SCRIPTED_REHEARSAL_NOT_HUMAN_DATA",
    "seed": SEED, "scripted_role_instances": 16, "episodes_per_condition": 4,
    "conditions": {"A": "single nominal recommendation", "B": "portfolio and risk dial"},
    "role_level_mean_regret_A": round(mean(a_reg), 4),
    "role_level_mean_regret_B": round(mean(b_reg), 4),
    "role_level_mean_paired_regret_difference_A_minus_B": round(mean(a - b for a, b in zip(a_reg, b_reg)), 4),
    "role_level_mean_time_s_A": round(mean(a_time), 1),
    "role_level_mean_time_s_B": round(mean(b_time), 1),
    "mean_time_ratio_B_to_A": round(mean(b_time) / mean(a_time), 3),
    "tail_or_workforce_citation_rate_A": round(mean(r["cites_tail_or_workforce_scripted"] for r in rows if r["condition"] == "A_nominal"), 3),
    "tail_or_workforce_citation_rate_B": round(mean(r["cites_tail_or_workforce_scripted"] for r in rows if r["condition"] == "B_portfolio"), 3),
    "median_SUS_B": median(int(r["sus_0_100_scripted"]) for r in rows if r["condition"] == "B_portfolio"),
    "overall_regret_reduction_percent": round(100 * (1 - mean_regret_b / mean_regret_a), 1),
    "overall_extra_time_s": round(mean_time_b - mean_time_a, 1),
    "time_headroom_to_1p5_ratio_s": round(1.5 * mean_time_a - mean_time_b, 1),
    "additional_B_time_that_breaches_1p5_s": math.floor(1.5 * mean_time_a - mean_time_b) + 1,
    "portfolio_regret_increase_to_erase_gain_percent": round(100 * (mean_regret_a / mean_regret_b - 1), 1),
    "risk_mention_gain_percentage_points": round(100 * (mean(r["cites_tail_or_workforce_scripted"] for r in rows if r["condition"] == "B_portfolio") - mean(r["cites_tail_or_workforce_scripted"] for r in rows if r["condition"] == "A_nominal")), 1),
    "grouped_tradeoff": grouped,
    "assumption_note": "Profile weights and initial dial settings are those of the archive-based regret analysis; regret is on an assumed per-profile scale that is not the archive regret of that analysis; behavioural effects, timing, confidence, SUS and variation are generator assumptions. No expert observations were available for calibration or validation.",
    "regret_baseline_assumptions_by_profile": base_regret,
    "portfolio_regret_factor_assumptions_by_profile": portfolio_regret_factor,
}
SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
macro_path = ROOT / "planner_rehearsal_macros.tex"
macro_names = {
    "RehRegretA": "role_level_mean_regret_A",
    "RehRegretB": "role_level_mean_regret_B",
    "RehRegretDiff": "role_level_mean_paired_regret_difference_A_minus_B",
    "RehTimeA": "role_level_mean_time_s_A",
    "RehTimeB": "role_level_mean_time_s_B",
    "RehTimeRatio": "mean_time_ratio_B_to_A",
    "RehCitationA": "tail_or_workforce_citation_rate_A",
    "RehCitationB": "tail_or_workforce_citation_rate_B",
    "RehSus": "median_SUS_B",
    "RehReductionPct": "overall_regret_reduction_percent",
    "RehExtraTime": "overall_extra_time_s",
    "RehTimeHeadroom": "time_headroom_to_1p5_ratio_s",
    "RehBreachExtra": "additional_B_time_that_breaches_1p5_s",
    "RehEraseGainPct": "portfolio_regret_increase_to_erase_gain_percent",
    "RehRiskGainPp": "risk_mention_gain_percentage_points",
}
macro_lines = ["% Generated by generate_planner_rehearsal.py (scripted rehearsal; no human participants).\n"]
macro_lines += [f"\\newcommand{{\\{name}}}{{{summary[key]}}}\n" for name, key in macro_names.items()]
for group_name, latex_prefix in [("cost_driven", "RehCost"), ("balanced", "RehBalanced"), ("risk_sensitive", "RehRisk")]:
    group = grouped[group_name]
    for suffix, key in [("N", "n"), ("RegA", "regret_A"), ("RegB", "regret_B"),
                        ("ExtraS", "extra_time_s"), ("MentionA", "risk_mention_rate_A"),
                        ("MentionB", "risk_mention_rate_B")]:
        macro_lines.append(f"\\newcommand{{\\{latex_prefix}{suffix}}}{{{group[key]}}}\n")
macro_path.write_text("".join(macro_lines))
print(json.dumps(summary, indent=2))
