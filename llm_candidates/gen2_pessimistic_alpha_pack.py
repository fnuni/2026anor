# Pack against a pessimistic alpha-cut. Rather than inventing a reserve parameter,
# the heuristic packs against the upper endpoint of the alpha-cut interval at the
# lowest alpha level present in the grid, i.e. against the most pessimistic
# duration the fuzzy number admits. Feasibility is then evaluated on the modal
# durations, so the schedule is by construction shift-feasible in the worst cut.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    shift = instance["shift"]
    grid = params.get("alpha_grid", [0.0, 0.5, 1.0])
    alpha = min(grid) if grid else 0.0
    worst = []
    for j in range(n):
        worst.append(proc[j] * (1.0 + vol[j] * (1.0 - alpha)))
    order = sorted(range(n), key=lambda j: -worst[j])
    machines = []
    loads = []
    for j in order:
        placed = False
        for k in range(len(machines)):
            if loads[k] + worst[j] <= shift:
                machines[k].append(j)
                loads[k] += worst[j]
                placed = True
                break
        if not placed:
            machines.append([j])
            loads.append(worst[j])
    return machines
