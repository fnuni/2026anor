# Volatility-aware reserve. Each work order is charged against the shift at an
# inflated size proc[j] * (1 + kappa * volatility[j]), so a technician carrying
# volatile work is left with real slack while a technician carrying calm work is
# still packed tight. The reserve is proportional to the volatility actually
# carried, as the evaluator diagnosis asked.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    shift = instance["shift"]
    kappa = 1.0
    charged = []
    for j in range(n):
        charged.append(proc[j] * (1.0 + kappa * vol[j]))
    order = sorted(range(n), key=lambda j: -charged[j])
    machines = []
    loads = []
    for j in order:
        best = -1
        best_slack = shift + 1.0
        for k in range(len(machines)):
            slack = shift - (loads[k] + charged[j])
            if slack >= 0.0 and slack < best_slack:
                best = k
                best_slack = slack
        if best < 0:
            machines.append([j])
            loads.append(charged[j])
        else:
            machines[best].append(j)
            loads[best] += charged[j]
    return machines
