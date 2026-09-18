# Exposure budget. Each technician is given a budget on the total *volatile* work
# it may carry, expressed as a fraction of the shift, while its total nominal load
# may still fill the shift. Calm work therefore consolidates freely and the reserve
# is spent only where it buys tail reduction.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    shift = instance["shift"]
    budget = 0.30 * shift
    exposure = []
    for j in range(n):
        exposure.append(proc[j] * vol[j])
    order = sorted(range(n), key=lambda j: -(proc[j] + 2.0 * exposure[j]))
    machines = []
    loads = []
    expo = []
    for j in order:
        best = -1
        best_slack = None
        for k in range(len(machines)):
            if loads[k] + proc[j] > shift:
                continue
            if expo[k] + exposure[j] > budget:
                continue
            slack = shift - (loads[k] + proc[j])
            if best_slack is None or slack < best_slack:
                best = k
                best_slack = slack
        if best < 0:
            machines.append([j])
            loads.append(proc[j])
            expo.append(exposure[j])
        else:
            machines[best].append(j)
            loads[best] += proc[j]
            expo[best] += exposure[j]
    return machines
