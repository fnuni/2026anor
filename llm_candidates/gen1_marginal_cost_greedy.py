# Marginal-cost greedy. Instead of a capacity rule, each placement is scored by the
# exact increase of the modal objective it causes, including the overtime term and
# the cost of opening a new technician. This is the myopically optimal placement
# under the nominal cost function.
import math


def _machine_cost(load, shift, rate, tech_cost):
    over = load - shift
    if over < 0.0:
        over = 0.0
    return tech_cost + load + rate * over


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    shift = instance["shift"]
    rate = instance["overtime_rate"]
    tech = instance["technician_cost"]
    order = sorted(range(n), key=lambda j: -proc[j])
    machines = []
    loads = []
    for j in order:
        best = -1
        best_delta = _machine_cost(proc[j], shift, rate, tech)
        for k in range(len(machines)):
            cur = _machine_cost(loads[k], shift, rate, tech)
            new = _machine_cost(loads[k] + proc[j], shift, rate, tech)
            if new - cur < best_delta:
                best_delta = new - cur
                best = k
        if best < 0:
            machines.append([j])
            loads.append(proc[j])
        else:
            machines[best].append(j)
            loads[best] += proc[j]
    return machines
