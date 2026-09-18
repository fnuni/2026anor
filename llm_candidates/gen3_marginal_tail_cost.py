# Marginal expected-overtime placement. Each placement is scored by the increase in
# expected overtime it causes under a coarse three-point fuzzy approximation of the
# technician load (optimistic, modal, pessimistic, weights 1/4, 1/2, 1/4), plus the
# cost of opening a technician. This puts the evaluator's own cost structure inside
# the construction rule instead of approximating it with a reserve.
import math


def _exp_cost(load, spread, shift, rate, tech):
    acc = 0.0
    pts = [(load * (1.0 - 0.5 * spread), 0.25),
           (load, 0.5),
           (load * (1.0 + spread), 0.25)]
    for value, w in pts:
        over = value - shift
        if over < 0.0:
            over = 0.0
        acc += w * (value + rate * over)
    return tech + acc


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    shift = instance["shift"]
    rate = instance["overtime_rate"]
    tech = instance["technician_cost"]
    order = sorted(range(n), key=lambda j: -proc[j])
    machines = []
    loads = []
    wvol = []
    for j in order:
        open_cost = _exp_cost(proc[j], vol[j], shift, rate, tech) - tech
        best = -1
        best_delta = open_cost + tech
        for k in range(len(machines)):
            new_load = loads[k] + proc[j]
            new_vol = (wvol[k] + proc[j] * vol[j]) / new_load
            cur = _exp_cost(loads[k], wvol[k] / loads[k], shift, rate, tech)
            new = _exp_cost(new_load, new_vol, shift, rate, tech)
            if new - cur < best_delta:
                best_delta = new - cur
                best = k
        if best < 0:
            machines.append([j])
            loads.append(proc[j])
            wvol.append(proc[j] * vol[j])
        else:
            machines[best].append(j)
            loads[best] += proc[j]
            wvol[best] += proc[j] * vol[j]
    return machines
