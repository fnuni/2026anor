# Construct then repair the tail. A tight best-fit schedule is built first, then a
# bounded local search moves single work orders away from the technicians with the
# largest pessimistic load, as long as the move does not create a worse offender.
# The pessimistic load of a technician is its modal load inflated by the volatility
# it carries.
import math


def _pess(mach, proc, vol):
    acc = 0.0
    for j in mach:
        acc += proc[j] * (1.0 + vol[j])
    return acc


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    shift = instance["shift"]
    order = sorted(range(n), key=lambda j: -proc[j])
    machines = []
    loads = []
    for j in order:
        best = -1
        best_slack = shift + 1.0
        for k in range(len(machines)):
            slack = shift - (loads[k] + proc[j])
            if slack >= 0.0 and slack < best_slack:
                best = k
                best_slack = slack
        if best < 0:
            machines.append([j])
            loads.append(proc[j])
        else:
            machines[best].append(j)
            loads[best] += proc[j]
    for _ in range(40):
        pess = [_pess(m, proc, vol) for m in machines]
        src = 0
        for k in range(1, len(machines)):
            if pess[k] > pess[src]:
                src = k
        if pess[src] <= shift or len(machines[src]) < 2:
            break
        moved = False
        cand = sorted(machines[src], key=lambda j: proc[j] * (1.0 + vol[j]))
        for j in cand:
            size = proc[j] * (1.0 + vol[j])
            for k in range(len(machines)):
                if k == src:
                    continue
                if pess[k] + size <= shift:
                    machines[src].remove(j)
                    machines[k].append(j)
                    moved = True
                    break
            if moved:
                break
        if not moved:
            j = cand[0]
            machines[src].remove(j)
            machines.append([j])
        machines = [m for m in machines if m]
    return machines
