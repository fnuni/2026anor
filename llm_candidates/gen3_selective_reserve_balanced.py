# Selective reserve, balanced setting. The reserve is applied only to the work orders
# whose volatility is in the upper part of the instance's own volatility range, and
# calm work is consolidated afterwards so the technician count is recovered. The
# aggressiveness parameter tau places this member at the balanced knee of the
# cost-risk frontier.
import math


def _consolidate(machines, proc, vol, shift, kappa):
    def charged(m):
        acc = 0.0
        for j in m:
            acc += proc[j] * (1.0 + kappa * vol[j])
        return acc
    merged = True
    while merged:
        merged = False
        machines = sorted(machines, key=charged)
        for a in range(len(machines)):
            for b in range(len(machines) - 1, a, -1):
                if charged(machines[a]) + charged(machines[b]) <= shift:
                    machines[a] = machines[a] + machines[b]
                    machines.pop(b)
                    merged = True
                    break
            if merged:
                break
    return machines


def _solve_with(instance, tau):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    shift = instance["shift"]
    lo = min(vol)
    hi = max(vol)
    thr = lo + tau * (hi - lo)
    charged = []
    for j in range(n):
        k = 1.0 if vol[j] >= thr else 0.0
        charged.append(proc[j] * (1.0 + k * vol[j]))
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
    return _consolidate(machines, proc, vol, shift, 0.35)


def solve(instance, params):
    return _solve_with(instance, 0.45)
