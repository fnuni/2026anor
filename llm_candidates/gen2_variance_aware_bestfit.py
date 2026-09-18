# Variance-aware best fit. A technician's exposure is modelled as the standard
# deviation of its load under independent zone shocks, sigma_k = sqrt(sum over
# zones of (volatility * zone load)^2). A placement is admissible when the mean
# load plus z * sigma stays within the shift, which is a chance-constrained rule
# with a fixed safety factor.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    zone = instance["zone"]
    shift = instance["shift"]
    z = 1.28
    machines = []
    loads = []
    zone_exposure = []

    def sigma(idx, extra_zone=None, extra=0.0):
        acc = 0.0
        for g, v in zone_exposure[idx].items():
            w = v + (extra if g == extra_zone else 0.0)
            acc += w * w
        if extra_zone is not None and extra_zone not in zone_exposure[idx]:
            acc += extra * extra
        return math.sqrt(acc)

    for j in sorted(range(n), key=lambda j: -proc[j]):
        g = zone[j]
        contrib = proc[j] * vol[j]
        best = -1
        best_slack = None
        for k in range(len(machines)):
            mean = loads[k] + proc[j]
            s = sigma(k, g, contrib)
            slack = shift - (mean + z * s)
            if slack >= 0.0 and (best_slack is None or slack < best_slack):
                best = k
                best_slack = slack
        if best < 0:
            machines.append([j])
            loads.append(proc[j])
            zone_exposure.append({g: contrib})
        else:
            machines[best].append(j)
            loads[best] += proc[j]
            zone_exposure[best][g] = zone_exposure[best].get(g, 0.0) + contrib
    return machines
