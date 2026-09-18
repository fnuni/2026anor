# Zone diversification. The diagnosis says the tail is driven by technicians whose
# load is dominated by one volatile zone, because a common-mode shock moves all of
# that zone at once. This heuristic therefore does the opposite of gen1_zone_blocks:
# it spreads each zone across technicians, so that no technician's load is
# concentrated in a single shock.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    zone = instance["zone"]
    shift = instance["shift"]
    total = 0.0
    for j in range(n):
        total += proc[j] * (1.0 + 0.5 * vol[j])
    m = max(1, int(math.ceil(total / shift)))
    machines = []
    loads = []
    zone_load = []
    for _ in range(m):
        machines.append([])
        loads.append(0.0)
        zone_load.append({})
    for j in sorted(range(n), key=lambda j: -proc[j]):
        g = zone[j]
        best = 0
        best_score = None
        for k in range(m):
            concentration = zone_load[k].get(g, 0.0)
            score = loads[k] + 2.0 * vol[j] * concentration
            if best_score is None or score < best_score:
                best = k
                best_score = score
        machines[best].append(j)
        loads[best] += proc[j]
        zone_load[best][g] = zone_load[best].get(g, 0.0) + proc[j]
    return [mach for mach in machines if mach]
