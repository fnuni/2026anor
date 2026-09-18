# Zone-block construction: work orders of one congestion zone are kept together so
# that a technician's route stays inside a single zone, which is operationally
# convenient. Within a zone the orders are packed by decreasing duration.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    zone = instance["zone"]
    shift = instance["shift"]
    by_zone = {}
    for j in range(n):
        g = zone[j]
        if g not in by_zone:
            by_zone[g] = []
        by_zone[g].append(j)
    machines = []
    for g in sorted(by_zone):
        tasks = sorted(by_zone[g], key=lambda j: -proc[j])
        cur = []
        load = 0.0
        for j in tasks:
            if cur and load + proc[j] > shift:
                machines.append(cur)
                cur = []
                load = 0.0
            cur.append(j)
            load += proc[j]
        if cur:
            machines.append(cur)
    return machines
