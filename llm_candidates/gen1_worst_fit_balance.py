# Worst-fit (least-loaded) assignment: every work order goes to the technician with
# the smallest current load, which is the greedy list-scheduling rule for makespan.
# It balances workload but does not exploit the shift boundary explicitly.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    shift = instance["shift"]
    order = sorted(range(n), key=lambda j: -proc[j])
    total = 0.0
    for j in range(n):
        total += proc[j]
    m = max(1, int(math.ceil(total / shift)))
    machines = []
    loads = []
    for _ in range(m):
        machines.append([])
        loads.append(0.0)
    for j in order:
        k = 0
        for i in range(1, m):
            if loads[i] < loads[k]:
                k = i
        machines[k].append(j)
        loads[k] += proc[j]
    return [mach for mach in machines if mach]
