# Best-fit decreasing: place each work order on the technician that will be left
# with the least remaining slack. Packs tighter than first-fit and therefore opens
# fewer technicians on the modal instance.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
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
    return machines
