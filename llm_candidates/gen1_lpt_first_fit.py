# Longest-processing-time first-fit decreasing.
# The classical parallel-machine construction: sort the work orders by decreasing
# modal duration and place each one on the first technician that can still take it
# within the regular shift. A new technician is opened only when none can.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    shift = instance["shift"]
    order = sorted(range(n), key=lambda j: -proc[j])
    machines = []
    loads = []
    for j in order:
        placed = False
        for k in range(len(machines)):
            if loads[k] + proc[j] <= shift:
                machines[k].append(j)
                loads[k] += proc[j]
                placed = True
                break
        if not placed:
            machines.append([j])
            loads.append(proc[j])
    return machines
