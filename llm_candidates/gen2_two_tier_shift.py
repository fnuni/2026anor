# Two-tier shift. Volatile work is packed against a reduced effective shift and
# calm work against the full shift; the two tiers are then merged whenever a calm
# technician still has room for a volatile remainder. The intent is to avoid the
# uniform packing-fraction reduction the diagnosis warned against.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    shift = instance["shift"]
    cut = 0.0
    for j in range(n):
        cut += vol[j]
    cut = cut / max(1, n)
    hot = [j for j in range(n) if vol[j] >= cut]
    cold = [j for j in range(n) if vol[j] < cut]

    def pack(tasks, cap):
        tasks = sorted(tasks, key=lambda j: -proc[j])
        out = []
        loads = []
        for j in tasks:
            placed = False
            for k in range(len(out)):
                if loads[k] + proc[j] <= cap:
                    out[k].append(j)
                    loads[k] += proc[j]
                    placed = True
                    break
            if not placed:
                out.append([j])
                loads.append(proc[j])
        return out, loads

    hot_m, hot_l = pack(hot, shift * 0.72)
    cold_m, cold_l = pack(cold, shift)
    for k in range(len(hot_m)):
        for i in range(len(cold_m)):
            if cold_l[i] + hot_l[k] <= shift * 0.92:
                cold_m[i] = cold_m[i] + hot_m[k]
                cold_l[i] += hot_l[k]
                hot_m[k] = []
                break
    return [m for m in (cold_m + hot_m) if m]
