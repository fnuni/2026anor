# Zone anti-chain with consolidation. Technicians are first built so that no two
# work orders of the same volatile zone share a technician beyond a cap, which
# breaks the common-mode correlation that drives the tail; calm zones are then
# consolidated to recover the technician count.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    vol = instance["volatility"]
    zone = instance["zone"]
    shift = instance["shift"]
    mean_vol = sum(vol) / max(1, n)
    total = sum(proc)
    m = max(1, int(math.ceil(total / (0.85 * shift))))
    machines = []
    loads = []
    zl = []
    for _ in range(m):
        machines.append([])
        loads.append(0.0)
        zl.append({})
    cap = 0.45 * shift
    for j in sorted(range(n), key=lambda j: -proc[j]):
        g = zone[j]
        volatile = vol[j] >= mean_vol
        best = -1
        best_score = None
        for k in range(m):
            if volatile and zl[k].get(g, 0.0) + proc[j] > cap:
                continue
            score = loads[k]
            if best_score is None or score < best_score:
                best = k
                best_score = score
        if best < 0:
            best = 0
            for k in range(1, m):
                if loads[k] < loads[best]:
                    best = k
        machines[best].append(j)
        loads[best] += proc[j]
        zl[best][g] = zl[best].get(g, 0.0) + proc[j]
    machines = [mm for mm in machines if mm]
    changed = True
    while changed:
        changed = False
        machines.sort(key=lambda mm: sum(proc[j] for j in mm))
        for a in range(len(machines)):
            for b in range(len(machines) - 1, a, -1):
                la = sum(proc[j] * (1.0 + 0.4 * vol[j]) for j in machines[a])
                lb = sum(proc[j] * (1.0 + 0.4 * vol[j]) for j in machines[b])
                if la + lb <= shift:
                    machines[a] = machines[a] + machines[b]
                    machines.pop(b)
                    changed = True
                    break
            if changed:
                break
    return machines
