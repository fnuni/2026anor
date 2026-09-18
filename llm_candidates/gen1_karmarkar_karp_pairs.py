# Differencing (Karmarkar-Karp style) on a fixed number of technicians. The two
# largest residual loads are repeatedly balanced, which yields very even workloads
# without ever consulting the shift boundary.
import math


def solve(instance, params):
    n = instance["n"]
    proc = instance["proc"]
    shift = instance["shift"]
    total = 0.0
    for j in range(n):
        total += proc[j]
    m = max(1, int(math.ceil(total / shift)))
    groups = []
    for j in sorted(range(n), key=lambda j: -proc[j]):
        groups.append([proc[j], [j]])
    while len(groups) > m:
        groups.sort(key=lambda g: -g[0])
        a = groups.pop(0)
        b = groups.pop(0)
        merged = [a[0] + b[0], a[1] + b[1]]
        groups.append(merged)
        groups.sort(key=lambda g: -g[0])
        if len(groups) <= m:
            break
        c = groups.pop(0)
        d = groups.pop()
        groups.append([c[0] + d[0], c[1] + d[1]])
    return [g[1] for g in groups if g[1]]
