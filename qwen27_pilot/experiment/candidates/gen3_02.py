def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    orness = params.get('orness', 0.5)
    train_spread = params.get('train_spread', 1.0)
    alpha_grid = params.get('alpha_grid', [])

    # Estimate effective load per task under a central scenario (alpha ~ 0.5)
    # Use a simple scaling: effective duration = proc * (1 + volatility * factor)
    # factor derived from orness and train_spread
    factor = (orness - 0.5) * 2.0 * train_spread  # range roughly [-1, 1] scaled
    # Ensure factor doesn't cause negative durations
    eff = [proc[j] * max(0.0, 1.0 + volatility[j] * factor) for j in range(n)]

    # Sort tasks by effective duration descending (LPT-like)
    order = sorted(range(n), key=lambda j: eff[j], reverse=True)

    machines = []
    loads = []

    # Heuristic cost to open a new machine vs. assign to existing
    # We use a greedy LPT with a threshold based on technician_cost and overtime

    def cost_of_load(L):
        ot = max(0.0, L - shift)
        return technician_cost + L + overtime_rate * ot

    # Precompute threshold: it's cheaper to open a new machine if
    # technician_cost < overtime_rate * max(0, L - shift) for some L
    # We'll just use LPT and then do a local improvement pass

    for j in order:
        # Find the machine with the smallest load that keeps total cost low
        # Try adding to each existing machine, or open new
        best_cost = None
        best_idx = -1
        for m in range(len(machines)):
            new_load = loads[m] + eff[j]
            c = cost_of_load(new_load) - cost_of_load(loads[m])
            if best_cost is None or c < best_cost:
                best_cost = c
                best_idx = m
        # Cost of opening a new machine
        new_cost = technician_cost + eff[j]
        if best_cost is None or new_cost < best_cost:
            machines.append([j])
            loads.append(eff[j])
        else:
            machines[best_idx].append(j)
            loads[best_idx] += eff[j]

    # Local improvement: try to move tasks from high-load machines to low-load
    # or to new machines to reduce overtime penalty
    # Simple pass: for each machine with load > shift, try moving smallest task
    # to a machine with load < shift or to a new machine if cheaper

    improved = True
    while improved:
        improved = False
        # Sort machines by load descending
        m_order = sorted(range(len(machines)), key=lambda m: loads[m], reverse=True)
        for m in m_order:
            if loads[m] <= shift:
                continue  # no overtime, skip
            # Try to move the smallest task in this machine to reduce overtime
            # Find smallest task in machines[m]
            min_task_idx = -1
            min_task_load = float('inf')
            for j in machines[m]:
                if eff[j] < min_task_load:
                    min_task_load = eff[j]
                    min_task_idx = j
            if min_task_idx == -1:
                continue
            # Try moving to existing machine with smallest load
            best_target = -1
            best_delta = float('inf')
            for m2 in range(len(machines)):
                if m2 == m:
                    continue
                new_load_m = loads[m] - min_task_load
                new_load_m2 = loads[m2] + min_task_load
                delta = (cost_of_load(new_load_m) - cost_of_load(loads[m]) +
                         cost_of_load(new_load_m2) - cost_of_load(loads[m2]))
                if delta < best_delta:
                    best_delta = delta
                    best_target = m2
            # Also consider opening a new machine
            new_machine_cost = technician_cost + min_task_load
            old_machine_new_load = loads[m] - min_task_load
            delta_new = (cost_of_load(old_machine_new_load) - cost_of_load(loads[m]) +
                         new_machine_cost)
            if delta_new < best_delta:
                best_delta = delta_new
                best_target = -2  # signal new machine

            if best_delta < -1e-9:
                # Perform the move
                machines[m].remove(min_task_idx)
                if best_target == -2:
                    machines.append([min_task_idx])
                    loads.append(min_task_load)
                else:
                    machines[best_target].append(min_task_idx)
                    loads[best_target] += min_task_load
                loads[m] -= min_task_load
                improved = True

    # Remove empty machines (shouldn't happen but just in case)
    machines = [m for m in machines if m]

    return machines
