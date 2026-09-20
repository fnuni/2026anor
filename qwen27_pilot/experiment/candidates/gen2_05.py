def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']

    orness = params.get('orness', 0.5)
    alpha_grid = params.get('alpha_grid', [0.0, 0.5, 1.0])
    train_spread = params.get('train_spread', 0.1)

    # Fuzzy duration: use a weighted combination that accounts for volatility
    # Higher orness -> more pessimistic (upper tail); lower -> optimistic
    # We use a simple fuzzy mean: proc * (1 + volatility * (orness - 0.5))
    # This gives a single representative duration per task that reflects uncertainty
    fuzzy_durations = []
    for j in range(n):
        # Linear interpolation between lower and upper bound based on orness
        # Lower bound: proc * (1 - volatility), Upper: proc * (1 + volatility)
        # orness=0.5 -> nominal; orness>0.5 -> toward upper; orness<0.5 -> toward lower
        factor = 1.0 + volatility[j] * (orness - 0.5) * 2.0
        factor = max(0.01, factor)  # ensure positive
        fuzzy_durations.append(proc[j] * factor)

    # Sort tasks by fuzzy duration descending (LPT-like)
    tasks_sorted = sorted(range(n), key=lambda j: -fuzzy_durations[j])

    # Decide number of technicians: balance between opening cost and overtime
    # Estimate total fuzzy load
    total_load = sum(fuzzy_durations)
    
    # For a given number of machines m, average load per machine is total_load/m
    # Cost approximation: m * (technician_cost + avg_load + overtime_rate * max(0, avg_load - shift))
    # Find m that minimizes this approximate cost
    best_m = 1
    best_cost = float('inf')
    max_m = max(1, int(total_load / (shift * 0.8)) + 2)  # upper bound on reasonable machines
    for m in range(1, min(max_m, n + 1)):
        avg = total_load / m
        ot = max(0.0, avg - shift)
        cost = m * (technician_cost + avg + overtime_rate * ot)
        if cost < best_cost:
            best_cost = cost
            best_m = m

    # Ensure we don't open more machines than tasks
    best_m = max(1, min(best_m, n))

    # Initialize machines with their current load and zone concentration tracking
    machine_loads = [0.0] * best_m
    machine_zone_counts = [dict() for _ in range(best_m)]
    machine_volatility_load = [0.0] * best_m  # sum of volatility-weighted durations

    # Assign tasks using a modified LPT with zone-diversity and volatility-aware capacity
    machines = [[] for _ in range(best_m)]

    for j in tasks_sorted:
        dur = fuzzy_durations[j]
        vol = volatility[j]
        z = zone[j]

        # Score each machine: prefer lower load, penalize zone concentration,
        # and add a volatility-aware capacity reservation
        best_machine = -1
        best_score = float('inf')

        for m in range(best_m):
            current_load = machine_loads[m]
            current_vol_load = machine_volatility_load[m]

            # Projected load after assignment
            new_load = current_load + dur
            new_vol_load = current_vol_load + dur * vol

            # Overtime cost for this machine if we add this task
            ot = max(0.0, new_load - shift)
            ot_cost = overtime_rate * ot

            # Zone concentration penalty: if this zone is already heavy on this machine, penalize
            zone_count = machine_zone_counts[m].get(z, 0)
            zone_penalty = zone_count * vol * 0.5  # penalize volatile zone concentration

            # Volatility-aware capacity reservation: machines with high volatile load
            # get a soft penalty to spread risk
            vol_penalty = current_vol_load * 0.1

            # Total score
            score = new_load + ot_cost + zone_penalty + vol_penalty

            if score < best_score:
                best_score = score
                best_machine = m

        # Assign task to best machine
        machines[best_machine].append(j)
        machine_loads[best_machine] += dur
        machine_volatility_load[best_machine] += dur * vol
        machine_zone_counts[best_machine][z] = machine_zone_counts[best_machine].get(z, 0) + 1

    # Post-processing: if any machine is empty, redistribute
    # Since we assigned all tasks and best_m <= n, and we always assign to a machine,
    # no machine should be empty if best_m <= n. But just in case, merge empty ones.
    non_empty = [m for m in range(best_m) if len(machines[m]) > 0]
    if len(non_empty) < best_m:
        # Rebuild with only non-empty machines
        machines = [machines[m] for m in non_empty]

    return machines
