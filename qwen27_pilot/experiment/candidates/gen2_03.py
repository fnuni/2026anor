def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']

    # Estimate effective duration per task using modal duration and volatility.
    # Use a conservative estimate: proc[j] * (1 + volatility[j])
    eff_dur = [proc[j] * (1.0 + volatility[j]) for j in range(n)]

    # Group tasks by zone to manage volatility concentration
    zones = {}
    for j in range(n):
        z = zone[j]
        if z not in zones:
            zones[z] = []
        zones[z].append(j)

    # Sort tasks by effective duration descending (largest first)
    task_order = sorted(range(n), key=lambda j: eff_dur[j], reverse=True)

    # Decide threshold for opening a new technician
    # Marginal cost of adding a task to a full machine: overtime_rate * (load - shift)
    # vs opening a new technician: technician_cost
    # We open a new machine if the current load exceeds shift and overtime penalty exceeds fixed cost
    # Heuristic: use a threshold based on technician_cost / overtime_rate as additional capacity
    threshold = shift + technician_cost / max(overtime_rate, 1e-9)

    # Initialize empty list of machines (each machine is a list of task indices)
    machines = []
    machine_loads = []  # current total effective load per machine
    machine_zone_vol = {}  # per machine, track sum of volatility to avoid concentration

    # Process tasks in order
    for j in task_order:
        z = zone[j]
        v = volatility[j]
        d = eff_dur[j]

        # Find best machine: prefer one that can fit within threshold
        # Among machines that can fit, choose the one with smallest load (LPT-like)
        # Also penalize machines that already have high volatility from the same zone
        best_idx = -1
        best_score = float('inf')

        for i, load in enumerate(machine_loads):
            new_load = load + d
            if new_load <= threshold:
                # Score: prefer lower load, and penalize if same zone already present
                # Check if this machine already has tasks from zone z
                zone_vol = machine_zone_vol.get(i, {}).get(z, 0.0)
                # Penalize concentration of volatile zone
                penalty = zone_vol * 2.0
                score = new_load + penalty
                if score < best_score:
                    best_score = score
                    best_idx = i

        if best_idx == -1:
            # No machine can fit within threshold; open a new machine
            machines.append([j])
            machine_loads.append(d)
            machine_zone_vol[len(machines) - 1] = {z: v}
        else:
            machines[best_idx].append(j)
            machine_loads[best_idx] += d
            if best_idx not in machine_zone_vol:
                machine_zone_vol[best_idx] = {}
            if z not in machine_zone_vol[best_idx]:
                machine_zone_vol[best_idx][z] = 0.0
            machine_zone_vol[best_idx][z] += v

    # Second pass: try to balance loads by moving tasks from overloaded machines
    # to underloaded ones, but only if it reduces total cost
    # Simple improvement: for each machine over threshold, try to move smallest task
    # to a machine that can accept it without exceeding threshold
    improved = True
    max_iter = 10
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        # Sort machines by load descending
        for i in range(len(machines)):
            if machine_loads[i] <= threshold:
                continue
            # Try to move the smallest task in this machine to another machine
            # Find smallest task in machine i
            min_task_idx = 0
            min_task_dur = eff_dur[machines[i][0]]
            for k in range(1, len(machines[i])):
                d_k = eff_dur[machines[i][k]]
                if d_k < min_task_dur:
                    min_task_dur = d_k
                    min_task_idx = k

            task_j = machines[i][min_task_idx]
            z_j = zone[task_j]
            v_j = volatility[task_j]

            # Find best target machine
            target_idx = -1
            target_score = float('inf')
            for k in range(len(machines)):
                if k == i:
                    continue
                new_load = machine_loads[k] + min_task_dur
                if new_load <= threshold:
                    zone_vol_k = machine_zone_vol.get(k, {}).get(z_j, 0.0)
                    penalty = zone_vol_k * 2.0
                    score = new_load + penalty
                    if score < target_score:
                        target_score = score
                        target_idx = k

            if target_idx != -1:
                # Move task
                machines[i].pop(min_task_idx)
                machines[target_idx].append(task_j)
                machine_loads[i] -= min_task_dur
                machine_loads[target_idx] += min_task_dur
                # Update zone vol
                if z_j in machine_zone_vol.get(i, {}):
                    machine_zone_vol[i][z_j] -= v_j
                    if machine_zone_vol[i][z_j] <= 0:
                        del machine_zone_vol[i][z_j]
                if target_idx not in machine_zone_vol:
                    machine_zone_vol[target_idx] = {}
                if z_j not in machine_zone_vol[target_idx]:
                    machine_zone_vol[target_idx][z_j] = 0.0
                machine_zone_vol[target_idx][z_j] += v_j
                improved = True

    # Remove any empty machines (shouldn't happen but just in case)
    machines = [m for m in machines if len(m) > 0]

    return machines
