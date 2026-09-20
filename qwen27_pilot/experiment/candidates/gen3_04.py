def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    orness = params.get('orness', 0.5)
    alpha_grid = params.get('alpha_grid', [0.1, 0.3, 0.5, 0.7, 0.9])
    train_spread = params.get('train_spread', 1.0)
    
    # Group tasks by zone
    zone_tasks = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_tasks:
            zone_tasks[z] = []
        zone_tasks[z].append(j)
    
    # For each zone, compute a risk-adjusted expected load
    # Use a conservative scaling based on volatility and orness
    # Higher orness -> more conservative (higher scaling)
    zone_load = {}
    for z, tasks in zone_tasks.items():
        base_load = sum(proc[j] for j in tasks)
        # Average volatility in zone
        avg_vol = sum(volatility[j] for j in tasks) / len(tasks)
        # Scale factor: blend of nominal and worst-case based on orness
        # worst-case approx: proc * (1 + volatility * train_spread)
        # Use orness to interpolate between nominal (orness=0) and conservative (orness=1)
        scale = 1.0 + orness * avg_vol * train_spread
        zone_load[z] = base_load * scale
    
    # Sort zones by their risk-adjusted load descending
    sorted_zones = sorted(zone_tasks.keys(), key=lambda z: zone_load[z], reverse=True)
    
    # First-fit decreasing by zone load, with capacity target
    # Target load per technician: shift + some buffer to avoid overtime
    # Buffer depends on orness: higher orness -> smaller buffer (more conservative)
    buffer = shift * (1.0 - orness) * 0.5  # 0.5 * shift * (1-orness) buffer
    target_load = shift + buffer
    
    machines = []
    machine_loads = []  # current risk-adjusted load per machine
    machine_tasks = []  # tasks per machine
    
    for z in sorted_zones:
        tasks = zone_tasks[z]
        z_load = zone_load[z]
        
        # Try to fit this zone's tasks into an existing machine
        # Find the machine with the most remaining capacity
        best_idx = -1
        best_remaining = -1.0
        
        for i, ml in enumerate(machine_loads):
            remaining = target_load - ml
            if remaining >= z_load * 0.5:  # Allow partial fit if at least half fits
                if remaining > best_remaining:
                    best_remaining = remaining
                    best_idx = i
        
        if best_idx >= 0:
            machine_loads[best_idx] += z_load
            machine_tasks[best_idx].extend(tasks)
        else:
            # Open new machine
            machine_loads.append(z_load)
            machine_tasks.append(list(tasks))
    
    # Now try to split large zones that exceed target_load significantly
    # to reduce overtime risk
    # For each machine, if load > target_load * 1.2, try to split
    new_machines = []
    new_loads = []
    
    for i, (ml, tasks) in enumerate(zip(machine_loads, machine_tasks)):
        if ml <= target_load * 1.2:
            new_machines.append(tasks)
            new_loads.append(ml)
        else:
            # Split this machine's tasks
            # Group tasks by their original zone within this machine
            local_zone_tasks = {}
            for j in tasks:
                z = zone[j]
                if z not in local_zone_tasks:
                    local_zone_tasks[z] = []
                local_zone_tasks[z].append(j)
            
            # Sort local zones by load descending
            local_sorted = sorted(local_zone_tasks.keys(), key=lambda z: zone_load[z], reverse=True)
            
            # Greedily pack into new sub-machines
            sub_machines = []
            sub_loads = []
            for z in local_sorted:
                zt = local_zone_tasks[z]
                zl = zone_load[z]
                # Try to fit into existing sub-machine
                placed = False
                for si, sl in enumerate(sub_loads):
                    if sl + zl <= target_load * 1.1:
                        sub_loads[si] += zl
                        sub_machines[si].extend(zt)
                        placed = True
                        break
                if not placed:
                    sub_machines.append(list(zt))
                    sub_loads.append(zl)
            
            new_machines.extend(sub_machines)
            new_loads.extend(sub_loads)
    
    # Remove empty machines (shouldn't happen but safety)
    machines = [m for m in new_machines if len(m) > 0]
    
    # Local improvement: try moving tasks from high-load machines to low-load ones
    # to reduce overtime penalty
    machine_loads = new_loads
    # Sort machines by load descending for easier high-to-low moves
    # Compute overtime proxy for each machine
    def overtime_proxy(load):
        return max(0.0, load - shift) * overtime_rate
    
    # Try to reduce total overtime by moving tasks
    # Only consider moves that don't open new machines
    for _iteration in range(3):
        improved = False
        # Sort machines by load descending
        for i in range(len(machines)):
            if len(machines[i]) <= 1:
                continue
            ml_i = machine_loads[i]
            if ml_i <= shift:
                continue  # No overtime, skip
            # Find task to move: the one with highest proc in this machine
            # that can fit into another machine
            # Sort tasks in machine i by proc descending
            task_order = sorted(machines[i], key=lambda j: proc[j], reverse=True)
            moved = False
            for j in task_order:
                if moved:
                    break
                j_load = proc[j] * (1.0 + orness * volatility[j] * train_spread)
                # Try to move to machine k with lowest load
                best_k = -1
                best_ml_k = float('inf')
                for k in range(len(machines)):
                    if k == i:
                        continue
                    if machine_loads[k] + j_load <= target_load * 1.05:
                        if machine_loads[k] < best_ml_k:
                            best_ml_k = machine_loads[k]
                            best_k = k
                if best_k >= 0:
                    # Move task j from i to best_k
                    machines[i].remove(j)
                    machines[best_k].append(j)
                    machine_loads[i] -= j_load
                    machine_loads[best_k] += j_load
                    improved = True
                    moved = True
            if not improved:
                break
        if not improved:
            break
    
    # Final cleanup: remove any empty machines
    machines = [m for m in machines if len(m) > 0]
    
    # Ensure all tasks are covered (safety check)
    covered = set()
    for m in machines:
        for j in m:
            covered.add(j)
    # If any task is missing, add it to the first machine (shouldn't happen)
    for j in range(n):
        if j not in covered:
            machines[0].append(j)
            covered.add(j)
    
    return machines
