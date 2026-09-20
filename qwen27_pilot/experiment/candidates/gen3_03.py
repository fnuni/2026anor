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
    alpha_grid = params.get('alpha_grid', [0.1, 0.5, 0.9])
    
    # Determine risk-adjusted duration for each task
    # Higher volatility means we should plan for longer durations
    # Use a weighted combination: modal + risk premium based on volatility
    risk_adjusted = []
    for j in range(n):
        # Volatility in [0, ~1], use it to scale up the planned duration
        # The premium is proportional to volatility and the shift length
        premium = volatility[j] * shift * 0.3  # 30% of shift as max premium
        adj = proc[j] + premium
        risk_adjusted.append(adj)
    
    # Group tasks by zone to consolidate calm-zone work
    zone_groups = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_groups:
            zone_groups[z] = []
        zone_groups[z].append(j)
    
    # Compute average volatility per zone to identify calm zones
    zone_avg_vol = {}
    for z, tasks in zone_groups.items():
        total_vol = sum(volatility[j] for j in tasks)
        zone_avg_vol[z] = total_vol / len(tasks)
    
    # Sort zones by average volatility (calm zones first)
    sorted_zones = sorted(zone_groups.keys(), key=lambda z: zone_avg_vol[z])
    
    # Strategy: Use a bin-packing approach with risk-adjusted durations
    # Sort tasks within each zone by risk-adjusted duration (decreasing) for better packing
    # Then assign tasks to technicians using a First-Fit Decreasing approach
    # with a threshold that considers overtime cost
    
    # Collect all tasks and sort by risk-adjusted duration (decreasing)
    task_indices = list(range(n))
    # Sort by risk-adjusted duration descending, with tie-breaking by zone volatility
    task_indices.sort(key=lambda j: (-risk_adjusted[j], zone_avg_vol[zone[j]]))
    
    # Determine the effective shift limit considering overtime cost
    # The breakeven point where overtime becomes too expensive
    # If technician_cost is high, we want to minimize number of technicians
    # If overtime_rate is high, we want to keep loads under shift
    
    # Initialize empty technicians
    machines = []
    loads = []
    task_counts = []
    
    # For each task, try to fit into an existing technician or open a new one
    for j in task_indices:
        best_machine = -1
        best_cost_delta = float('inf')
        
        # Try to fit into an existing machine
        for m in range(len(machines)):
            new_load = loads[m] + risk_adjusted[j]
            # Compute the incremental cost
            old_ot = max(0, loads[m] - shift)
            new_ot = max(0, new_load - shift)
            delta_cost = overtime_rate * (new_ot - old_ot)
            # Also consider the risk: if load is close to shift, volatile tasks add risk
            # Add a penalty for being near the shift boundary
            if new_load > shift * 0.9:
                # Proximity penalty based on volatility
                vol_penalty = volatility[j] * (new_load - shift * 0.9) * overtime_rate * 0.5
                delta_cost += vol_penalty
            
            if delta_cost < best_cost_delta:
                best_cost_delta = delta_cost
                best_machine = m
        
        # Cost of opening a new machine
        new_machine_cost = technician_cost + risk_adjusted[j]
        new_machine_ot = max(0, risk_adjusted[j] - shift)
        new_machine_cost += overtime_rate * new_machine_ot
        
        if best_machine == -1 or new_machine_cost < best_cost_delta:
            # Open a new machine
            machines.append([j])
            loads.append(risk_adjusted[j])
            task_counts.append(1)
        else:
            # Assign to best existing machine
            machines[best_machine].append(j)
            loads[best_machine] += risk_adjusted[j]
            task_counts[best_machine] += 1
    
    # Local improvement: try to move tasks from high-overtime machines to reduce risk
    # Only do this if it doesn't increase the number of machines
    improved = True
    max_iterations = 3
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Find the machine with the highest overtime
        if not machines:
            break
        
        # Sort machines by load descending
        machine_order = sorted(range(len(machines)), key=lambda m: loads[m], reverse=True)
        
        for m in machine_order:
            if loads[m] <= shift:
                continue
            
            # Try to move the smallest task from this machine to another machine
            if len(machines[m]) <= 1:
                continue
            
            # Sort tasks in machine m by risk-adjusted duration ascending
            tasks_in_m = sorted(machines[m], key=lambda j: risk_adjusted[j])
            
            moved = False
            for j in tasks_in_m:
                if len(machines[m]) <= 1:
                    break
                
                # Try to move task j to another machine
                best_target = -1
                best_delta = float('inf')
                
                for t in range(len(machines)):
                    if t == m:
                        continue
                    
                    new_load_t = loads[t] + risk_adjusted[j]
                    old_load_t = loads[t]
                    old_load_m = loads[m]
                    new_load_m = loads[m] - risk_adjusted[j]
                    
                    # Compute cost change
                    old_ot_m = max(0, old_load_m - shift)
                    new_ot_m = max(0, new_load_m - shift)
                    old_ot_t = max(0, old_load_t - shift)
                    new_ot_t = max(0, new_load_t - shift)
                    
                    delta = overtime_rate * ((new_ot_m + new_ot_t) - (old_ot_m + old_ot_t))
                    
                    # Risk adjustment
                    if new_load_t > shift * 0.9:
                        delta += volatility[j] * (new_load_t - shift * 0.9) * overtime_rate * 0.3
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_target = t
                
                if best_target != -1 and best_delta < -1e-9:
                    # Move task j from m to best_target
                    machines[m].remove(j)
                    machines[best_target].append(j)
                    loads[m] -= risk_adjusted[j]
                    loads[best_target] += risk_adjusted[j]
                    task_counts[m] -= 1
                    task_counts[best_target] += 1
                    improved = True
                    moved = True
                    break
            
            if moved:
                break
    
    # Remove any empty machines (shouldn't happen, but just in case)
    machines = [m for m in machines if len(m) > 0]
    
    return machines
