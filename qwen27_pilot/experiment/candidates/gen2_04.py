import math

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

    # Determine scenario scaling factors for each zone
    # Use orness to decide the alpha-cut level: higher orness -> more conservative (higher scaling)
    zones = {}
    for j in range(n):
        z = zone[j]
        if z not in zones:
            zones[z] = []
        zones[z].append(j)

    # For each zone, compute a scenario scaling factor
    # Higher volatility -> higher scaling factor
    # Use orness to weight the tail risk
    zone_scale = {}
    for z, tasks in zones.items():
        # Average volatility in this zone
        avg_vol = sum(volatility[j] for j in tasks) / len(tasks)
        # Scale factor: 1 + avg_vol * orness * train_spread
        # This represents a conservative scenario scaling
        zone_scale[z] = 1.0 + avg_vol * orness * train_spread

    # Compute effective duration for each task in the chosen scenario
    eff_dur = [proc[j] * zone_scale[zone[j]] for j in range(n)]

    # Sort tasks by effective duration descending (LPT-like)
    tasks_sorted = sorted(range(n), key=lambda j: eff_dur[j], reverse=True)

    # Determine number of technicians to open
    # Total effective load
    total_load = sum(eff_dur)
    
    # Marginal cost analysis:
    # Opening a new technician costs technician_cost
    # Filling a technician up to shift costs L (linear)
    # Beyond shift, overtime_rate * (L - shift)
    # 
    # Heuristic: open enough technicians so that average load per technician
    # is around shift * (1 + buffer) where buffer accounts for volatility
    # Use a conservative estimate
    buffer = 0.15 * orness * train_spread  # buffer fraction
    target_load_per_tech = shift * (1 + buffer)
    
    # Estimate number of technicians
    num_techs = max(1, math.ceil(total_load / target_load_per_tech))
    
    # Also consider: if opening one more tech saves overtime, do it
    # Marginal overtime cost if one tech is overloaded by delta:
    # overtime_rate * delta
    # vs technician_cost
    # So if we have num_techs and total_load, check if reducing load per tech
    # by opening one more is worth it
    
    # Start with LPT assignment to num_techs technicians
    machines = [[] for _ in range(num_techs)]
    loads = [0.0] * num_techs
    
    for j in tasks_sorted:
        # Assign to technician with minimum current load
        min_idx = 0
        min_load = loads[0]
        for k in range(1, num_techs):
            if loads[k] < min_load:
                min_load = loads[k]
                min_idx = k
        machines[min_idx].append(j)
        loads[min_idx] += eff_dur[j]
    
    # Now consider whether opening more technicians reduces cost
    # Compute current cost
    def compute_cost(machines_list, loads_list):
        total = 0.0
        for k in range(len(machines_list)):
            L = loads_list[k]
            ot = max(0.0, L - shift)
            total += technician_cost + L + overtime_rate * ot
        return total
    
    current_cost = compute_cost(machines, loads)
    
    # Try adding technicians and reassigning
    # For each possible additional technician, check if it reduces cost
    improved = True
    while improved:
        improved = False
        best_cost = current_cost
        best_machines = None
        best_loads = None
        
        # Try adding one more technician
        new_num_techs = len(machines) + 1
        new_machines = [list(m) for m in machines] + [[]]
        new_loads = list(loads) + [0.0]
        
        # Reassign tasks using LPT to new_num_techs
        new_machines = [[] for _ in range(new_num_techs)]
        new_loads = [0.0] * new_num_techs
        for j in tasks_sorted:
            min_idx = 0
            min_load = new_loads[0]
            for k in range(1, new_num_techs):
                if new_loads[k] < min_load:
                    min_load = new_loads[k]
                    min_idx = k
            new_machines[min_idx].append(j)
            new_loads[min_idx] += eff_dur[j]
        
        new_cost = compute_cost(new_machines, new_loads)
        
        if new_cost < best_cost - 1e-9:
            best_cost = new_cost
            best_machines = new_machines
            best_loads = new_loads
            improved = True
        
        # Also try removing one technician if possible
        if len(machines) > 1:
            rem_num_techs = len(machines) - 1
            rem_machines = [[] for _ in range(rem_num_techs)]
            rem_loads = [0.0] * rem_num_techs
            for j in tasks_sorted:
                min_idx = 0
                min_load = rem_loads[0]
                for k in range(1, rem_num_techs):
                    if rem_loads[k] < min_load:
                        min_load = rem_loads[k]
                        min_idx = k
                rem_machines[min_idx].append(j)
                rem_loads[min_idx] += eff_dur[j]
            
            rem_cost = compute_cost(rem_machines, rem_loads)
            if rem_cost < best_cost - 1e-9:
                best_cost = rem_cost
                best_machines = rem_machines
                best_loads = rem_loads
                improved = True
        
        if improved:
            machines = best_machines
            loads = best_loads
            current_cost = best_cost
    
    # Remove empty machines (shouldn't happen but just in case)
    machines = [m for m in machines if len(m) > 0]
    
    # Final check: ensure all tasks are assigned
    assigned = set()
    for m in machines:
        for j in m:
            assigned.add(j)
    
    # If any task is missing, add it to the least loaded technician
    if len(assigned) < n:
        for j in range(n):
            if j not in assigned:
                # Find least loaded machine
                min_idx = 0
                min_load = sum(eff_dur[t] for t in machines[0])
                for k in range(1, len(machines)):
                    load_k = sum(eff_dur[t] for t in machines[k])
                    if load_k < min_load:
                        min_load = load_k
                        min_idx = k
                machines[min_idx].append(j)
                assigned.add(j)
    
    return machines
