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
    alpha_grid = params.get('alpha_grid', [0.0, 0.5, 1.0])
    train_spread = params.get('train_spread', 1.0)
    
    if n == 0:
        return []
    
    # Estimate the effective load per task using a weighted combination of modal and volatile scenarios
    # Use orness to control the optimism/pessimism of the fuzzy estimate
    effective_loads = []
    for j in range(n):
        p = proc[j]
        v = volatility[j]
        # Create a simple fuzzy interval: [p*(1-v), p*(1+v)]
        low = p * (1.0 - v)
        high = p * (1.0 + v)
        # Use orness to blend between low and high
        # orness=0 -> optimistic (low), orness=1 -> pessimistic (high)
        eff = low * (1.0 - orness) + high * orness
        effective_loads.append(eff)
    
    # Group tasks by zone to avoid concentration of volatile zones
    zone_tasks = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_tasks:
            zone_tasks[z] = []
        zone_tasks[z].append(j)
    
    # Calculate marginal cost of assigning a task to a technician vs opening a new one
    # For a technician with load L, the marginal cost of adding a task with load d is:
    # d + overtime_rate * max(0, L + d - shift) - overtime_rate * max(0, L - shift)
    # = d + overtime_rate * max(0, min(d, shift - L)) if L < shift, else d
    # Actually simpler: if L + d <= shift, marginal cost is d. If L + d > shift, marginal cost is d + overtime_rate * (L + d - shift)
    # But we need to compare with opening a new technician: technician_cost + d
    
    # Sort tasks by effective load descending (largest first)
    sorted_tasks = sorted(range(n), key=lambda j: effective_loads[j], reverse=True)
    
    # Initialize technicians
    technicians = []  # list of (load, task_indices)
    
    # For each task, decide whether to assign to an existing technician or open a new one
    for j in sorted_tasks:
        d = effective_loads[j]
        z = zone[j]
        
        # Find the best technician to assign to, or open a new one
        best_cost = float('inf')
        best_tech_idx = -1
        
        for i, (load, tasks) in enumerate(technicians):
            new_load = load + d
            # Cost of assigning to this technician
            if new_load <= shift:
                cost = d
            else:
                cost = d + overtime_rate * (new_load - shift)
            
            # Penalize if this technician already has tasks from the same zone (to spread volatility)
            same_zone_count = sum(1 for t in tasks if zone[t] == z)
            if same_zone_count > 0:
                cost += 0.1 * same_zone_count * d  # small penalty for zone concentration
            
            if cost < best_cost:
                best_cost = cost
                best_tech_idx = i
        
        # Cost of opening a new technician
        new_tech_cost = technician_cost + d
        
        if best_tech_idx == -1 or new_tech_cost < best_cost:
            technicians.append((d, [j]))
        else:
            load, tasks = technicians[best_tech_idx]
            new_load = load + d
            technicians[best_tech_idx] = (new_load, tasks + [j])
    
    # Convert to list of lists
    machines = [tasks for (_, tasks) in technicians]
    
    # Ensure all tasks are covered (safety check)
    assigned = set()
    for tasks in machines:
        for t in tasks:
            assigned.add(t)
    
    # If any task is missing, add it to the first technician (shouldn't happen)
    missing = set(range(n)) - assigned
    if missing:
        if not machines:
            machines.append(list(missing))
        else:
            machines[0].extend(missing)
    
    return machines
