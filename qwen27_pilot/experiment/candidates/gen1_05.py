import math

def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Extract parameters with defaults
    orness = params.get('orness', 0.5)
    alpha_grid = params.get('alpha_grid', [0.0, 0.5, 1.0])
    train_spread = params.get('train_spread', 1.0)
    
    # Determine the alpha cut to use based on orness
    # Higher orness means more risk-averse, so use a higher alpha (more conservative)
    # Interpolate alpha from the grid based on orness
    if len(alpha_grid) == 0:
        alpha = 0.5
    elif len(alpha_grid) == 1:
        alpha = alpha_grid[0]
    else:
        # Map orness [0,1] to index in alpha_grid
        idx = orness * (len(alpha_grid) - 1)
        i0 = int(math.floor(idx))
        i1 = min(i0 + 1, len(alpha_grid) - 1)
        frac = idx - i0
        alpha = alpha_grid[i0] * (1 - frac) + alpha_grid[i1] * frac
    
    # Compute effective task durations considering fuzzy uncertainty
    # For each task, scale the modal duration by a factor that accounts for volatility and alpha
    # Higher alpha means we consider more extreme scenarios (higher tail risk)
    # We use a simple scaling: effective_duration = proc * (1 + alpha * volatility * train_spread)
    # This is a heuristic approximation of the alpha-cut upper bound
    
    effective_proc = []
    for j in range(n):
        # Scale by volatility and alpha
        # The factor represents the expected worst-case duration under the alpha cut
        scale_factor = 1.0 + alpha * volatility[j] * train_spread
        effective_proc.append(proc[j] * scale_factor)
    
    # Group tasks by zone to handle common-mode scaling
    # Tasks in the same zone move together, so we should consider their aggregate load
    zone_tasks = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_tasks:
            zone_tasks[z] = []
        zone_tasks[z].append(j)
    
    # Compute zone-level aggregate effective load
    zone_load = {}
    for z, tasks in zone_tasks.items():
        total = sum(effective_proc[j] for j in tasks)
        zone_load[z] = total
    
    # Heuristic: Sort tasks by effective duration descending (LPT-like)
    # This is a classic heuristic for parallel machine scheduling
    task_order = sorted(range(n), key=lambda j: -effective_proc[j])
    
    # Determine the number of technicians to open
    # We need to balance technician_cost vs overtime penalty
    # A simple heuristic: start with enough technicians to avoid overtime,
    # but not too many to avoid excessive technician_cost
    
    total_load = sum(effective_proc)
    
    # Estimate optimal number of technicians
    # If we use k technicians, average load is total_load/k
    # Cost per technician = technician_cost + load + overtime_rate * max(0, load - shift)
    # We want to minimize total cost
    
    # Try a range of k values and pick the best
    min_k = 1
    max_k = n  # worst case, one task per technician
    
    best_cost = float('inf')
    best_k = min_k
    
    # Sample a few k values to avoid O(n^2)
    # Use logarithmic sampling
    k_candidates = set()
    for k in range(min_k, max_k + 1):
        if k <= 10 or k % 10 == 0 or k == max_k:
            k_candidates.add(k)
    # Also add k based on total_load / shift
    k_from_shift = max(1, int(math.ceil(total_load / shift)))
    k_candidates.add(k_from_shift)
    k_candidates.add(max(1, k_from_shift - 1))
    k_candidates.add(max(1, k_from_shift + 1))
    
    k_candidates = sorted(list(k_candidates))
    
    for k in k_candidates:
        # Estimate cost with k technicians
        avg_load = total_load / k
        # Approximate cost assuming balanced load
        if avg_load <= shift:
            est_cost = k * (technician_cost + avg_load)
        else:
            est_cost = k * (technician_cost + avg_load + overtime_rate * (avg_load - shift))
        
        # Add a penalty for imbalance (rough estimate)
        # The more uneven the distribution, the higher the actual cost
        # We'll use a simple heuristic: add variance estimate
        # For now, just use the average cost
        if est_cost < best_cost:
            best_cost = est_cost
            best_k = k
    
    # Now assign tasks to best_k technicians using LPT
    num_machines = best_k
    
    # Initialize machines with empty lists
    machines = [[] for _ in range(num_machines)]
    machine_loads = [0.0] * num_machines
    
    # Assign tasks in descending order of effective duration
    for j in task_order:
        # Find the machine with the smallest current load
        min_load = float('inf')
        target_machine = 0
        for m in range(num_machines):
            if machine_loads[m] < min_load:
                min_load = machine_loads[m]
                target_machine = m
        
        machines[target_machine].append(j)
        machine_loads[target_machine] += effective_proc[j]
    
    # Remove empty machines
    machines = [m for m in machines if len(m) > 0]
    
    # If no machines (shouldn't happen if n > 0), create one
    if len(machines) == 0 and n > 0:
        machines = [list(range(n))]
    
    return machines
