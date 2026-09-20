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
    
    # Determine effective duration per task using a weighted alpha-cut
    # We use a conservative duration that accounts for volatility
    # Higher orness means more focus on worst-case (higher alpha)
    # Use a simple weighted average of alpha-cuts
    alpha_weights = []
    for a in alpha_grid:
        # Weight towards higher alphas if orness is high (more risk-averse)
        # Simple linear weight: w = 1 - a for low risk, a for high risk
        # Use orness to blend: weight = (1-orness)*(1-a) + orness*a
        w = (1 - orness) * (1 - a) + orness * a
        alpha_weights.append(w)
    
    total_weight = sum(alpha_weights)
    if total_weight == 0:
        total_weight = 1.0
    
    # Compute effective duration for each task
    # For a given alpha, the duration is proc[j] * (1 + alpha * volatility[j])
    # We take a weighted average over the alpha grid
    effective_proc = [0.0] * n
    for j in range(n):
        dur = 0.0
        for i, a in enumerate(alpha_grid):
            # Duration at this alpha-cut
            d = proc[j] * (1.0 + a * volatility[j])
            dur += alpha_weights[i] * d
        effective_proc[j] = dur / total_weight
    
    # Group tasks by zone to avoid concentrating volatile zones
    # We'll use a list of open technicians, each with current load
    # Sort tasks by a priority: larger effective duration first, with tie-break by zone volatility
    # Compute zone-level volatility for tie-breaking
    zone_vol = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_vol:
            zone_vol[z] = volatility[j]
        else:
            # Average volatility in zone
            zone_vol[z] = max(zone_vol[z], volatility[j])  # Use max for conservatism
    
    # Sort tasks: primary by effective duration descending, secondary by zone volatility descending
    task_order = sorted(range(n), key=lambda j: (-effective_proc[j], -zone_vol[zone[j]]))
    
    # Greedy assignment with dynamic opening of new technicians
    # A technician is opened if the marginal cost of adding a task to an existing
    # technician exceeds the cost of opening a new one
    technicians = []  # list of lists of task indices
    tech_loads = []   # current load per technician
    tech_zone_counts = []  # count of tasks per zone per technician, to avoid concentration
    
    # For marginal cost comparison:
    # Cost of adding task j to technician with load L:
    #   delta = effective_proc[j] + overtime_rate * max(0, L + effective_proc[j] - shift) - overtime_rate * max(0, L - shift)
    # Cost of opening new technician: technician_cost + effective_proc[j] + overtime_rate * max(0, effective_proc[j] - shift)
    # Open new if delta > new_tech_cost
    
    for j in task_order:
        ej = effective_proc[j]
        zj = zone[j]
        
        # Find the best existing technician (lowest marginal cost increase)
        best_idx = -1
        best_delta = float('inf')
        
        for t in range(len(technicians)):
            L = tech_loads[t]
            # Marginal overtime cost of adding ej
            ot_before = max(0.0, L - shift)
            ot_after = max(0.0, L + ej - shift)
            delta = ej + overtime_rate * (ot_after - ot_before)
            
            # Penalize if this technician already has many tasks from the same zone
            # to avoid concentration
            if tech_zone_counts[t].get(zj, 0) > 0:
                delta += 0.5 * ej  # small penalty for same-zone concentration
            
            if delta < best_delta:
                best_delta = delta
                best_idx = t
        
        # Cost of opening a new technician
        ot_new = max(0.0, ej - shift)
        new_tech_cost = technician_cost + ej + overtime_rate * ot_new
        
        # Decide: open new or add to existing
        if best_idx == -1 or best_delta > new_tech_cost:
            # Open new technician
            technicians.append([j])
            tech_loads.append(ej)
            tech_zone_counts.append({zj: 1})
        else:
            # Add to best existing technician
            technicians[best_idx].append(j)
            tech_loads[best_idx] += ej
            tc = tech_zone_counts[best_idx]
            tc[zj] = tc.get(zj, 0) + 1
    
    return technicians
