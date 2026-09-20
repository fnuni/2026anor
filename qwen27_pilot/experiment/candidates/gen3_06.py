def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    orness = params.get('orness', 0.5)
    alpha_grid = params.get('alpha_grid', [0.0, 0.25, 0.5, 0.75, 1.0])
    train_spread = params.get('train_spread', 0.2)
    
    if n == 0:
        return []
    
    # Group tasks by zone to exploit common-mode scaling
    zone_tasks = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_tasks:
            zone_tasks[z] = []
        zone_tasks[z].append(j)
    
    # Compute risk-adjusted duration for each task
    # Higher volatility tasks get more conservative estimates
    adjusted_proc = []
    for j in range(n):
        # Use a weighted combination: modal duration plus volatility-based buffer
        # The buffer scales with volatility and orness (risk aversion)
        buffer = volatility[j] * shift * orness
        adjusted_proc.append(proc[j] + buffer)
    
    # Sort tasks by adjusted duration in descending order (largest first)
    sorted_tasks = sorted(range(n), key=lambda j: -adjusted_proc[j])
    
    # Initialize machines
    machines = []
    machine_loads = []
    
    # For each task, assign to the machine that minimizes incremental cost
    # Incremental cost = overtime penalty if adding this task causes overtime
    # We use a threshold-based approach: try to keep loads under shift + buffer
    
    # Compute a dynamic threshold that accounts for risk
    # Higher orness means we want more slack
    risk_threshold = shift * (1.0 + train_spread * orness)
    
    for j in sorted_tasks:
        best_machine = -1
        best_cost = float('inf')
        
        # Check existing machines
        for m in range(len(machines)):
            new_load = machine_loads[m] + adjusted_proc[j]
            # Compute incremental cost of adding this task
            # Base cost is the load increase
            incremental = adjusted_proc[j]
            # Overtime penalty if we cross the threshold
            old_overtime = max(0, machine_loads[m] - shift)
            new_overtime = max(0, new_load - shift)
            incremental += overtime_rate * (new_overtime - old_overtime)
            
            # Add a penalty for exceeding risk threshold (encourages consolidation)
            if new_load > risk_threshold:
                incremental += 0.5 * (new_load - risk_threshold)
            
            if incremental < best_cost:
                best_cost = incremental
                best_machine = m
        
        # Decide whether to open a new machine
        # New machine cost is technician_cost + adjusted_proc[j]
        new_machine_cost = technician_cost + adjusted_proc[j]
        
        # Only open new machine if it's significantly cheaper
        # or if no existing machine can take it without excessive overtime
        if best_machine == -1 or new_machine_cost < best_cost * 0.8:
            machines.append([j])
            machine_loads.append(adjusted_proc[j])
        else:
            machines[best_machine].append(j)
            machine_loads[best_machine] += adjusted_proc[j]
    
    # Local search: try to reduce overtime by moving tasks between machines
    # This is a simplified version to keep runtime low
    improved = True
    max_iterations = 3
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for m in range(len(machines)):
            if not machines[m]:
                continue
            
            # Try moving the largest task from this machine to another
            # or to a new machine if it reduces total cost
            # Sort tasks in this machine by adjusted duration descending
            machine_tasks = sorted(machines[m], key=lambda j: -adjusted_proc[j])
            
            for j in machine_tasks[:2]:  # Only try top 2 tasks per machine
                # Try moving to other machines
                best_target = -1
                best_delta = 0
                
                for m2 in range(len(machines)):
                    if m2 == m:
                        continue
                    
                    # Cost of removing j from m
                    old_load_m = machine_loads[m]
                    new_load_m = old_load_m - adjusted_proc[j]
                    old_overtime_m = max(0, old_load_m - shift)
                    new_overtime_m = max(0, new_load_m - shift)
                    delta_m = overtime_rate * (new_overtime_m - old_overtime_m)
                    
                    # Cost of adding j to m2
                    old_load_m2 = machine_loads[m2]
                    new_load_m2 = old_load_m2 + adjusted_proc[j]
                    old_overtime_m2 = max(0, old_load_m2 - shift)
                    new_overtime_m2 = max(0, new_load_m2 - shift)
                    delta_m2 = overtime_rate * (new_overtime_m2 - old_overtime_m2)
                    
                    total_delta = delta_m + delta_m2
                    
                    if total_delta < best_delta - 1e-9:
                        best_delta = total_delta
                        best_target = m2
                
                if best_target != -1:
                    # Perform the move
                    machines[m].remove(j)
                    machines[best_target].append(j)
                    machine_loads[m] -= adjusted_proc[j]
                    machine_loads[best_target] += adjusted_proc[j]
                    improved = True
    
    # Remove empty machines (shouldn't happen but just in case)
    machines = [m for m in machines if m]
    
    return machines
