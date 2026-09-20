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
    
    # If no tasks, return empty
    if n == 0:
        return []
    
    # Group tasks by zone for consolidation of calm-zone work
    zone_groups = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_groups:
            zone_groups[z] = []
        zone_groups[z].append(j)
    
    # Calculate risk-adjusted duration for each task
    # Higher volatility tasks get a premium to encourage spreading them out
    # Use orness to control the degree of risk aversion
    # When orness is high, we are more risk-averse
    risk_adjusted = []
    for j in range(n):
        vol = volatility[j]
        # The risk premium: more volatile tasks get a higher effective duration
        # This encourages assigning volatile tasks to less loaded technicians
        # Use a convex penalty that increases with volatility
        # The premium is proportional to volatility and orness
        premium = vol * orness * train_spread
        # Effective duration for scheduling purposes
        eff = proc[j] * (1.0 + premium)
        risk_adjusted.append(eff)
    
    # Sort tasks by risk-adjusted duration in descending order (LPT-like)
    # This helps balance loads and reduces overtime
    task_order = sorted(range(n), key=lambda j: risk_adjusted[j], reverse=True)
    
    # Greedy assignment: assign each task to the technician with the lowest
    # risk-adjusted load, but with a bias toward consolidation for calm tasks
    # We use a threshold to decide whether to open a new technician or add to existing
    
    machines = []  # list of lists of task indices
    loads = []     # current load (effective) for each machine
    real_loads = [] # actual modal load for each machine
    
    # Threshold for opening a new machine vs adding to existing
    # If the best existing machine would exceed shift significantly, open new
    # The threshold depends on overtime_rate and technician_cost
    # Breakeven: overtime_cost vs technician_cost
    # overtime_rate * (L - shift) = technician_cost
    # L = shift + technician_cost / overtime_rate
    if overtime_rate > 0:
        breakeven_load = shift + technician_cost / overtime_rate
    else:
        breakeven_load = float('inf')
    
    for j in task_order:
        # Find the machine with the lowest effective load
        best_idx = -1
        best_load = float('inf')
        
        for i in range(len(machines)):
            if loads[i] < best_load:
                best_load = loads[i]
                best_idx = i
        
        # Decide whether to open a new machine or add to existing
        # If no machines exist, must open new
        if len(machines) == 0:
            machines.append([j])
            loads.append(risk_adjusted[j])
            real_loads.append(proc[j])
        else:
            # Check if adding to best machine would be too expensive
            new_load = best_load + risk_adjusted[j]
            new_real = real_loads[best_idx] + proc[j]
            
            # Cost of adding to existing machine
            add_cost = new_load + overtime_rate * max(0.0, new_real - shift)
            
            # Cost of opening new machine
            new_machine_cost = technician_cost + risk_adjusted[j] + overtime_rate * max(0.0, proc[j] - shift)
            
            # If opening new machine is cheaper, or if the best machine is already
            # over the breakeven point, open new
            if new_load > breakeven_load or new_machine_cost < add_cost:
                machines.append([j])
                loads.append(risk_adjusted[j])
                real_loads.append(proc[j])
            else:
                machines[best_idx].append(j)
                loads[best_idx] = new_load
                real_loads[best_idx] = new_real
    
    # Local improvement: try to reduce risk-adjusted overtime by moving tasks
    # between machines. We do a simple pass of moves that reduce the total
    # risk-adjusted cost.
    
    def calc_cost(machine_loads, real_machine_loads):
        total = 0.0
        for i in range(len(machine_loads)):
            total += technician_cost
            total += machine_loads[i]
            total += overtime_rate * max(0.0, real_machine_loads[i] - shift)
        return total
    
    def calc_risk_cost(machine_loads, real_machine_loads, vols):
        # Risk-adjusted cost with OWA-like tail emphasis
        total = 0.0
        for i in range(len(machine_loads)):
            total += technician_cost
            total += machine_loads[i]
            ot = max(0.0, real_machine_loads[i] - shift)
            # Convex overtime penalty with risk emphasis
            total += overtime_rate * ot * ot * orness
        return total
    
    # Simple local search: for each task, try moving it to another machine
    # if it reduces the total risk-adjusted cost
    improved = True
    max_iterations = 3
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for j in range(n):
            # Find current machine for task j
            cur_machine = -1
            for i in range(len(machines)):
                if j in machines[i]:
                    cur_machine = i
                    break
            if cur_machine == -1:
                continue
            
            # Try moving to each other machine
            best_target = -1
            best_delta = 0.0
            cur_cost = calc_risk_cost(loads, real_loads, volatility)
            
            for target in range(len(machines)):
                if target == cur_machine:
                    continue
                
                # Compute cost delta
                # Remove j from cur_machine
                new_cur_load = loads[cur_machine] - risk_adjusted[j]
                new_cur_real = real_loads[cur_machine] - proc[j]
                # Add j to target
                new_target_load = loads[target] + risk_adjusted[j]
                new_target_real = real_loads[target] + proc[j]
                
                # Cost of cur_machine after removal
                cur_after = technician_cost + new_cur_load + overtime_rate * max(0.0, new_cur_real - shift)
                cur_after += overtime_rate * max(0.0, new_cur_real - shift) * orness
                # Cost of target after addition
                target_after = technician_cost + new_target_load + overtime_rate * max(0.0, new_target_real - shift)
                target_after += overtime_rate * max(0.0, new_target_real - shift) * orness
                
                # Original costs
                cur_before = technician_cost + loads[cur_machine] + overtime_rate * max(0.0, real_loads[cur_machine] - shift)
                cur_before += overtime_rate * max(0.0, real_loads[cur_machine] - shift) * orness
                target_before = technician_cost + loads[target] + overtime_rate * max(0.0, real_loads[target] - shift)
                target_before += overtime_rate * max(0.0, real_loads[target] - shift) * orness
                
                delta = (cur_after + target_after) - (cur_before + target_before)
                
                if delta < -1e-9 and delta < best_delta:
                    best_delta = delta
                    best_target = target
            
            if best_target != -1:
                # Perform the move
                machines[cur_machine].remove(j)
                machines[best_target].append(j)
                loads[cur_machine] -= risk_adjusted[j]
                real_loads[cur_machine] -= proc[j]
                loads[best_target] += risk_adjusted[j]
                real_loads[best_target] += proc[j]
                improved = True
    
    # Remove any empty machines (shouldn't happen, but just in case)
    machines = [m for m in machines if len(m) > 0]
    
    return machines
