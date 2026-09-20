import math

def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    tech_cost = instance['technician_cost']
    
    orness = params.get('orness', 0.5)
    alpha_grid = params.get('alpha_grid', 0.5)
    train_spread = params.get('train_spread', 1.0)
    
    # Compute risk-adjusted duration for each task
    # Use a conservative estimate: modal + volatility * shift * sensitivity
    # Higher volatility tasks get more "padding" to account for scenario variability
    risk_adj = [proc[j] + volatility[j] * shift * 0.5 for j in range(n)]
    
    # Group tasks by zone to consolidate calm-zone work
    zone_tasks = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_tasks:
            zone_tasks[z] = []
        zone_tasks[z].append(j)
    
    # Sort zones by average volatility (calm zones first for consolidation)
    zone_avg_vol = {}
    for z, tasks in zone_tasks.items():
        avg_vol = sum(volatility[j] for j in tasks) / len(tasks)
        zone_avg_vol[z] = avg_vol
    
    # Process zones in order of increasing volatility (calm first)
    sorted_zones = sorted(zone_tasks.keys(), key=lambda z: zone_avg_vol[z])
    
    # Initialize machines (technicians)
    machines = []
    machine_loads = []
    
    # For each zone, try to consolidate tasks into existing machines
    # or create new ones, considering risk-adjusted loads
    for z in sorted_zones:
        tasks_in_zone = zone_tasks[z]
        # Sort tasks within zone by risk-adjusted duration (descending for LPT-like)
        tasks_in_zone.sort(key=lambda j: risk_adj[j], reverse=True)
        
        for j in tasks_in_zone:
            # Find the machine that minimizes risk-adjusted cost increase
            best_machine = -1
            best_cost = float('inf')
            
            # Try adding to existing machines
            for m_idx, load in enumerate(machine_loads):
                new_load = load + risk_adj[j]
                # Cost proxy: technician_cost if new, plus load + overtime
                # For existing machine, we only care about marginal cost
                marginal_cost = risk_adj[j]
                if new_load > shift:
                    marginal_cost += overtime_rate * (new_load - shift)
                # Prefer machines that are not yet in overtime (reserve for volatile)
                # Add penalty if this would push into overtime
                if new_load > shift and load <= shift:
                    marginal_cost += overtime_rate * 0.5  # penalty for crossing shift boundary
                if marginal_cost < best_cost:
                    best_cost = marginal_cost
                    best_machine = m_idx
            
            # Determine if we should open a new machine
            new_machine_cost = tech_cost + risk_adj[j]
            if risk_adj[j] > shift:
                new_machine_cost += overtime_rate * (risk_adj[j] - shift)
            
            if best_machine == -1 or new_machine_cost < best_cost * 0.8:
                # Open new machine
                machines.append([j])
                machine_loads.append(risk_adj[j])
            else:
                machines[best_machine].append(j)
                machine_loads[best_machine] += risk_adj[j]
    
    # Local improvement: try to move tasks from high-overtime machines to reduce risk
    # Focus on machines that are in overtime
    for m_idx in range(len(machines)):
        if machine_loads[m_idx] <= shift:
            continue
        # Try moving the smallest task from this machine to another
        m = machines[m_idx]
        if len(m) <= 1:
            continue
        
        # Find the task with smallest risk-adjusted duration in this machine
        min_task_idx = min(range(len(m)), key=lambda i: risk_adj[m[i]])
        j = m[min_task_idx]
        
        # Try moving to another machine
        best_target = -1
        best_improvement = 0
        
        for t_idx in range(len(machines)):
            if t_idx == m_idx:
                continue
            target_load = machine_loads[t_idx]
            new_target_load = target_load + risk_adj[j]
            
            # Cost reduction from source
            source_reduction = risk_adj[j]
            if machine_loads[m_idx] > shift:
                source_reduction += overtime_rate * min(risk_adj[j], machine_loads[m_idx] - shift)
            
            # Cost increase at target
            target_increase = risk_adj[j]
            if new_target_load > shift and target_load <= shift:
                target_increase += overtime_rate * 0.5
            elif new_target_load > shift:
                target_increase += overtime_rate * (new_target_load - shift)
            
            improvement = source_reduction - target_increase
            if improvement > best_improvement:
                best_improvement = improvement
                best_target = t_idx
        
        if best_target != -1:
            machines[m_idx].remove(j)
            machine_loads[m_idx] -= risk_adj[j]
            machines[best_target].append(j)
            machine_loads[best_target] += risk_adj[j]
    
    # Ensure no empty machines (shouldn't happen, but safety check)
    machines = [m for m in machines if len(m) > 0]
    
    return machines
