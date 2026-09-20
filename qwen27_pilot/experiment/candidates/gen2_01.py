import math

def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Get parameters with defaults
    orness = params.get('orness', 0.5)
    train_spread = params.get('train_spread', 1.0)
    
    # Build a conservative (alpha-cut) duration for each task
    # Higher orness -> more risk-averse -> larger buffer
    # Use a simple convex combination: effective_dur = proc + buffer * volatility
    # buffer scales with orness and train_spread
    buffer_factor = orness * train_spread * 1.5
    effective_durations = [proc[j] + buffer_factor * volatility[j] * proc[j] for j in range(n)]
    
    # Sort tasks by effective duration descending (LPT-like) to balance loads
    task_order = sorted(range(n), key=lambda j: effective_durations[j], reverse=True)
    
    # Group tasks by zone to track zone concentration
    zone_tasks = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_tasks:
            zone_tasks[z] = []
        zone_tasks[z].append(j)
    
    # For each zone, compute a "zone risk" = sum of volatilities weighted by duration
    zone_risk = {}
    for z, tasks in zone_tasks.items():
        risk = sum(volatility[j] * effective_durations[j] for j in tasks)
        zone_risk[z] = risk
    
    # Heuristic: decide number of technicians to open
    # Base on total effective load and shift capacity
    total_load = sum(effective_durations)
    # Each technician can handle 'shift' in regular time; overtime is penalized
    # We want to avoid excessive overtime, so target load per tech ~ shift * (1 + some slack)
    # Slack depends on orness: higher orness -> more slack (more technicians)
    target_load_per_tech = shift * (1.0 + orness * 0.3)
    
    # Minimum number of technicians needed
    min_techs = math.ceil(total_load / target_load_per_tech) if target_load_per_tech > 0 else 1
    
    # Also consider zone dispersion: if one zone has very high risk, we may want
    # to spread it across more technicians to reduce tail risk
    max_zone_risk = max(zone_risk.values()) if zone_risk else 0
    # If max zone risk is high relative to total load, add extra technicians
    zone_penalty = 0
    if total_load > 0 and max_zone_risk > 0:
        zone_ratio = max_zone_risk / total_load
        # If a single zone dominates, add technicians proportional to orness
        zone_penalty = int(math.ceil(orness * zone_ratio * min_techs * 0.5))
    
    num_techs = max(1, min_techs + zone_penalty)
    
    # Initialize technicians with empty loads
    tech_loads = [0.0] * num_techs
    tech_zone_risk = [0.0] * num_techs  # track zone risk per technician
    tech_zone_count = [0] * num_techs   # count of distinct zones per tech
    assignments = [[] for _ in range(num_techs)]
    
    # Assign tasks in descending order of effective duration
    for j in task_order:
        # Find the technician with the lowest "cost" metric
        # Cost metric combines load and zone risk to avoid concentration
        best_tech = -1
        best_cost = float('inf')
        
        for t in range(num_techs):
            # Load cost: prefer lower load
            load_cost = tech_loads[t]
            
            # Zone risk penalty: if this task's zone is already heavily represented
            # on this technician, penalize to spread zone risk
            z = zone[j]
            zone_risk_penalty = 0.0
            if z in zone_tasks:
                # Check if this zone's risk is already high on this tech
                # Use a simple proxy: tech_zone_risk[t] / (tech_zone_count[t] + 1)
                avg_zone_risk = tech_zone_risk[t] / (tech_zone_count[t] + 1)
                # Penalize if adding this task increases concentration
                zone_risk_penalty = orness * volatility[j] * effective_durations[j] * avg_zone_risk / (shift + 1e-9)
            
            # Overtime cost estimate: convex penalty for exceeding shift
            new_load = tech_loads[t] + effective_durations[j]
            overtime = max(0.0, new_load - shift)
            overtime_cost = overtime_rate * overtime * overtime  # convex penalty
            
            # Fixed cost amortized: if this is the first task on this tech, add fixed cost
            fixed_cost = technician_cost if len(assignments[t]) == 0 else 0.0
            
            # Total heuristic cost
            cost = load_cost + zone_risk_penalty + overtime_cost + fixed_cost
            
            if cost < best_cost:
                best_cost = cost
                best_tech = t
        
        # Assign task to best technician
        assignments[best_tech].append(j)
        tech_loads[best_tech] += effective_durations[j]
        z = zone[j]
        if z not in zone_tasks:
            zone_risk[best_tech] = zone_risk.get(best_tech, 0.0) + volatility[j] * effective_durations[j]
        else:
            # Update zone risk for this technician
            if z in zone_tasks:
                # Simple approximation: add this task's contribution
                tech_zone_risk[best_tech] += volatility[j] * effective_durations[j]
                tech_zone_count[best_tech] += 1
    
    # Remove empty technicians (shouldn't happen, but just in case)
    machines = [a for a in assignments if len(a) > 0]
    
    return machines
