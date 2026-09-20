import math

def solve(instance, params):
    n = instance.get('n', 0)
    if n == 0:
        return []
    
    proc = instance.get('proc', [0.0] * n)
    volatility = instance.get('volatility', [0.0] * n)
    zone = instance.get('zone', [0] * n)
    shift = instance.get('shift', 8.0)
    overtime_rate = instance.get('overtime_rate', 1.5)
    technician_cost = instance.get('technician_cost', 10.0)
    
    orness = params.get('orness', 0.5)
    alpha_grid = params.get('alpha_grid', [0.1, 0.5, 0.9])
    train_spread = params.get('train_spread', 1.0)
    
    # Compute a robust effective duration for each task using a weighted combination of modal and volatility
    # We use a simple convex combination to estimate expected load
    # effective[j] = proc[j] * (1 + volatility[j] * factor)
    # The factor depends on orness and alpha_grid to balance between mean and tail risk
    if alpha_grid:
        # Use the median alpha as a central tendency, but weight by orness
        # Higher orness -> more weight on higher alphas (tail risk)
        alpha_vals = sorted(alpha_grid)
        # Simple heuristic: take a weighted average of alphas
        num_alphas = len(alpha_vals)
        alpha_weighted = 0.0
        for i, a in enumerate(alpha_vals):
            # Linear weight: higher alpha gets more weight if orness is high
            w = (i + 1) / num_alphas
            alpha_weighted += w * a
        alpha_weighted /= num_alphas
    else:
        alpha_weighted = 0.5
    
    # Adjust the volatility impact based on orness
    # orness close to 1 means we care more about worst-case (higher volatility impact)
    vol_factor = 0.5 + orness * 0.5  # ranges from 0.5 to 1.0
    # Also consider train_spread to adjust sensitivity
    vol_factor *= (1.0 + 0.5 * train_spread)
    
    effective = [proc[j] * (1.0 + volatility[j] * vol_factor) for j in range(n)]
    
    # Group tasks by zone to handle common-mode scaling
    # For each zone, we might want to keep tasks together or spread them
    # Here we use a simple approach: sort tasks by effective duration descending
    # and assign to technicians using a greedy LPT-like strategy with overtime consideration
    
    # Sort task indices by effective duration descending
    task_indices = list(range(n))
    task_indices.sort(key=lambda j: effective[j], reverse=True)
    
    # Initialize technicians (machines) as empty lists
    machines = []
    # Track current load for each technician
    loads = []
    
    # Heuristic: assign each task to the technician that minimizes the incremental cost
    # Incremental cost of adding task j to technician k:
    # delta = effective[j] + overtime_rate * (max(0, load_k + effective[j] - shift) - max(0, load_k - shift))
    # If no technician exists, we pay technician_cost + load_k + overtime...
    
    for j in task_indices:
        ej = effective[j]
        best_cost = float('inf')
        best_idx = -1
        
        for k, load in enumerate(loads):
            # Incremental overtime cost
            new_load = load + ej
            old_ot = max(0.0, load - shift)
            new_ot = max(0.0, new_load - shift)
            delta_ot = overtime_rate * (new_ot - old_ot)
            inc_cost = ej + delta_ot
            
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_idx = k
        
        # Decide whether to open a new technician or assign to existing
        # Cost of new technician: technician_cost + ej + overtime_rate * max(0, ej - shift)
        new_technician_cost = technician_cost + ej + overtime_rate * max(0.0, ej - shift)
        
        if not machines or new_technician_cost < best_cost:
            # Open a new technician
            machines.append([j])
            loads.append(ej)
        else:
            # Assign to existing technician
            machines[best_idx].append(j)
            loads[best_idx] += ej
    
    return machines
