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
    alpha_grid = params.get('alpha_grid', 0.5)
    train_spread = params.get('train_spread', 1.0)
    
    # Compute effective load per task: modal duration scaled by a risk-aware factor.
    # Higher volatility and higher orness increase the effective duration to hedge against
    # common-mode zone scaling. We use a simple convex combination that penalizes high
    # volatility more when orness is high.
    effective_durations = []
    for j in range(n):
        vol = volatility[j]
        # Scale factor: 1 + vol * orness * train_spread
        # This increases effective duration for high volatility tasks, with orness controlling
        # the degree of conservatism.
        scale = 1.0 + vol * orness * train_spread
        effective_durations.append(proc[j] * scale)
    
    # Sort tasks by effective duration in descending order (LPT-like)
    task_indices = list(range(n))
    task_indices.sort(key=lambda j: effective_durations[j], reverse=True)
    
    # Initialize with zero machines
    machines = []
    machine_loads = []
    
    # For each task, assign to the machine that minimizes the marginal cost increase.
    # Marginal cost of adding task j to a machine with current load L:
    #   delta_cost = effective_durations[j] + overtime_rate * (max(0, L + eff_j - shift) - max(0, L - shift))
    # This is equivalent to:
    #   if L + eff_j <= shift: delta = eff_j
    #   else: delta = eff_j + overtime_rate * (L + eff_j - shift)
    #   Note: the second term simplifies because max(0, L - shift) is 0 if L <= shift, but
    #   if L > shift, the difference is overtime_rate * eff_j.
    # Actually, the marginal cost is:
    #   eff_j + overtime_rate * max(0, L + eff_j - shift) - overtime_rate * max(0, L - shift)
    # We compute this directly.
    
    for j in task_indices:
        eff_j = effective_durations[j]
        best_machine = -1
        best_delta = float('inf')
        
        for m_idx, load in enumerate(machine_loads):
            # Compute marginal cost
            new_load = load + eff_j
            old_overtime = max(0.0, load - shift)
            new_overtime = max(0.0, new_load - shift)
            delta = eff_j + overtime_rate * (new_overtime - old_overtime)
            if delta < best_delta:
                best_delta = delta
                best_machine = m_idx
        
        # Decide whether to open a new machine or assign to existing one
        # Cost of opening a new machine: technician_cost + eff_j + overtime_rate * max(0, eff_j - shift)
        new_machine_cost = technician_cost + eff_j + overtime_rate * max(0.0, eff_j - shift)
        
        if best_machine == -1 or new_machine_cost < best_delta:
            # Open a new machine
            machines.append([j])
            machine_loads.append(eff_j)
        else:
            machines[best_machine].append(j)
            machine_loads[best_machine] += eff_j
    
    # Remove any empty machines (shouldn't happen, but just in case)
    machines = [m for m in machines if len(m) > 0]
    
    return machines
