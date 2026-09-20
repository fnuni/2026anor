import math

def solve(instance, params):
    n = instance.get('n', 0)
    if n == 0:
        return []
    
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    orness = params.get('orness', 0.5)
    alpha_grid = params.get('alpha_grid', [0.0, 0.5, 1.0])
    train_spread = params.get('train_spread', 1.0)
    
    # Compute a robust effective duration for each task
    # Using a weighted combination of modal and worst-case estimates
    effective_durations = []
    for j in range(n):
        p = proc[j]
        v = volatility[j]
        # Conservative estimate: modal + volatility * some factor
        # Use orness to blend between optimistic and pessimistic
        # Higher orness -> more conservative
        robust_factor = 1.0 + orness * v * train_spread
        eff = p * robust_factor
        effective_durations.append(eff)
    
    # Sort tasks by effective duration in descending order (LPT-like)
    sorted_tasks = sorted(range(n), key=lambda j: effective_durations[j], reverse=True)
    
    # Determine number of technicians
    # Estimate total work and decide how many machines to open
    total_work = sum(effective_durations)
    
    # Heuristic: start with a number of machines that balances fixed cost vs overtime
    # Try different machine counts and pick the best heuristic estimate
    # For simplicity, use a greedy LPT with a reasonable number of machines
    
    # Estimate optimal number of machines
    # Cost per machine if load is L: technician_cost + L + overtime_rate * max(0, L - shift)
    # We want to find k such that total cost is minimized
    
    # Simple heuristic: try k from 1 to n, but limit to a small range
    # Since n can be up to a few hundred, we limit the search
    
    # First, compute a rough estimate of optimal k
    # If all tasks fit in shift, k = ceil(total_work / shift) is a good start
    # But we also need to consider fixed cost
    
    # Let's try a range of k values
    best_cost = float('inf')
    best_partition = None
    
    # Determine range of k to try
    # Minimum k: 1
    # Maximum k: n (each task on its own machine)
    # But we limit to avoid O(n^2) overhead
    
    # Use a heuristic to narrow down k
    # The marginal cost of adding a machine is technician_cost minus the reduction in overtime
    # Rough estimate: k_opt ~ total_work / shift, adjusted for fixed cost
    
    # Try k values around the estimated optimum
    k_min = 1
    k_max = min(n, int(math.ceil(total_work / shift)) + 5 if shift > 0 else n)
    k_max = max(k_min, k_max)
    
    # Limit the number of k values to try for performance
    num_k_to_try = min(50, k_max - k_min + 1)
    k_step = max(1, (k_max - k_min + 1) // num_k_to_try)
    
    for k in range(k_min, k_max + 1, k_step):
        # Greedy LPT assignment for k machines
        machine_loads = [0.0] * k
        partition = [[] for _ in range(k)]
        
        for task in sorted_tasks:
            eff_dur = effective_durations[task]
            # Assign to machine with smallest current load
            min_load = float('inf')
            min_idx = 0
            for m in range(k):
                if machine_loads[m] < min_load:
                    min_load = machine_loads[m]
                    min_idx = m
            
            machine_loads[min_idx] += eff_dur
            partition[min_idx].append(task)
        
        # Compute cost for this partition
        cost = 0.0
        for m in range(k):
            L = machine_loads[m]
            if L > 0:
                cost += technician_cost + L + overtime_rate * max(0.0, L - shift)
            else:
                # Empty machine, but we shouldn't have empty machines in this heuristic
                # If load is 0, skip (but LPT shouldn't create empty machines if k <= n)
                pass
        
        # Remove empty machines
        non_empty_partition = [p for p in partition if len(p) > 0]
        
        # Recompute cost without empty machines
        cost = 0.0
        for p in non_empty_partition:
            L = sum(effective_durations[t] for t in p)
            cost += technician_cost + L + overtime_rate * max(0.0, L - shift)
        
        if cost < best_cost:
            best_cost = cost
            best_partition = non_empty_partition
    
    # If best_partition is None, fall back to single machine
    if best_partition is None:
        best_partition = [list(range(n))]
    
    return best_partition
