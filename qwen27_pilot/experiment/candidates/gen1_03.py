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
    
    # Get parameters
    orness = params.get('orness', 0.5)
    train_spread = params.get('train_spread', 1.0)
    
    # Compute effective duration for each task considering volatility and orness
    # Higher orness -> more pessimistic (higher effective duration)
    # Use a simple scaling: effective = proc * (1 + orness * volatility * train_spread)
    effective = [proc[j] * (1.0 + orness * volatility[j] * train_spread) for j in range(n)]
    
    # Sort tasks by effective duration descending (LPT-like heuristic)
    task_indices = list(range(n))
    task_indices.sort(key=lambda j: -effective[j])
    
    # Determine optimal number of machines
    total_load = sum(effective)
    
    # Try different numbers of machines and pick the one with minimum estimated cost
    # Cost model: for a machine with load L, cost = technician_cost + L + overtime_rate * max(0, L - shift)
    
    def estimate_cost(num_machines, loads):
        total = 0.0
        for L in loads:
            total += technician_cost + L + overtime_rate * max(0.0, L - shift)
        return total
    
    # Use LPT to assign tasks to machines
    def lpt_assign(num_machines):
        loads = [0.0] * num_machines
        machine_tasks = [[] for _ in range(num_machines)]
        for j in task_indices:
            # Assign to machine with smallest current load
            min_load = float('inf')
            min_idx = 0
            for m in range(num_machines):
                if loads[m] < min_load:
                    min_load = loads[m]
                    min_idx = m
            loads[min_idx] += effective[j]
            machine_tasks[min_idx].append(j)
        return loads, machine_tasks
    
    # Find optimal number of machines
    # Minimum machines: 1, Maximum: n
    # But we can bound it: at least ceil(total_load / (shift * (1 + overtime_rate)))
    # to avoid excessive overtime
    
    # Estimate optimal num_machines
    # If we have k machines, ideal load per machine is total_load / k
    # Cost per machine ~ technician_cost + total_load/k + overtime_rate * max(0, total_load/k - shift)
    # Total cost ~ k * technician_cost + total_load + k * overtime_rate * max(0, total_load/k - shift)
    
    # Find k that minimizes this
    best_cost = float('inf')
    best_k = 1
    best_loads = None
    best_machine_tasks = None
    
    # Upper bound on k: n
    # Lower bound: 1
    # We can be smart: only check k in a reasonable range
    # The optimal k is around total_load / shift (if overtime is expensive)
    
    # Check k from 1 to min(n, some reasonable upper bound)
    # Upper bound: if technician_cost is small, we might want many machines
    # Let's check up to n but skip intelligently
    max_k = n
    
    # To keep it fast, check a subset
    # Check k = 1, 2, ..., max_k
    # For n up to a few hundred, this is fine
    
    for k in range(1, max_k + 1):
        loads, machine_tasks = lpt_assign(k)
        cost = estimate_cost(k, loads)
        if cost < best_cost:
            best_cost = cost
            best_k = k
            best_loads = loads
            best_machine_tasks = machine_tasks
    
    # Remove empty machines (shouldn't happen with LPT, but just in case)
    result = [mt for mt in best_machine_tasks if len(mt) > 0]
    
    # Ensure all tasks are covered
    covered = set()
    for mt in result:
        for j in mt:
            covered.add(j)
    if len(covered) != n:
        # Fallback: assign uncovered tasks to new machines
        for j in range(n):
            if j not in covered:
                result.append([j])
    
    return result
