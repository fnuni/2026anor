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

    # Aggregate tasks by zone
    zone_tasks = {}
    for j in range(n):
        z = zone[j]
        if z not in zone_tasks:
            zone_tasks[z] = []
        zone_tasks[z].append(j)

    # Compute modal and worst-case total load per zone
    zone_modal = {}
    zone_worst = {}
    for z, tasks in zone_tasks.items():
        modal_sum = sum(proc[j] for j in tasks)
        # Worst-case: assume each task scales up by (1 + volatility[j])
        # Common-mode factor: worst case is max over tasks of (1+volatility), but since
        # it's common-mode, the factor is the same for all in zone. Use max volatility
        # in zone as the worst-case scale factor.
        max_vol = max(volatility[j] for j in tasks)
        worst_sum = modal_sum * (1 + max_vol)
        zone_modal[z] = modal_sum
        zone_worst[z] = worst_sum

    # Effective penalty per unit of load
    # Cost = technician_cost + L + overtime_rate * max(0, L - shift)
    # Marginal cost: 1 for L <= shift, 1 + overtime_rate for L > shift
    # We want to balance between opening new technicians and overloading.
    # A good threshold: when the marginal cost of adding to a machine exceeds the
    # cost of opening a new machine.
    # New machine cost: technician_cost
    # Adding load to a machine near shift: marginal cost ~ 1 + overtime_rate (if beyond shift)
    # So if technician_cost < 1 + overtime_rate, it's cheaper to open new machines.
    # But we also need to consider the load itself.
    
    # Strategy: Group zones together, then partition into technicians.
    # Use a greedy approach: sort zones by some priority, then assign to bins.
    
    # Compute a risk-adjusted load for each zone
    # Use orness to blend between modal and worst-case
    # orness close to 1 means more pessimistic (use worst-case)
    zone_load = {}
    for z in zone_tasks:
        modal = zone_modal[z]
        worst = zone_worst[z]
        # Blend: orness=0 -> modal, orness=1 -> worst
        # Use a convex combination
        zone_load[z] = (1 - orness) * modal + orness * worst

    # Sort zones by load descending (largest first for better bin packing)
    sorted_zones = sorted(zone_tasks.keys(), key=lambda z: zone_load[z], reverse=True)

    # Determine number of technicians to open
    # Estimate total load
    total_load = sum(zone_load.values())
    # Cost of k technicians: k * technician_cost + total_load + overtime penalties
    # Optimal k is roughly total_load / shift, but need to account for overtime
    # Try a few k values and pick the best
    k_min = max(1, int(math.ceil(total_load / shift)))
    k_max = min(n, k_min + 5)  # Try a few more

    best_cost = float('inf')
    best_machines = None

    for k in range(k_min, k_max + 1):
        # Greedy bin packing: assign zones to k bins
        # Sort zones by load descending
        bins = [[] for _ in range(k)]
        bin_load = [0.0] * k
        
        for z in sorted_zones:
            # Assign to bin with minimum load (best-fit decreasing)
            min_idx = min(range(k), key=lambda i: bin_load[i])
            bins[min_idx].append(z)
            bin_load[min_idx] += zone_load[z]
        
        # Compute cost
        cost = 0.0
        for i in range(k):
            L = bin_load[i]
            cost += technician_cost + L + overtime_rate * max(0, L - shift)
        
        if cost < best_cost:
            best_cost = cost
            best_machines = bins

    # Convert zone-level assignment to task-level
    machines = []
    for z_list in best_machines:
        task_list = []
        for z in z_list:
            task_list.extend(zone_tasks[z])
        if task_list:  # Only add non-empty technicians
            machines.append(task_list)

    # Ensure all tasks are covered
    # (The above should cover all tasks since we iterate over all zones)
    # But let's verify and fix if needed
    assigned = set()
    for m in machines:
        for t in m:
            assigned.add(t)
    
    # If some tasks are missing (shouldn't happen), add them to the last machine
    if len(assigned) < n:
        missing = [j for j in range(n) if j not in assigned]
        if machines:
            machines[-1].extend(missing)
        else:
            machines.append(missing)

    return machines
