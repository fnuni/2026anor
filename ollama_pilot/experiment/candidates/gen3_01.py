def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Group tasks by zone for common-mode scaling
    zone_tasks = {z: [] for z in set(zone)}
    for j in range(n):
        zone_tasks[zone[j]].append(j)
    
    # Sort tasks within each zone by their volatility
    for z, tasks in zone_tasks.items():
        zone_tasks[z] = sorted(tasks, key=lambda j: volatility[j], reverse=True)
    
    # Initialize machines list
    machines = []
    
    # Assign tasks to machines
    for z, tasks in zone_tasks.items():
        for j in tasks:
            # Find a machine to assign the task to
            assigned = False
            for m in machines:
                L = sum(proc[k] for k in m)
                if L + proc[j] <= shift:
                    m.append(j)
                    assigned = True
                    break
            if not assigned:
                machines.append([j])
    
    # Add fixed cost for each machine opened
    for m in machines:
        m.append(-technician_cost)
    
    return machines
