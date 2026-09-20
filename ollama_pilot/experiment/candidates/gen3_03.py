def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Group tasks by zone
    zone_tasks = {}
    for j in range(n):
        if zone[j] not in zone_tasks:
            zone_tasks[zone[j]] = []
        zone_tasks[zone[j]].append(j)
    
    # Sort tasks by volatility
    sorted_tasks = sorted(range(n), key=lambda j: volatility[j], reverse=True)
    
    # Initialize machines
    machines = []
    
    # Assign tasks to machines
    for j in sorted_tasks:
        found_machine = False
        for machine in machines:
            if len(machine) < 10:  # Arbitrary limit to prevent overly large machines
                found_machine = True
                machine.append(j)
                break
        if not found_machine:
            machines.append([j])
    
    # Add a technician cost to each machine
    for machine in machines:
        machine_cost = technician_cost + sum(proc[j] for j in machine)
        if len(machine) > 1:
            machine_cost += overtime_rate * max(0, sum(proc[j] for j in machine) - shift)
        machine.append(machine_cost)
    
    return machines
