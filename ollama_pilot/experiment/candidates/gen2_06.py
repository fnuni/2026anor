def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Initialize variables
    machines = [[] for _ in range(n)]
    technicians = [0] * n
    zone_counts = [0] * n
    
    # Sort tasks by volatility in descending order
    tasks = sorted(range(n), key=lambda j: volatility[j], reverse=True)
    
    # Assign tasks to technicians
    for j in tasks:
        min_cost = float('inf')
        min_tech = 0
        for i in range(n):
            if technicians[i] < shift:
                cost = technician_cost + technicians[i] + overtime_rate * max(0, technicians[i] - shift)
                if cost < min_cost:
                    min_cost = cost
                    min_tech = i
        machines[min_tech].append(j)
        technicians[min_tech] += proc[j]
        zone_counts[zone[j]] += 1
    
    # Ensure all tasks are assigned
    for j in range(n):
        if not machines[technicians[j]]:
            for i in range(n):
                if technicians[i] < shift:
                    machines[i].append(j)
                    technicians[i] += proc[j]
                    break
    
    return machines
