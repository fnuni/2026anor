def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']

    # Initialize technician loads and task assignments
    technician_loads = [0] * (n + 1)  # technician indices start from 1
    task_assignments = [[] for _ in range(n)]

    # Sort tasks by their volatility to handle volatile tasks first
    task_indices = list(range(n))
    task_indices.sort(key=lambda j: volatility[j], reverse=True)

    for task in task_indices:
        best_technician = None
        best_cost = float('inf')

        for tech in range(1, n + 1):
            if technician_loads[tech] + proc[task] <= shift:
                load = technician_loads[tech] + proc[task]
                cost = technician_cost + load + overtime_rate * max(0, load - shift)
                if cost < best_cost:
                    best_technician = tech
                    best_cost = cost

        if best_technician is not None:
            technician_loads[best_technician] += proc[task]
            task_assignments[task] = best_technician

    # Filter out empty technicians
    machines = [tasks for tasks in task_assignments if tasks]

    return machines
