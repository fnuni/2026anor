def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    orness = params['ornoess']
    alpha_grid = params['alpha_grid']
    train_spread = params['train_spread']

    # Initialize the list of machines
    machines = [[] for _ in range(n)]

    # Sort tasks by their volatility in descending order
    task_indices = sorted(range(n), key=lambda j: volatility[j], reverse=True)

    # Assign tasks to machines based on their volatility
    for j in task_indices:
        min_cost = float('inf')
        best_machine = None
        for i, machine in enumerate(machines):
            L = sum(proc[j] for j in machine)
            cost = technician_cost + L + overtime_rate * max(0, L - shift)
            if cost < min_cost:
                min_cost = cost
                best_machine = i
        machines[best_machine].append(j)

    return machines
