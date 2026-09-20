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

    # Create a list of tasks sorted by their volatility
    tasks = sorted(range(n), key=lambda j: volatility[j], reverse=True)

    # Initialize technicians
    technicians = [[] for _ in range(n)]

    # Assign tasks to technicians
    for j in tasks:
        assigned = False
        for i in range(len(technicians)):
            if not assigned:
                current_load = sum(proc[k] for k in technicians[i])
                if current_load + proc[j] <= shift + train_spread:
                    technicians[i].append(j)
                    assigned = True
        if not assigned:
            technicians.append([j])

    # Filter out empty technicians
    technicians = [tech for tech in technicians if tech]

    return technicians
