import math

def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    orness = params['ornois']
    alpha_grid = params['alpha_grid']
    train_spread = params['train_spread']

    # Calculate the expected duration of each task considering volatility
    expected_duration = [proc[j] * (1 + volatility[j]) for j in range(n)]

    # Sort tasks by their expected duration in descending order
    sorted_tasks = sorted(range(n), key=lambda j: expected_duration[j], reverse=True)

    # Initialize the list of technicians
    machines = []

    # Assign tasks to technicians
    for j in sorted_tasks:
        assigned = False
        for m in machines:
            if len(m) < 10:  # Limit the number of tasks per technician to 10
                m.append(j)
                assigned = True
                break
        if not assigned:
            machines.append([j])

    return machines
