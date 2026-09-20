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

    # Calculate expected durations considering volatility
    expected_duration = [proc[j] * (1 + train_spread * volatility[j]) for j in range(n)]

    # Sort tasks by expected duration in descending order
    sorted_tasks = sorted(range(n), key=lambda j: expected_duration[j], reverse=True)

    # Initialize machines
    machines = [[] for _ in range(n)]

    # Assign tasks to machines
    for j in sorted_tasks:
        assigned = False
        for i in range(n):
            if len(machines[i]) == 0 or (len(machines[i]) > 0 and expected_duration[j] <= machines[i][-1]):
                machines[i].append(j)
                assigned = True
                break
        if not assigned:
            machines[0].append(j)

    # Ensure all machines are open
    machines = [machine for machine in machines if len(machine) > 0]

    return machines
