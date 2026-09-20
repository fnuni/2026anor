import math

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

    # Initialize the machine schedule and task assignments
    machines = [[] for _ in range(10)]  # Assuming a maximum of 10 machines for simplicity
    machine_loads = [0.0] * 10

    # Assign tasks to machines based on their volatility and zone
    for j in range(n):
        best_machine = 0
        best_cost = math.inf
        for i in range(10):
            L = machine_loads[i] + proc[j]
            cost = technician_cost + L + overtime_rate * max(0, L - shift)
            if cost < best_cost:
                best_cost = cost
                best_machine = i
        machines[best_machine].append(j)
        machine_loads[best_machine] += proc[j]

    return machines
