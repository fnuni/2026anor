import math

def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Initialize the list of machines
    machines = []
    
    # Sort tasks by their modal duration in descending order
    sorted_tasks = sorted(range(n), key=lambda j: proc[j], reverse=True)
    
    # Assign tasks to machines
    for j in sorted_tasks:
        assigned = False
        for machine in machines:
            L = sum(proc[i] for i in machine) + proc[j]
            if L <= shift:
                machine.append(j)
                assigned = True
                break
        if not assigned:
            machines.append([j])
    
    # Calculate the cost
    total_cost = technician_cost * len(machines)
    for machine in machines:
        L = sum(proc[i] for i in machine)
        total_cost += L + overtime_rate * max(0, L - shift)
    
    return machines
