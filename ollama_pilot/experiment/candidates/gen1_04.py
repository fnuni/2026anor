def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Create a list of tasks sorted by their modal duration
    tasks = list(range(n))
    tasks.sort(key=lambda j: proc[j])
    
    # Initialize the list of machines (technicians)
    machines = []
    
    # Function to calculate the cost of a machine with given load
    def machine_cost(load):
        return technician_cost + load + overtime_rate * max(0, load - shift)
    
    # Assign tasks to machines
    for j in tasks:
        assigned = False
        for i, machine in enumerate(machines):
            current_load = sum(proc[k] for k in machine)
            new_load = current_load + proc[j]
            if new_load <= shift:
                machine.append(j)
                assigned = True
                break
        if not assigned:
            machines.append([j])
    
    return machines
