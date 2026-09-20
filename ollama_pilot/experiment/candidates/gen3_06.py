def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    zone = instance['zone']
    shift = instance['shift']
    technician_cost = instance['technician_cost']
    overtime_rate = instance['overtime_rate']
    
    # Sort tasks by their volatility in descending order
    tasks = sorted(range(n), key=lambda j: -instance['volatility'][j])
    
    # Initialize the list of technicians
    technicians = []
    
    # Assign tasks to technicians
    for task in tasks:
        assigned = False
        for i in range(len(technicians)):
            technician = technicians[i]
            if sum(proc[j] for j in technician) + proc[task] <= shift:
                technician.append(task)
                assigned = True
                break
        if not assigned:
            technicians.append([task])
    
    # Add the fixed cost of opening each technician
    for i in range(len(technicians)):
        technicians[i] = [technician_cost] + technicians[i]
    
    return technicians
