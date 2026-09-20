def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Initialize technicians and task assignments
    technicians = []
    task_assignment = [-1] * n
    
    # Function to calculate the cost of a technician with load L
    def technician_cost_func(L):
        return technician_cost + L + overtime_rate * max(0, L - shift)
    
    # Sort tasks by their volatility
    tasks_by_volatility = sorted(range(n), key=lambda j: volatility[j])
    
    # Assign tasks to technicians
    for task in tasks_by_volatility:
        assigned = False
        for i in range(len(technicians)):
            if not assigned:
                if task_assignment[task] == -1:
                    technicians[i].append(task)
                    task_assignment[task] = i
                    assigned = True
        if not assigned:
            technicians.append([task])
            task_assignment[task] = len(technicians) - 1
    
    # Remove empty technicians
    technicians = [tech for tech in technicians if tech]
    
    return technicians
