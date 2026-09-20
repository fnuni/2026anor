import math

def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Calculate capacity reservations based on volatility
    capacity = [1.0 / (1 + 2 * v) for v in volatility]
    
    # Sort tasks by their modal duration
    tasks = sorted(range(n), key=lambda j: proc[j])
    
    # Initialize the list of technicians and their loads
    technicians = []
    current_loads = []
    
    for j in tasks:
        task_volatility = volatility[j]
        task_capacity = capacity[j]
        
        # Find a technician with enough capacity
        for i in range(len(current_loads)):
            if current_loads[i] + task_capacity <= shift:
                current_loads[i] += task_capacity
                if not technicians[i]:
                    technicians.append([])
                technicians[i].append(j)
                break
        else:
            # If no technician has enough capacity, open a new one
            current_loads.append(task_capacity)
            technicians.append([j])
    
    return technicians
