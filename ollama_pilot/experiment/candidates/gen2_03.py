def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    
    # Create a list of tasks sorted by their volatility
    tasks = list(range(n))
    tasks.sort(key=lambda j: volatility[j], reverse=True)
    
    # Initialize the list of machines
    machines = []
    
    # Initialize a dictionary to keep track of the total load on each machine
    machine_loads = {}
    
    # Initialize a dictionary to keep track of the zones on each machine
    machine_zones = {}
    
    # Iterate over the sorted list of tasks
    for j in tasks:
        # Find the machine with the least load and the least number of zones
        min_load = float('inf')
        min_zones = float('inf')
        min_machine = -1
        
        for i in range(len(machines)):
            if i not in machine_loads or machine_loads[i] < min_load or (machine_loads[i] == min_load and len(machine_zones[i]) < min_zones):
                min_load = machine_loads.get(i, 0)
                min_zones = len(machine_zones.get(i, set()))
                min_machine = i
        
        # Assign the task to the selected machine
        if min_machine == -1:
            min_machine = len(machines)
            machines.append([])
            machine_loads[min_machine] = 0
            machine_zones[min_machine] = set()
        
        machines[min_machine].append(j)
        machine_loads[min_machine] += proc[j]
        machine_zones[min_machine].add(zone[j])
    
    return machines
