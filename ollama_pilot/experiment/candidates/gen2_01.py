def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']

    # Create a list of tasks sorted by their volatility (high to low)
    sorted_tasks = sorted(range(n), key=lambda j: volatility[j], reverse=True)

    # Initialize a dictionary to store tasks per technician
    technicians = {}

    # Iterate over sorted tasks and assign them to technicians
    for j in sorted_tasks:
        # Find the technician with the least loaded zone
        min_zone_load = float('inf')
        min_zone = None
        for z in set(zone):
            if z not in technicians:
                continue
            zone_load = sum(proc[j] for j in technicians[z])
            if zone_load < min_zone_load:
                min_zone_load = zone_load
                min_zone = z

        # If no technician has this zone, create a new one
        if min_zone is None:
            min_zone = max(technicians.keys()) + 1 if technicians else 0
            technicians[min_zone] = []

        # Assign the task to the chosen technician
        technicians[min_zone].append(j)

    # Convert the dictionary to a list of lists
    machines = list(technicians.values())

    return machines
