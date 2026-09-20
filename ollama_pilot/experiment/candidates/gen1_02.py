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

    # Create a list of tasks sorted by their volatility
    tasks = list(range(n))
    tasks.sort(key=lambda j: volatility[j], reverse=True)

    # Initialize the list of machines
    machines = []

    # Initialize a dictionary to keep track of the load on each machine
    machine_load = {}

    # Iterate over the tasks and assign them to machines
    for j in tasks:
        # Find the machine with the least load
        min_load = float('inf')
        min_machine = None
        for i, load in machine_load.items():
            if load < min_load:
                min_load = load
                min_machine = i

        # If no machine is available, create a new one
        if min_machine is None:
            min_machine = len(machines)
            machines.append([])
            machine_load[min_machine] = 0

        # Assign the task to the machine
        machines[min_machine].append(j)
        machine_load[min_machine] += proc[j]

    return machines
