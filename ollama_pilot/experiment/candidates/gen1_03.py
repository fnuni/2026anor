def solve(instance, params):
    n = instance['n']
    proc = instance['proc']
    volatility = instance['volatility']
    zone = instance['zone']
    shift = instance['shift']
    overtime_rate = instance['overtime_rate']
    technician_cost = instance['technician_cost']
    orness = params['orness']
    alpha_grid = params['alpha_grid']
    train_spread = params['train_spread']

    # Heuristic: Greedy allocation based on task duration and zone
    tasks = list(range(n))
    tasks.sort(key=lambda j: (proc[j], -volatility[j], zone[j]))

    machines = []
    while tasks:
        machine = []
        remaining_duration = shift
        assigned_tasks = set()
        
        for task in tasks[:]:
            if task in assigned_tasks:
                continue
            duration = proc[task] + train_spread * volatility[task]
            if duration <= remaining_duration:
                machine.append(task)
                remaining_duration -= duration
                assigned_tasks.add(task)
                tasks.remove(task)
            else:
                break
        
        if machine:
            machines.append(machine)

    return machines
