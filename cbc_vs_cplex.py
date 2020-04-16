from problem import Problem

'''
This script can be very slow, depending on the LP problems that are create randomly.
'''

N, H, Kmax, xmax, ymax = 5, 12, 6, 10, 10
n_trials = 3
n_instances = 3
first_seed = 1

results = {}

for seed in range(first_seed, first_seed + n_instances):

    P = Problem(N, H, Kmax, xmax, ymax)

    P.create_data(seed=seed)

    P.create_problem()

    cbc_time, cplex_time = 0, 0

    for _ in range(n_trials):
    
        P.solve_cbc(keepFiles = 0, mip=1, msg=1, cuts='forceOn', presolve='on', strong=5, heur='off')
        cbc_time += P.problem.solutionTime

        P.solve_cplex()
        cplex_time += P.problem.solutionTime

    results[(seed, 'cbc')] = cbc_time / n_trials
    results[(seed, 'cplex')] = cplex_time / n_trials

with open(f'cbc_vs_cplex_seed{first_seed}_{first_seed+n_instances-1}.txt', 'w') as f:
    
    for key, val in results.items():
        seed, solver = key
        f.write(f'seed = {seed}, solver = {solver}     time = {val}\n')