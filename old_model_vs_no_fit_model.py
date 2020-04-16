from problem import Problem

N, H, Kmax, xmax, ymax = 6, 12, 7, 10, 10
n_trials = 3
first_seed = 1
n_instances = 3

results_no_fit = []
results_old = []

P = Problem(N, H, Kmax, xmax, ymax)

for seed in range(first_seed, first_seed + n_instances):
    
    P.create_data(seed=seed)
    no_fit_time, old_time = 0, 0
    for _ in range(n_trials):
        # no fit model
        P.create_problem(use_no_fit=True)
        P.solve_cplex()
        no_fit_time += P.problem.solutionTime

        # old model
        P.create_problem(use_no_fit=False)
        P.solve_cplex()
        old_time += P.problem.solutionTime
    
    results_no_fit.append(no_fit_time / n_trials)
    results_old.append(old_time / n_trials)


with open(f'old_model_vs_no_fit_model_seed{first_seed}_{first_seed+n_instances-1}.txt', 'w') as f:
    f.write('results no-fit model\n')
    for i, el in enumerate(results_no_fit, first_seed):
        f.write(f'time with seed {i} = {el}\n')

    f.write('results old model\n')
    for i, el in enumerate(results_old, first_seed):
        f.write(f'time with seed {i} = {el}\n')