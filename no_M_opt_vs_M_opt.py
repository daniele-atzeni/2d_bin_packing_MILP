from problem import Problem

'''
This script can be very slow.
'''

N, H, Kmax, xmax, ymax = 7, 15, 7, 10, 10
n_instances = 5
n_trials = 3

results_no_opt = []
results_opt = []

P = Problem(N, H, Kmax, xmax, ymax)

for n in range(n_instances):

    P.create_data()
    
    no_opt_time, opt_time = 0, 0

    for _ in range(n_trials):
        # If I call solve with M_opt without
        # re-create the problem, it 
        # keeps adding constraint
        P.create_problem()

        P.solve_cplex(use_M_opt=False)
        no_opt_time += P.problem.solutionTime

        P.solve_cplex(use_M_opt=True)
        opt_time += P.problem.solutionTime
    
    results_no_opt.append(no_opt_time / n_trials)
    results_opt.append(opt_time / n_trials)

with open(f'no_M_opt_vs_M_opt.txt', 'w') as f:
    
    f.write('results without opt M\n')
    for i, el in enumerate(results_no_opt):
        f.write(f'time at instance {i} = {el}\n')

    f.write('results with opt M\n')
    for i, el in enumerate(results_opt):
        f.write(f'time at instance {i} = {el}\n')
