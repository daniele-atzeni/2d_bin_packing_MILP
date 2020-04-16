from problem import Problem

'''
This script can be very slow, depending on the LP problems that are create randomly.
I don't know why it breaks after 4 or 5 configs
'''

seed = 1
n_trials = 3
results = {}

cuts_list = ['on', 'off', 'root', 'ifmove', 'forceOn']
presolve_list = ['on', 'off', 'more', 'file']
strong_list = [0, 5]
heur_list = ['on', 'off']

N, H, Kmax, xmax, ymax = 5, 12, 6, 10, 10
P = Problem(N, H, Kmax, xmax, ymax)
P.create_data(seed=seed)
P.create_problem()

configs = []
for cuts in cuts_list:
    for presolve in presolve_list:
        for strong in strong_list:
            for heur in heur_list:
                configs.append((cuts, presolve, strong, heur))

for i, config in enumerate(configs):
    cuts, presolve, strong, heur = config
    print(f'config n.{i}, params:', cuts, presolve, strong, heur)
    tot_time = 0
    keepFiles, mip, msg = 0, 1, 0
    for _ in range(n_trials):
        P.solve_cbc(keepFiles, mip, msg, cuts, presolve, strong, heur, maxSeconds=600)
        tot_time += P.problem.solutionTime
    results[(cuts, presolve, strong, heur)] = tot_time / n_trials

    with open(f'cbc_grid_search_seed{seed}.txt', 'a') as f:
        f.write(f'cuts = {cuts}, presolve = {presolve}, strong = {strong}, heur = {heur}\n')
        f.write(f'time = {tot_time / n_trials}\n')
