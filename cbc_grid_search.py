from problem import Problem

'''
This script can be very slow, depending on the LP problems that are create randomly.
'''

cuts_list = [False, True]
presolve_list = [False, True]
dual_list = [False, True]
strong_list = [None, 2, 5, 9]

N, H, Kmax, xmax, ymax = 6, 15, 5, 10, 10
P = Problem(N, H, Kmax, xmax, ymax)
P.create_data()
P.create_problem()

results = {}

for cuts in cuts_list:
    for presolve in presolve_list:
        for dual in dual_list:
            for strong in strong_list:
                    
                P.solve_cbc(cuts, presolve, dual, strong, maxSeconds=700)

                results[(cuts, presolve, dual, strong)] = P.problem.solutionTime


with open('cbc_grid_search.txt', 'w') as f:
    
    for key, val in sorted(list(results.items()), key = lambda x: x[1]):
        cuts, presolve, dual, strong, mip_start = key
        f.write(f'cuts = {cuts}, presolve = {presolve}, dual = {dual}, strong = {strong}\n')
        f.write(f'time = {val}\n')