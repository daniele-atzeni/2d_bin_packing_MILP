from problem import Problem

'''
This script can be very slow, depending on the LP problems that are create randomly.
'''

H, Kmax, xmax, ymax = 15, 5, 10, 10

results = {}

for N in range(3, 8):

    P = Problem(N, H, Kmax, xmax, ymax)

    P.create_data()

    P.create_problem()

    P.solve_cbc(maxSeconds=700)

    results[(N, 'cbc')] = P.problem.solutionTime

    P.solve_cplex(timelimit=700)

    results[(N, 'cplex')] = P.problem.solutionTime

with open('confronto_solver.txt', 'w') as f:
    
    for key, val in results.items():
        N, solver = key
        f.write(f'N = {N}, solver = {solver}     time = {val}\n')