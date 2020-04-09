from problem import Problem

'''
This script can be very slow, depending on the LP problems that are create randomly.
'''

N, H, Kmax, xmax, ymax = 7, 15, 5, 10, 10

results_old = []
results_new = []

i = 0
while i < 5:

    P = Problem(N, H, Kmax, xmax, ymax)

    P.create_data()

    if P.is_improvable:

        P.show_original_polygons(filename='problem_init.png')

        P.create_problem()

        P.solve_cplex()

        results_old.append(P.problem.solutionTime)

        P.solve_with_M_constr()

        results_new.append(P.problem.solutionTime)

        P.show_solution(filename='problem_sol_cplex.png', plt_points=False)

        i += 1


with open('old_M_vs_new_M.txt', 'w') as f:
    
    f.write('results with old M\n')
    for i, el in enumerate(results_old):
        f.write(f'time at istance {i} = {el}\n')

    f.write('results with new M\n')
    for i, el in enumerate(results_new):
        f.write(f'time at istance {i} = {el}\n')
