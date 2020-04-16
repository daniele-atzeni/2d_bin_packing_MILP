# 2d Bin Packing Mixed Integer Linear Problem with PuLP

In order to run the scripts, you have to add cbc.exe and/or cplex.exe in your repository.
You can also modify the path variable in solve_cbc and solve_cplex methods in problem.py, 
setting it to the path where solvers are saved.

PuLP LPProblem class does not support the creation of a model from file. For this reason
in order to reproduce your experiment you can use the seed parameter when you call
the create_data method.

- main.py:
  it creates an instance of the problem and solves it with cbc and cplex. It also creates problem.lp, with the problem instance, problem_init.png,
  that shows the polygons of the instance, and problem_sol_cbc.png and problem_sol_cplex.png,
  that show the solution of the problem.
 
 - cbc_vs_cplex.py:
  it creates instances of the problems,
  solves them with cbc and cplex, computes the time needed and saves everything in a text file.
 
 - cbc_grid_search.py:
  It creates an instance of the problem and solves it with different combination of 
  cbc parameters, computes the time needed for each combination and saves everything
  in a text file

 - cbc_grid_search_read_sol.py:
  It shows the results obtained by cbc_grid_search
  ordered by time

 - no_M_opt_vs_M_opt.py:
  It creates instances of the problem and solves them using default cplex and a different
  model which use a different value of M, computes the time needed for each instance 
  and saves everything in a text file.

 - old_model_vs_no_fit_model.py
  It creates instances of the problem and solves them
  using the two models explained in final_report.pdf,
  computes the time needed for each instance and
  saves everything in a text file.