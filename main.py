from problem import Problem

# PuLP apis code at https://github.com/coin-or/pulp/tree/master/pulp/apis
# use options parameter to insert any command line option

N, H, Kmax, xmax, ymax = 6, 15, 5, 10, 10

P = Problem(N, H, Kmax, xmax, ymax)

# PuLP does not support the creation of a problem from file
# Set the seed if you want to reproduce the experiment
P.create_data(seed=None)

# if filename is None, it doesn't save the model in an lp file
P.create_problem(filename='problem.lp', equality=False)

# if filename is None, it shows the figure without saving it
P.show_original_polygons(filename='problem_init.png')

# ------------------------------------------------------ #
# ----------------solve with cbc------------------------ #
# ------------------------------------------------------ #

# Possible cbc parameters at https://projects.coin-or.org/CoinBinary/export/1059/OptimizationSuite/trunk/Installer/files/doc/cbcCommandLine.pdf
# Possible cuts parameters:
# cuts on (off) :  activates (deactivates) all cuts at once.
# Clique, Lift, Mixed, TwoMirCuts, Knapsack, Flow, ProbingCuts, Residual + 
# Off : never try this cut
# Root : cuts applied only at root node;
# IfMove : cuts will be used of they succeed on improving the dual bound;
# ForceOn : forces the use of the cut generator at every node.
# For Probing cut generator other more aggressive options are available: 
# forceOn, forceOnGlobal, forceOnStrong, forceOnButStrong and strongRoot.

keepFiles = 0
mip = 1
msg = 1
cuts = True
presolve = None
dual = None
strong = 2
options = []
fracGap = None
maxSeconds = None
threads = None
mip_start = True

P.solve_cbc(keepFiles, mip, msg, cuts, presolve, dual, strong, options, fracGap, maxSeconds, threads, mip_start)

# filename as in show_original_polygons
# if plt_points, separating points between polygons are shown
P.show_solution(filename='problem_sol_cbc.png', plt_points=False)

# ------------------------------------------------------ #
# ----------------solve with cplex---------------------- #
# ------------------------------------------------------ #

# Possible cplex parameters at https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.0/ilog.odms.ide.help/OPL_Studio/refoplrun/topics/oplrun_syntax_commandline.html

keepFiles = 0
mip = 1 
msg = 1 
options = []
timelimit = None 
mip_start = False

P.solve_cplex(keepFiles, mip, msg, options, timelimit, mip_start)

P.show_solution(filename='problem_sol_cplex.png', plt_points=False)