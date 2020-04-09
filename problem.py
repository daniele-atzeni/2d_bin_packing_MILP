import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import ConvexHull

import pulp
from pulp import LpProblem, LpMinimize, LpVariable, value, LpStatus, lpSum

from utils import compute_coefficients, compute_M

class Problem:

    def __init__(self, N, H, Kmax, xmax, ymax):
        self.N = N
        self.H = H
        self.Kmax = Kmax
        self.xmax = xmax
        self.ymax = ymax

        self.polygons = None
        self.worst_val = None
        self.a = None
        self.b = None
        self.c = None
        self.M = None

        self.is_improvable = None

        self.problem = None

        self.filename = None            # LP file associated to the problem


    def create_data(self, seed=None):

        self.polygons = []
        # set the seed of pseudo-random number generator
        np.random.seed(seed)

        while len(self.polygons) < self.N:
            # generate Kmax points on the plane
            xs = np.random.uniform(0, self.xmax, self.Kmax, )
            ys = np.random.uniform(0, self.ymax, self.Kmax)
            points = np.stack((xs, ys), axis=1)
            # move points to match the origin with minimums of points' coordinates
            mins = np.min(points, axis=0)
            points = points - mins
            # compute height of the polygon
            height = np.max(points, axis=0)[1]
            # check if height is less than H. If yes, compute convex hull
            if height <= self.H:
                polygon = ConvexHull(points)
                self.polygons.append(polygon.points[polygon.vertices])
        
        self.a, self.b, self.c = compute_coefficients(self.polygons, self.H)
        self.M, self.is_improvable = compute_M(self.polygons, self.H, self.a, self.b, self.c)


    def create_problem(self, filename=None, equality=False):

        self.problem = LpProblem('polygons_packing', LpMinimize)

        self.w = LpVariable('w', 0)

        self.tx = LpVariable.dicts('tx', range(self.N), 0)
        self.ty = LpVariable.dicts('ty', range(self.N), 0)
        z_idx = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                K_i = self.polygons[i].shape[0]
                K_j = self.polygons[j].shape[0]
                for v in range(K_i + K_j):
                    z_idx.append((i, j, v))
        self.z = LpVariable.dicts('z', z_idx, 0, 1, 'Integer')

        # objective
        self.problem.setObjective(self.w)

        # w constraints
        widths = [np.max(pol[:, 0]) - np.min(pol[:, 0]) for pol in self.polygons]
        for i in range(self.N):
            self.problem += self.w - self.tx[i] >= widths[i], f"w_constraint_{i}"

        # height constraints
        heights = [np.max(pol[:, 1]) - np.min(pol[:, 1]) for pol in self.polygons]
        for i in range(self.N):
            self.problem += self.ty[i] <= self.H - heights[i], f"height_constraint_{i}"
        
        # non overlapping constraints
        cnt = 0
        for i in range(self.N):
            for j in range(i+1, self.N):
                K_i = self.polygons[i].shape[0]
                K_j = self.polygons[j].shape[0]
                for v1 in range(K_i):
                    for v2 in range(K_j):
                        # 1
                        self.problem += self.a[(i, v1)] * self.tx[j] - self.a[(i, v1)] * self.tx[i] + \
                            self.b[(i, v1)] * self.ty[j] - self.b[(i, v1)] * self.ty[i] - self.M * self.z[(i, j, v1)] <= \
                                - self.c[(i, v1)] - self.a[(i, v1)] * self.polygons[j][v2][0] - self.b[(i, v1)] * self.polygons[j][v2][1], f"non_overlapping_constraint_{cnt}"
                        cnt += 1
                        # 2
                        self.problem += self.a[(j, v2)] * self.tx[i] - self.a[(j, v2)] * self.tx[j] + \
                            self.b[(j, v2)] * self.ty[i] - self.b[(j, v2)] * self.ty[j] - self.M * self.z[(i, j, K_i + v2)]  <= \
                                - self.c[(j, v2)] - self.a[(j, v2)] * self.polygons[i][v1][0] - self.b[(j, v2)] * self.polygons[i][v1][1], f"non_overlapping_constraint_{cnt}"
                        cnt += 1
        # 3
        for i in range(self.N):
            for j in range(i+1, self.N):
                K_i = self.polygons[i].shape[0]
                K_j = self.polygons[j].shape[0]
                if not equality:
                    self.problem += sum([self.z[(i, j, v)] for v in range(K_i + K_j)]) <= K_i + K_j - 1, f"non_overlapping_constraint_{cnt}"
                else:
                    self.problem += sum([self.z[(i, j, v)] for v in range(K_i + K_j)]) == K_i + K_j - 1, f"non_overlapping_constraint_{cnt}"
                cnt += 1
        
        if filename is not None:
            self.filename = filename
            self.problem.writeLP(filename)


    def solve_cbc(self, 
            keepFiles = 0,
            mip = 1,
            msg = 1,
            cuts = None,
            presolve = None,
            dual = None,
            strong = None,
            options = [],
            fracGap = None,
            maxSeconds = None,
            threads = None,
            mip_start = False):
        
        path = os.path.join(os.getcwd(), 'cbc.exe')

        solver = pulp.solvers.COIN_CMD(path, keepFiles, mip, msg, cuts, presolve, dual, strong, options, fracGap, 
                                        maxSeconds, threads, mip_start)

        if self.problem is None:
            raise ValueError('You must create the model first')
                
        self.problem.solve(solver=solver)

        print(f'time = {self.problem.solutionTime}')
        print(f'status = {LpStatus[self.problem.status]}')
        print(f'objective = {value(self.problem.objective)}')


    def solve_cplex(self,
            keepFiles=0, 
            mip=1, 
            msg=1, 
            options=[],
            timelimit = None, 
            mip_start=False):
        
        path = os.path.join(os.getcwd(), 'cplex.exe')

        solver = pulp.solvers.CPLEX_CMD(path, keepFiles, mip, msg, options, timelimit, mip_start)

        if self.problem is None:
            raise ValueError('You must create the model first')

        self.problem.solve(solver=solver)

        print(f'time = {self.problem.solutionTime}')
        print(f'status = {LpStatus[self.problem.status]}')
        print(f'objective = {value(self.problem.objective)}')
    

    def solve_with_M_constr(self,
            keepFiles=0, 
            mip=1, 
            msg=1, 
            options=[],
            timelimit = None, 
            mip_start=False):
        if self.is_improvable:
            path = os.path.join(os.getcwd(), 'cplex.exe')

            solver = pulp.solvers.CPLEX_CMD(path, keepFiles, mip, msg=None, options=options, timelimit=5, mip_start=mip_start)

            if self.problem is None:
                raise ValueError('You must create the model first')
            
            # first solve
            self.problem.solve(solver=solver)
            print(f'First solve done, new worst case = {value(self.problem.objective)}')

            # compute new M
            self.M, _ = compute_M(self.polygons, self.H, self.a, self.b, self.c, value(self.problem.objective))

            # add constraints
            cnt = 0
            for i in range(self.N):
                for j in range(i+1, self.N):
                    K_i = self.polygons[i].shape[0]
                    K_j = self.polygons[j].shape[0]
                    for v1 in range(K_i):
                        for v2 in range(K_j):
                            # 1
                            self.problem += self.a[(i, v1)] * self.tx[j] - self.a[(i, v1)] * self.tx[i] + \
                                self.b[(i, v1)] * self.ty[j] - self.b[(i, v1)] * self.ty[i] - self.M * self.z[(i, j, v1)] <= \
                                    - self.c[(i, v1)] - self.a[(i, v1)] * self.polygons[j][v2][0] - self.b[(i, v1)] * self.polygons[j][v2][1], f"new_constraint_{cnt}"
                            cnt += 1
                            # 2
                            self.problem += self.a[(j, v2)] * self.tx[i] - self.a[(j, v2)] * self.tx[j] + \
                                self.b[(j, v2)] * self.ty[i] - self.b[(j, v2)] * self.ty[j] - self.M * self.z[(i, j, K_i + v2)]  <= \
                                    - self.c[(j, v2)] - self.a[(j, v2)] * self.polygons[i][v1][0] - self.b[(j, v2)] * self.polygons[i][v1][1], f"new_constraint_{cnt}"
                            cnt += 1
            
            # last solve
            # need to reinitialize solver with timelimiti
            solver = pulp.solvers.CPLEX_CMD(path, keepFiles, mip, msg, options, timelimit, mip_start)
            self.problem.solve(solver=solver)
            
            print(f'time = {self.problem.solutionTime}')
            print(f'status = {LpStatus[self.problem.status]}')
            print(f'objective = {value(self.problem.objective)}')
        
        else:
            self.solve_cplex(keepFiles, mip, msg, options, timelimit, mip_start)


    def show_original_polygons(self, filename=None):
        
        plt.figure()
        max_width = 0   # need this variable to plot polygons next to each other
        colors = cm.get_cmap('rainbow')

        for i, polygon in enumerate(self.polygons):
            for j, v in enumerate(polygon):
                next_v = polygon[0] if j == polygon.shape[0] - 1 else polygon[j + 1]
                plt.plot((v[0] + max_width, next_v[0] + max_width), (v[1], next_v[1]), c=colors(float(i) / self.N))
            max_width += np.max(polygon, axis=0)[0]

        self.worst_val = max_width
        
        # roll plot
        plt.plot((0, 0), (0, self.H), c='k')
        plt.plot((0, self.worst_val), (self.H, self.H), c='k')
        plt.plot((self.worst_val, self.worst_val), (self.H, 0), c='k')
        plt.plot((self.worst_val, 0), (0, 0), c='k')

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

        plt.close()
    
    def show_solution(self, filename=None, plt_points=False):

        tx = [var.value() for var in self.problem.variables() if var.name[:2] == 'tx']
        ty = [var.value() for var in self.problem.variables() if var.name[:2] == 'ty']
        opt_val = value(self.problem.objective)

        plt.figure()
        colors = cm.get_cmap('rainbow')

        for i, polygon in enumerate(self.polygons):
            for j, v in enumerate(polygon):
                next_v = polygon[0] if j == polygon.shape[0] - 1 else polygon[j + 1]
                plt.plot((v[0] + tx[i], next_v[0] + tx[i]), (v[1] + ty[i], next_v[1] + ty[i]), c=colors(float(i) / self.N))
            
            if plt_points:
            # plot points whose edge (counterclokwise from the point) separate polygons
                for j in range(i+1, self.N):
                    K_i = self.polygons[i].shape[0]
                    K_j = self.polygons[j].shape[0]
                    plotted = False
                    if not plotted:
                        for v in range(K_i):
                            for var in self.problem.variables():
                                if var.name == f'z_({i},_{j},_{v})' and var.value() == 0:
                                    plt.plot(self.polygons[i][v][0] + tx[i], self.polygons[i][v][1] + ty[i], 'o', c=colors(float(j) / self.N))
                                    plotted = True
                        for v in range(K_j):
                            for var in self.problem.variables():
                                if var.name == f'z_({i},_{j},_{K_i+v})' and var.value() == 0:
                                    plt.plot(self.polygons[j][v][0] + tx[j], self.polygons[j][v][1] + ty[j], 'o', c=colors(float(i) / self.N))
                                    plotted=True

        # roll plot
        plt.plot((0, 0), (0, self.H), c='k')
        plt.plot((0, opt_val), (self.H, self.H), c='k')
        plt.plot((opt_val, opt_val), (self.H, 0), c='k')
        plt.plot((opt_val, 0), (0, 0), c='k')

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

        plt.close()
