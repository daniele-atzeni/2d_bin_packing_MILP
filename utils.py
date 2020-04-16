import numpy as np
from scipy.spatial import ConvexHull

def compute_coefficients(polygon):
    '''
    Function that takes a polygon (2d numpy array)
    and return the coefficients of its edges as list
    NB: the points of the polygon must be in order!
    :param polygons: list of polygons, each polygon is np.array, shape = (n_vertices, 2)
    :param H: float, value of the roll
    :return: a, b, c, dictionaries (n, v): coefficient of the straight line connecting vertex v with the next
    '''
    a_list = []
    b_list = []
    c_list = []

    n_edges = polygon.shape[0]
    for v in range(n_edges):
        next_v = 0 if v == n_edges - 1 else v + 1
        x1, y1 = polygon[v]
        x2, y2 = polygon[next_v]
        # straight line formula on the plane
        a = y2 - y1
        b = -x2 + x1
        c = -x1 * (y2 - y1) + y1 * (x2 - x1)

        # check directional vector
        # special case, triangles
        if n_edges == 3:
            if v == 0:
                x_pol, y_pol = polygon[2]
            elif v == 1:
                x_pol, y_pol = polygon[0]
            else:
                x_pol, y_pol = polygon[1]
        else:
            if v == 0 or v == n_edges - 1: 
                # I can consider vertex number 2
                x_pol, y_pol = polygon[2]
            else:   
                # I can consider vertex 0
                x_pol, y_pol = polygon[0]
            
        if a * x_pol + b * y_pol + c < 0:
            a = -a
            b = -b
            c = -c
        
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

    return a_list, b_list, c_list


def Minkowski_sum(A, B):
    # A, B numpy 2d arrays (n_verteces, 2)
    new_vertices = []
    for v1 in A:
        for v2 in B:
            new_vertices.append(v1 + v2)
    
    polygon = ConvexHull(np.array(new_vertices))
    return polygon.points[polygon.vertices]


def compute_no_fit_polygons(polygons):
    
    N = len(polygons)
    no_fit_polygons = {}

    for i in range(N):
        for j in range(i+1, N):
            A = polygons[i]
            B = polygons[j]
            no_fit_polygons[(i, j)] = Minkowski_sum(A, -B)
    
    return no_fit_polygons
