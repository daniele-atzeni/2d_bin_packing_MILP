import numpy as np

def compute_coefficients(polygons, H):
    '''
    NB: the points of the polygon must be in order!
    :param polygons: list of polygons, each polygon is np.array, shape = (n_vertices, 2)
    :param H: float, value of the roll
    :return: a, b, c, dictionaries (n, v): coefficient of the straight line connecting vertex v with the next
    '''
    N = len(polygons)
    widths = [np.max(pol[:, 0]) - np.min(pol[:, 0]) for pol in polygons]
    a = {}
    b = {}
    c = {}

    for n in range(N):
        n_edges = polygons[n].shape[0]
        for v in range(n_edges):
            next_v = 0 if v == n_edges - 1 else v + 1
            x1, y1 = polygons[n][v]
            x2, y2 = polygons[n][next_v]
            # straight line formula on the plane
            a[(n, v)] = y2 - y1
            b[(n, v)] = -x2 + x1
            c[(n, v)] = -x1 * (y2 - y1) + y1 * (x2 - x1)

            # check directional vector
            # special case, triangles
            if n_edges == 3:
                if v == 0:
                    x_pol, y_pol = polygons[n][2]
                elif v == 1:
                    x_pol, y_pol = polygons[n][0]
                else:
                    x_pol, y_pol = polygons[n][1]
            else:
                if v == 0 or v == n_edges - 1: 
                    # I can consider vertex number 2
                    x_pol, y_pol = polygons[n][2]
                else:   
                    # I can consider vertex 0
                    x_pol, y_pol = polygons[n][0]
                
            if a[(n, v)] * x_pol + b[(n, v)] * y_pol + c[(n, v)] < 0:
                a[(n, v)] = -a[(n, v)]
                b[(n, v)] = -b[(n, v)]
                c[(n, v)] = -c[(n, v)]

    # computation of M
    W = sum(widths)
    possibleM = []
    for n in range(N):
        n_edges = polygons[n].shape[0]
        for v in range(n_edges):
            m1 = c[(n, v)]
            m2 = b[(n, v)] * H + c[(n, v)]
            m3 = a[(n, v)] * W + c[(n, v)]
            m4 = a[(n, v)] * W + b[(n, v)] * H + c[(n, v)]
            possibleM.extend([m1, m2, m3, m4])
    M = max(possibleM)

    return a, b, c, M