import numpy as np
import networkx as nx
import torch
import torch.nn.init as init


#cython
import graph
import utils

cdef double initialization_stddev = 0.01  # 权重初始化的方差

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def truncated_normal(size, requires_grad=True, dtype=torch.float32) -> torch.Tensor:
    """

    :param size: tensor size
    :param requires_grad: tensor "requires_grad" parameter
    :param dtype: tensor "dtype" parameter
    :return: An initialized tensor with truncated normal data.

    * Range: [-2*initialization_stddev, 2*initialization_stddev]
    * Mean: 0
    * Standard Deviation: initialization_stddev
    """

    t = torch.ones(size, requires_grad=requires_grad, dtype=dtype)

    return init.trunc_normal_(
        t,
        std=initialization_stddev,
        a=(-2.0) * initialization_stddev,
        b=2.0 * initialization_stddev
    )

class Functional:
    def __init__(self):
        self.utils = utils.py_Utils()



    def GenNetwork(self, g):  #networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), A, B)

    def argMax(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos

    def Max(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best

    def HXA(self, g, method):
        # 'HDA', 'HBA', 'HPRA', ''
        sol = []
        G = g.copy()
        while (nx.number_of_edges(G) > 0):
            if method == 'HDA':
                dc = nx.degree_centrality(G)
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)
            node = keys[maxTag]
            sol.append(int(node))
            G.remove_node(node)
        solution = sol + list(set(g.nodes()) ^ set(sol))
        solutions = [int(i) for i in solution]
        Robustness = self.utils.getRobustness(self.GenNetwork(g), solutions)
        return Robustness, sol
