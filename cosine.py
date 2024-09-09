import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform


def cooccurrences(G, nodes, **kwargs):
    M = nx.bipartite.biadjacency_matrix(G, nodes, **kwargs)
    coocc = 1.0 - squareform(pdist(M, "cosine"))
    coocc = np.triu(coocc)
    x, y = np.asarray(coocc).nonzero()
    coocc = [((nodes[i], nodes[j]), coocc[i, j]) for i, j in zip(x, y) if i != j]
    return coocc
