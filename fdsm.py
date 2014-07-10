import math
import networkx as nx
import random

from networkx import bipartite
from operator import itemgetter
from tangle.util import progressbar, log


def edge_swap(G, nodes, nswap=1, max_tries=100):
    """
    Randomly swap edges while maintaining degrees and bipartivity

    If we have top nodes `u` and `v` connected to bottom nodes `x` and `y`:

        u -- x
        v -- y

    we swap the edges to obtain:

        u -- y
        v -- x

    This retains the bipartivity and node degrees.
    """
    assert nx.is_bipartite(G)

    swap_count, n = 0, 0
    nodelist = list(nodes)

    while swap_count < nswap:
        u = random.choice(nodelist)
        v = random.choice(nodelist)
        if u == v:  # Same top node
            continue
        x = random.choice(list(G[u]))
        y = random.choice(list(G[v]))
        if x == y:  # Same bottom node
            continue
        if not G.has_edge(u, y) and not G.has_edge(v, x):
            G.add_edge(u, y)
            G.add_edge(v, x)
            G.remove_edge(u, x)
            G.remove_edge(v, y)
            swap_count += 1
        if n > max_tries:
            log.logger.warning("Maximum number of tries exceeded")
            break
        n += 1

    return G


def random_bipartite_graph_model(G, nodes, nsample=10, **kwargs):
    for i in progressbar(range(nsample), "Generating random graphs"):
        G = edge_swap(G, nodes, **kwargs)
        yield G


def z_scores(G, nodes, **kwargs):
    projection = bipartite.weighted_projected_graph(G, nodes)
    cooccs = projection.edges()

    # sum_n is the sum of the number of co-occurrences over all random graphs
    # sum_n2 is the square sum.
    # These can be used to determine the mean and standard deviation.
    sum_n = {}
    sum_n2 = {}

    # Add the current network as the first sampled observation.
    # XXX Maybe we should NOT do this?
    for x, y in cooccs:
        n = len(set(G[x]) & set(G[y]))
        sum_n[(x, y)] = float(n)
        sum_n2[(x, y)] = float(n ** 2)
    for i, R in enumerate(random_bipartite_graph_model(G, nodes, **kwargs)):
        for x, y in cooccs:
            n = len(set(R[x]) & set(R[y]))
            sum_n[(x, y)] += n
            sum_n2[(x, y)] += n ** 2

    z = {}
    n = i + 1
    for (x, y), total in sum_n.iteritems():
        mean = total / n
        # http://stackoverflow.com/questions/1174984
        try:
            stdev = math.sqrt((sum_n2[(x, y)] / n) - (mean * mean))
        except ValueError:
            # This is typically due to negative numbers because of rounding
            # errors
            log.logger.warning("Stdev = sqrt(%.4f)" % ((sum_n2[(x, y)] / n) -
                                                       (mean * mean)))
            stdev = 0.0
        difference = projection.edge[x][y]['weight'] - mean
        try:
            z[(x, y)] = difference / stdev
        except ZeroDivisionError:
            log.logger.warning("Ignoring pair (%s, %s) with difference %.4f "
                               "and stdev %.4f" % (x, y, difference, stdev))
    return z


def cooccurrences(G, nodes, min_z=3.29, **kwargs):
    z = z_scores(G, nodes, **kwargs)
    for k, v in sorted(z.items(), key=itemgetter(1), reverse=True):
        if v > min_z:
            yield k, v