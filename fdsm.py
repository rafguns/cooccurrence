import logging
import math
import networkx as nx
import sys
import random

from networkx import bipartite
from operator import itemgetter

logger = logging.getLogger(__name__)


def _progressbar(it, prefix="", size=60):
    """Show progress bar

    Source: http://code.activestate.com/recipes/576986-progress-bar-for-console-programs-as-iterator/

    """
    count = len(it)

    def _show(_i):
        x = int(size * _i / count)
        sys.stdout.write(
            "%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x),
                                  _i, count))
        sys.stdout.flush()

    _show(0)
    for i, item in enumerate(it, start=1):
        yield item
        _show(i)
    sys.stdout.write("\n")
    sys.stdout.flush()


def edge_swap(G, nodes, nswap=1, max_tries=100):
    """
    Randomly swap edges while maintaining degrees and bipartivity

    If we have top nodes `u` and `v` connected to bottom nodes `x` and `y`:

        u -- x
        v -- y

    we swap the edges to obtain:

        u -- y
        v -- x

    This retains the bipartivity and node degrees. It is not checked whether
    G is actually bipartite.

    """
    swap_count, num_tries = 0, 0
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
            G.add_edges_from([(u, y), (v, x)])
            G.remove_edges_from([(u, x), (v, y)])
            swap_count += 1
        if num_tries > max_tries:
            logger.warning("Maximum number of tries exceeded")
            break
        num_tries += 1

    return G


def random_bipartite_graph_model(G, nodes, nsample=10, **kwargs):
    assert nx.is_bipartite(G)

    for i in _progressbar(range(nsample), "Generating random graphs"):
        G = edge_swap(G, nodes, **kwargs)
        yield G


def obs_mean_stdev(G, nodes, nsample=10, **kwargs):
    projection = bipartite.weighted_projected_graph(G, nodes)
    cooccs = projection.edges()

    # sum_n is the sum of the number of co-occurrences over all random graphs
    # sum_n2 is the square sum.
    # These can be used to determine the mean and standard deviation.
    all_n = {}
    sum_n = {}
    sum_n2 = {}

    # Add the current network as the first sampled observation.
    # XXX Maybe we should NOT do this?
    for x, y in cooccs:
        n = len(set(G[x]) & set(G[y]))
        all_n["{} - {}".format(x, y)] = [n]
        sum_n[(x, y)] = float(n)
        sum_n2[(x, y)] = float(n ** 2)
    for R in random_bipartite_graph_model(G, nodes, nsample, **kwargs):
        for x, y in cooccs:
            n = len(set(R[x]) & set(R[y]))
            all_n["{} - {}".format(x, y)].append(n)
            sum_n[(x, y)] += n
            sum_n2[(x, y)] += n ** 2

    import json
    with open("very-raw.json", "wb") as fh:
        json.dump(all_n, fh)
    for (x, y), total in sum_n.iteritems():
        obs = projection.edge[x][y]['weight']
        mean = total / nsample
        # Running standard deviation
        # http://stackoverflow.com/questions/1174984
        try:
            stdev = math.sqrt((sum_n2[(x, y)] / nsample) - (mean * mean))
        except ValueError:
            # This is typically due to negative numbers because of rounding
            # errors
            logger.warning("Stdev = sqrt(%.4f)" % ((sum_n2[(x, y)] / nsample) -
                                                   (mean * mean)))
            stdev = 0.0
        yield (x, y), obs, mean, stdev


def z_scores(G, nodes, **kwargs):
    z = {}
    for (x, y), obs, mean, stdev in obs_mean_stdev(G, nodes, **kwargs):
        difference = obs - mean
        try:
            z[(x, y)] = difference / stdev
        except ZeroDivisionError:
            logger.warning("Ignoring pair (%s, %s) with difference %.4f "
                           "and stdev %.4f" % (x, y, difference, stdev))
    return z


def cooccurrences(G, nodes, min_z=3.29, **kwargs):
    z = z_scores(G, nodes, **kwargs)
    for k, v in sorted(z.items(), key=itemgetter(1), reverse=True):
        if v > min_z:
            yield k, v
