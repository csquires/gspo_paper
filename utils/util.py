import itertools as itr

import causaldag as cd

from utils.poset import Poset


def mag2sparsity(mag):
    return mag.num_edges, mag.num_bidirected


def mag2skellsparsity(mag):
    return (mag.num_edges,)


def mag2poset(mag):
    d = cd.DAG(nodes=mag.nodes, arcs=mag.directed)
    p = Poset.from_dag(d)
    return p


def poset2mag(poset: Poset, ci_tester, maximal_completion=True):
    nodes = poset.underlying_dag.nodes
    m = cd.AncestralGraph(nodes=nodes)
    for i, j in itr.combinations(nodes, r=2):
        # if({1,3} == {i,j}):
        #    #import pdb; pdb.set_trace()
        ancestors_i = poset._ancestors[i]
        ancestors_j = poset._ancestors[j]
        S = (ancestors_i | ancestors_j) - {i, j}
        if not ci_tester.is_ci(i, j, S):
            if poset.incomparable(i, j):
                m.add_bidirected(i, j)
            elif i in ancestors_j:
                m.add_directed(i, j)
            else:
                m.add_directed(j, i)
    # print("------------------before---")
    # print(m.directed)
    # print(m.bidirected)
    # print("------------------after---")
    # print(m.directed)
    # print(m.bidirected)
    if (maximal_completion):
        m.to_maximal()
    return m


def poset2mag_stable(poset: Poset, ci_tester):
    """returns max(G_po(G_pi))"""
    m = poset2mag(poset, ci_tester, maximal_completion=False)
    poset_stable = mag2poset(m)
    m_stable = poset2mag(poset_stable, ci_tester, maximal_completion=True)
    return m_stable


def is_legitimate(mag, a, b, check_disc_paths=True, check_parents=True, check_spouses=True):
    """(a,b) is a (bi)directed edge in mag"""
    # 1. There is no directed path from a to b
    if a in mag.ancestors_of(b, exclude_arcs={(a, b)}):
        return False

    # 2. a) If C->A then C->B
    if check_parents:
        for c in mag._parents[a]:
            if c not in mag._parents[b]:
                return False
    else:
        for c in mag._parents[a]:
            if c in mag._neighbors[b] and c not in mag._parents[b]:
                return False

    # b) If D<->A, then D<->B or D<->B
    if check_spouses:
        for d in mag._spouses[a]:
            if d == b:
                continue
            if d not in mag._spouses[b] and d not in mag._parents[b]:
                return False
    else:
        for d in mag._spouses[a]:
            if d == b:
                continue
            if b in mag._parents[d]:
                return False

    # 3. There is disc path for A in which B is the endpoint adjacent to A
    if check_disc_paths:
        disc_paths = mag.discriminating_paths()
        for path in disc_paths:
            a_cand = path[-2]
            b_cand = path[-1]
            if (a_cand == a and b_cand == b):
                return False
    return True


def is_reversal(mag, a, b):
    """(a,b) is a directed edge in mag"""
    # 1. Parents(a) + {a} = Parents(b)
    if not mag._parents[a].union({a}) == mag._parents[b]:
        return False

    # 2. Spouses(a) = Spouses(b)
    if not mag._spouses[a] == mag._spouses[b]:
        return False

    return True
