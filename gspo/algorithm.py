import causaldag as cd
from causaldag.structure_learning import perm2dag, min_degree_alg_amat, threshold_ug, gsp
import random
import itertools as itr
from utils.util import poset2mag_stable, mag2poset


def update_after_deletion(i, j, closure_matrix):

    # 1) find R = what's no longer an ancestor of j
    # 2) remove r in R from descendant k of j if:
    #       none of ancestors of k that aren't descendants of j have r as an ancestor
    queue = [j]
    while queue:
        pass
    pass


def apply_lmc(imap, i, j):
    # if {i, j} == {9, 3}:
    #     print(imap)
    #     print(imap._check_ancestral())
    if imap.has_directed(i, j):
        imap.remove_directed(i, j)
        imap.add_bidirected(i, j)
    else:
        imap.remove_bidirected(i, j)
        imap.add_directed(i, j)


def get_lmc_altered_edges(imap: cd.AncestralGraph, i, j, ci_tester, closure_matrix, true_mag=None):
    """
    Given an IMAP and a legitimate mark change applied to it, test which edges can be removed after the legitimate
    mark change.

    TODO: THIS CAN BE OPTIMIZED A LOT BY NOT MAKING A COPY OF THE IMAP AND UPDATING ANCESTORS USING DYNAMIC TRANSITIVE
    CLOSURE ALGORITHMS
    """
    #
    #
    # get linear extension
    # try deleting any edges whose ancestors have changed, in order of the linear extension

    # if i->j ====> i<->j
    # other_children = imap.children_of(i) - {i}
    # changed_ancestors = {
    #     k for k in imap.nodes
    #     if j in ancestor_dict[k] and not any(child in ancestor_dict[k] for child in other_children)
    # }

    imap_copy = imap.copy()   # TODO: BOTTLENECK
    apply_lmc(imap_copy, i, j)

    desc_j = imap.descendants_of(j) | {j}

    # removed_edges = set()
    for k, l in list(imap.directed) + list(imap.bidirected):
        if k in desc_j or l in desc_j:
            c = imap_copy.ancestors_of({k, l}) - {k, l}  # TODO: BOTTLENECK
            # c = ancestor_dict[k] | ancestor_dict[l] - {k, l}
            if ci_tester.is_ci(k, l, c):
                imap_copy.remove_edge(k, l)
                # removed_edges.add((k, l))
    removed_dir = imap.directed - imap_copy.directed
    if imap.has_directed(i, j): removed_dir = removed_dir - {(i, j)}
    removed_bidir = bidirected_frozenset(imap) - bidirected_frozenset(imap_copy)
    if imap.has_bidirected(i, j): removed_bidir = removed_bidir - {frozenset({i, j})}

    return removed_dir, removed_bidir


def get_lmc_altered_edges2(imap: cd.AncestralGraph, i, j, ci_tester, _, true_mag=None):
    desc_j = imap.descendants_of(j) | {j}

    imap_copy = imap.copy()
    apply_lmc(imap_copy, i, j)

    ancestor_dict = imap.ancestor_dict()
    for k, l in imap.directed | imap.bidirected:
        if k in desc_j or l in desc_j:
            c = (ancestor_dict[k] | ancestor_dict[l]) - {k, l}
            if ci_tester.is_ci(k, l, c):
                imap_copy.remove_edge(k, l)

    # for k, l in set(itr.combinations(imap.nodes, 2)):
    #     if not imap.has_any_edge(k, l) and (k in desc_j or l in desc_j):
    #         c = ancestor_dict[k] | ancestor_dict[l] - {k, l}
    #         if not ci_tester.is_ci(k, l, c):
    #             if k in ancestor_dict[l]:
    #                 imap_copy._add_directed(k, l)
    #             elif l in ancestor_dict[k]:
    #                 imap_copy._add_directed(l, k)
    #             else:
    #                 imap_copy._add_bidirected(k, l)

    second_imap_copy = imap_copy.copy()
    ancestor_dict = imap_copy.ancestor_dict()
    for k, l in imap_copy.directed | imap_copy.bidirected:
        if k in desc_j or l in desc_j:
            c = (ancestor_dict[k] | ancestor_dict[l]) - {k, l}
            if ci_tester.is_ci(k, l, c):
                second_imap_copy.remove_edge(k, l)

    for k, l in set(itr.combinations(imap_copy.nodes, 2)):
        if not imap_copy.has_any_edge(k, l) and (k in desc_j or l in desc_j):
            c = (ancestor_dict[k] | ancestor_dict[l]) - {k, l}
            if not ci_tester.is_ci(k, l, c):
                if k in ancestor_dict[l]:
                    second_imap_copy._add_directed(k, l)
                elif l in ancestor_dict[k]:
                    second_imap_copy._add_directed(l, k)
                else:
                    second_imap_copy._add_bidirected(k, l)

    second_imap_copy.to_maximal(new=True)
    # is_maximal = second_imap_copy.is_maximal()
    # if not is_maximal:
    #     new_mag = second_imap_copy.copy()
    #     new_mag.to_maximal()
    #     print(new_mag)
    #     print(second_imap_copy)
    # print('maximal:', is_maximal)

    # second_imap_copy._check_ancestral()
    if true_mag is not None and not second_imap_copy.is_minimal_imap(true_mag):
        raise Exception

    removed_dir = imap.directed - second_imap_copy.directed
    if imap.has_directed(i, j): removed_dir = removed_dir - {(i, j)}
    removed_bidir = bidirected_frozenset(imap) - bidirected_frozenset(second_imap_copy)
    if imap.has_bidirected(i, j): removed_bidir = removed_bidir - {frozenset({i, j})}

    added_dir = second_imap_copy.directed - imap.directed - {(i, j)}
    added_bidir = bidirected_frozenset(second_imap_copy) - bidirected_frozenset(imap) - {frozenset({i, j})}
    if added_dir or added_bidir:
        return None, None

    return removed_dir, removed_bidir


def get_lmc_altered_edges3(imap: cd.AncestralGraph, i, j, ci_tester, _, true_mag=None):
    # print('previous imap:', imap)
    # print(m2)
    # print(imap == m2)
    # print(imap.legitimate_mark_changes(strict=True))
    # print(m2.legitimate_mark_changes())
    # print(imap.discriminating_paths(verbose=True))
    # print(m2.discriminating_paths())

    imap_copy = imap.copy()
    apply_lmc(imap_copy, i, j)
    poset = mag2poset(imap_copy)
    imap_copy = poset2mag_stable(poset, ci_tester)

    # print('lmc:', i, j)
    # print('poset:', poset)
    # print('imap:', imap_copy)

    if true_mag is not None:
        if not imap_copy.is_minimal_imap(true_mag, certify=True):
            raise Exception

    removed_dir = imap.directed - imap_copy.directed
    if imap.has_directed(i, j): removed_dir = removed_dir - {(i, j)}
    removed_bidir = bidirected_frozenset(imap) - bidirected_frozenset(imap_copy)
    if imap.has_bidirected(i, j): removed_bidir = removed_bidir - {frozenset({i, j})}

    added_dir = imap_copy.directed - imap.directed - {(i, j)}
    added_bidir = bidirected_frozenset(imap_copy) - bidirected_frozenset(imap) - {frozenset({i, j})}

    return removed_dir, removed_bidir


def bidirected_frozenset(m):
    return frozenset({frozenset({*e}) for e in m._bidirected})


def gspo(
        nodes: set,
        ci_tester,
        depth=4,
        initial_imap='permutation',
        strict=True,
        verbose=False,
        max_iters=float('inf'),
        true_mag=None,
        nruns=5,
        make_minimal='deletion'
):
    """
    Estimate a MAG using the Greedy Sparsest Poset algorithm.

    Parameters
    ----------
    nodes:
        Labels of nodes in the graph.
    ci_tester:
        A conditional independence tester, which has a method is_ci taking two sets A and B, and a conditioning set C,
        and returns True/False.
    depth:
        Maximum depth in depth-first search. Use None for infinite search depth.
    initial_imap:
        String indicating how to obtain the initial IMAP. Must be "permutation" or "empty".
    strict:
        If True, check discriminating paths condition for legitimate mark changes.
    verbose:
        If True, print information about algorithm progress.
    max_iters:
        Maximum number of depth-first search steps without score improvement before stopping.
    true_mag:
        for debugging purposes
    """
    if initial_imap == 'permutation':
        ug = threshold_ug(nodes, ci_tester)
        amat = ug.to_amat()
        perms = [min_degree_alg_amat(amat) for _ in range(nruns)]
        dags = [perm2dag(perm, ci_tester) for perm in perms]
        starting_imaps = [cd.AncestralGraph(dag.nodes, directed=dag.arcs) for dag in dags]
    elif initial_imap == 'empty':
        edges = {(i, j) for i, j in itr.combinations(nodes, 2) if not ci_tester.is_ci(i, j)}
        starting_imaps = [cd.AncestralGraph(nodes, bidirected=edges) for _ in range(nruns)]
    elif initial_imap == 'gsp':
        ug = threshold_ug(nodes, ci_tester)
        amat = ug.to_amat()
        perms = [min_degree_alg_amat(amat) for _ in range(nruns)]
        dags = [gsp(nodes, ci_tester, nruns=1, initial_permutations=[perm]) for perm in perms]
        starting_imaps = [cd.AncestralGraph(dag.nodes, directed=dag.arcs) for dag in dags]

    # print('starting:', starting_imaps)

    get_alt_edges = get_lmc_altered_edges if make_minimal == 'deletion' else get_lmc_altered_edges2

    sparsest_imap = None
    for r in range(nruns):
        current_imap = starting_imaps[r]
        if verbose: print(f"Starting run {r} with {current_imap.num_edges} edges")

        # TODO: BOTTLENECK
        current_lmcs_directed, current_lmcs_bidirected = current_imap.legitimate_mark_changes(strict=strict)
        current_lmcs = current_lmcs_directed | current_lmcs_bidirected

        # TODO: BOTTLENECK
        lmcs2altered_edges = [
            (lmc, get_alt_edges(current_imap, *lmc, ci_tester, None, true_mag=true_mag))
            for lmc in current_lmcs
        ]
        lmcs2altered_edges = [(lmc, (a, b)) for lmc, (a, b) in lmcs2altered_edges if a is not None]
        lmcs2edge_delta = [
            (lmc, len(removed_dir) + len(removed_bidir))
            for lmc, (removed_dir, removed_bidir) in lmcs2altered_edges
        ]

        mag2number = dict()
        graph_counter = 0
        trace = []
        iters_since_improvement = 0
        while True:
            if iters_since_improvement > max_iters:
                break

            mag_hash = (frozenset(current_imap._directed), bidirected_frozenset(current_imap))
            if mag_hash not in mag2number:
                mag2number[mag_hash] = graph_counter
            graph_num = mag2number[mag_hash]
            if verbose: print(f"Number of visited MAGs: {len(mag2number)}. Exploring MAG #{graph_num} with {current_imap.num_edges} edges.")
            max_delta = max([delta for lmc, delta in lmcs2edge_delta], default=0)

            sparser_exists = max_delta > 0
            keep_searching_mec = len(trace) != depth and len(lmcs2altered_edges) > 0
            if true_mag:
                if current_imap.markov_equivalent(true_mag):
                    return current_imap
                print(f'#{graph_num}', current_imap)
                is_min_imap, certificate = current_imap.is_minimal_imap(true_mag, certify=True)
                print(is_min_imap, certificate)
                if not is_min_imap:
                    raise Exception

            if sparser_exists:
                trace = []

                lmc_ix = random.choice([ix for ix, (lmc, delta) in enumerate(lmcs2edge_delta) if delta == max_delta])
                (i, j), (removed_dir, removed_bidir) = lmcs2altered_edges.pop(lmc_ix)
                apply_lmc(current_imap, i, j)
                current_imap.remove_edges(removed_dir | removed_bidir)

                if verbose: print(f"Starting over at a sparser IMAP with {current_imap.num_edges} edges")
            elif keep_searching_mec:
                if verbose: print(f"{'='*len(trace)}Continuing search through the MEC at {current_imap.num_edges} edges. "
                                  f"Picking from {len(lmcs2altered_edges)} neighbors of #{graph_num}.")
                trace.append((current_imap.copy(), current_lmcs, lmcs2altered_edges, lmcs2edge_delta))
                (i, j), _ = lmcs2altered_edges.pop(0)
                lmcs2edge_delta.pop(0)
                apply_lmc(current_imap, i, j)
            elif len(trace) != 0:  # BACKTRACK IF POSSIBLE
                if verbose: print(f"{'='*len(trace)}Backtracking")
                current_imap, current_lmcs, lmcs2altered_edges, lmcs2edge_delta = trace.pop()
                iters_since_improvement += 1
            else:
                break

            # IF WE MOVED TO A NOVEL IMAP, WE NEED TO UPDATE LMCs
            if sparser_exists or keep_searching_mec:
                graph_counter += 1
                current_lmcs_dir, current_lmcs_bidir = current_imap.legitimate_mark_changes(strict=strict)
                current_lmcs = current_lmcs_dir | current_lmcs_bidir
                lmcs2altered_edges = [
                    (lmc, get_alt_edges(current_imap, *lmc, ci_tester, None, true_mag=true_mag))
                    for lmc in current_lmcs
                ]
                lmcs2altered_edges = [(lmc, (a, b)) for lmc, (a, b) in lmcs2altered_edges if a is not None]
                current_directed, current_bidirected = frozenset(current_imap.directed), bidirected_frozenset(current_imap)

                # === FILTER OUT ALREADY-VISITED IMAPS
                filtered_lmcs2altered_edges = []
                for lmc, (removed_dir, removed_bidir) in lmcs2altered_edges:
                    if current_imap.has_directed(*lmc):
                        new_directed = current_directed - {lmc} - removed_dir
                        new_bidirected = current_bidirected | {frozenset({*lmc})} - {frozenset({*e}) for e in removed_bidir}
                    else:
                        new_directed = current_directed | {lmc} - removed_dir
                        new_bidirected = current_bidirected - {frozenset({*lmc})} - {frozenset({*e}) for e in removed_bidir}

                    if (new_directed, new_bidirected) not in mag2number:
                        filtered_lmcs2altered_edges.append((lmc, (removed_dir, removed_bidir)))
                lmcs2altered_edges = filtered_lmcs2altered_edges

                lmcs2edge_delta = [
                    (lmc, len(removed_dir)+len(removed_bidir))
                    for lmc, (removed_dir, removed_bidir) in lmcs2altered_edges
                ]
        if sparsest_imap is None or sparsest_imap.num_edges > current_imap.num_edges:
            sparsest_imap = current_imap

    return current_imap


if __name__ == '__main__':
    from causaldag.utils.ci_tests import MemoizedCI_Tester, msep_test, gauss_ci_suffstat, gauss_ci_test
    from R_algs.fci_wrapper import fci
    from line_profiler import LineProfiler
    #
    lp = LineProfiler()

    import time
    import numpy as np

    seed = np.random.randint(0, 100000)
    # seed = 95831
    np.random.seed(seed)
    random.seed(seed)

    m2 = cd.AncestralGraph(directed={(3, 4), (0, 4), (1, 4)}, bidirected={(0, 1), (1, 3), (2, 3), (1, 2)})
    m3 = cd.AncestralGraph(directed={(1, 4), (3, 4)}, bidirected={(0, 4), (0, 1), (1, 2), (2, 3), (1, 3)})
    m4 = cd.AncestralGraph(directed={
        (4, 7),
        (9, 1),
        (9, 4),
        (2, 5),
        (0, 3),
        (8, 5),
        (1, 2),
        (1, 5),
        (0, 4),
        (8, 2),
        (9, 3),
        (0, 5),
        (6, 5),
        (2, 7),
        (8, 3),
        (9, 2),
        (6, 1),
        (0, 6),
        (9, 5),
        (3, 4),
        (2, 4),
    }, bidirected={
        (6, 8),
        (1, 8),
        (2, 3),
        (3, 6),
        (0, 9),
        (0, 2),
    })
    m5 = cd.AncestralGraph(directed={
        (4, 7),
        (3, 2),
        (3, 5),
        (9, 0),
        (4, 9),
        (7, 0),
        (4, 5),
        (3, 0),
        (5, 6),
        (6, 8),
        (5, 7),
        (6, 7),
        (3, 8),
        (2, 0),
        (3, 9),
        (4, 3),
        (7, 9),
    }, bidirected={
        (2, 9),
    })

    p = 10
    K = 3
    exp_nbrs = 3
    d = cd.rand.directed_erdos(p+K, exp_nbrs/(p+K-1))
    g = cd.rand.rand_weights(d)
    samples = g.sample(10000)[:, K:]
    suffstat = gauss_ci_suffstat(samples)

    C = g.correlation
    K = np.linalg.inv(C)
    rho = K / np.sqrt(np.diag(K)) / np.sqrt(np.diag(K))[:, None]
    suffstat_oracle = dict(C=C, K=np.linalg.inv(C), n=100, rho=rho)

    gauss_ci_tester = MemoizedCI_Tester(gauss_ci_test, suffstat, alpha=1e-4)
    m = d.marginal_mag(set(range(K)), relabel='default')

    PROFILE = False
    if PROFILE:
        lp.add_function(cd.AncestralGraph.legitimate_mark_changes)
        lp.add_function(get_lmc_altered_edges)
        gspo = lp(gspo)

    print(f"True number of edges: {m.num_edges}")
    time.sleep(1)
    ci_tester = MemoizedCI_Tester(msep_test, m)
    start_time = time.time()
    depth = float('inf')
    max_iters = float('inf')
    # depth = 4
    est_mag = gspo(
        set(range(p)),
        ci_tester,
        verbose=False,
        strict=True,
        max_iters=max_iters,
        initial_imap='permutation',
        depth=depth,
        make_minimal='deletion'
    )

    if PROFILE:
        lp.print_stats()

    print(est_mag.is_imap(m))
    print(est_mag.is_minimal_imap(m, certify=True))
    print(time.time() - start_time)

    print(est_mag.shd_skeleton(m))



    # start = time.time()
    # est_mag_fci = fci(samples, .01)
    # print(time.time() - start)
    # print(est_mag_fci.shd_skeleton(m))

    #   - first, try construction instead (just a different get_lmc_altered_edges function)
    #   - if even construction gives wrong answer, maybe code logic is wrong. otherwise, might find a counterexample.



