import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from create_mags_and_samples import get_mag_samples, get_alg_estimate_filename, get_alg_time_filename
import argparse
import multiprocessing
from tqdm import tqdm
import os
import numpy as np
import causaldag as cd
from evaluate import evaluate
from gspo.algorithm import gspo
from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_suffstat, gauss_ci_test, msep_test
import time
from numpy.linalg import matrix_rank
import random

OVERWRITE = False
GSPO_NAME = 'gspo'
MULTITHREAD = False


def mag2sparsity(mag):
    return mag.num_edges, mag.num_bidirected


def test_alg():
    nnodes = 6
    nlatents = 2
    all_nodes = set(range(nnodes + nlatents))
    obs_nodes = set(range(nnodes))
    exp_nbrs = 2
    ngraphs = 100000

    initial = 'empty'
    depth = float('inf')
    nruns = 1
    lmc_update = "construction"
    strict = False
    max_iters = float('inf')

    # np.random.seed(98713)
    # random.seed(98713)
    seeds = np.random.randint(1,2**32-1, ngraphs)
    # seeds = [3399445488]  # for p=10, K=5, s=2 counterexample
    # seeds = [3066111257]  # for p=6, K=2, s=2 counterexample
    # seeds = [3863021417]  # for p=6, K=2, s=2 counterexample
    # seeds = [3333646883]  # for p=6, K=2, s=2 counterexample
    # seeds = [2105899552]  # for p=6, K=2, s=2 counterexample

    for i, seed in enumerate(seeds):
        print(i, seed)
        np.random.seed(seed)
        random.seed(seed)

        true_dag = cd.rand.directed_erdos(nnodes+nlatents, exp_nbrs/(nnodes+nlatents-1), 1)
        # latents = set(random.sample(all_nodes, nlatents))
        latents = set(range(nlatents))
        true_mag = true_dag.marginal_mag(latents)
        # print(true_mag)
        try:
            true_mag._check_ancestral()
        except Exception:
            continue
        # print(true_mag._check_ancestral())
        true_mag.to_maximal()
        # print('true mag', true_mag)
        # time.sleep(1)

        if true_mag.num_edges > 0:
            ci_tester = MemoizedCI_Tester(msep_test, true_mag)
            np.random.seed(seed)
            random.seed(seed)
            est_mag = gspo(
                true_mag.nodes,
                ci_tester,
                max_iters=max_iters,
                verbose=False,
                initial_imap=initial,
                strict = strict,
                depth=depth,
                nruns=nruns,
                make_minimal=lmc_update,
                # true_mag=true_mag
            )
            # print(est_mag.is_maximal())
            if est_mag.markov_equivalent(true_mag):
                with open("sparsity_tried.txt", "a") as sf:
                    # print(est_mag)
                    print(str(mag2sparsity(true_mag)))
                    sf.write(str(mag2sparsity(true_mag)))
                    sf.write("\n")
            else:
                print('true', true_mag)
                print('true is maximal', true_mag.is_maximal())
                print('est', est_mag)
                print('est is maximal', est_mag.is_maximal())
                print('est is imap', est_mag.is_imap(true_mag))
                print('true num edge', true_mag.num_edges)
                print('est num edges', est_mag.num_edges)
                print('skeleton shd', true_mag.shd_skeleton(est_mag))
                print('true vstructures', true_mag.vstructures())
                print('est vstructures', est_mag.vstructures())
                print("NOT MEC")
                mags = est_mag.get_all_mec()
                print("MEC size", len(mags))
                #exit()


if __name__ == '__main__':
    test_alg()
    # from utils.util import poset2mag_stable, mag2poset, is_legitimate
    # from causaldag.utils.ci_tests import MemoizedCI_Tester, msep_test

    # m = cd.AncestralGraph(directed={(6, 7), (5, 6), (3, 6), (2, 5)}, bidirected={(3, 7), (3, 4)})
    # m2 = cd.AncestralGraph(directed={(3, 7), (3, 4), (6, 7), (5, 2)}, bidirected={(5, 6), (4, 7), (3, 6)})
    #
    # mags = m2.get_all_mec()
    # are_minimal = [mag.is_minimal_imap(m) for mag in mags]
    # print(all(are_minimal))

    # m = cd.AncestralGraph(directed={(6, 7), (5, 7), (3, 6)}, bidirected={(4, 5), (2, 3), (2, 5), (3, 4), (2, 4), (3, 5)})
    # m2 = cd.AncestralGraph(directed={(2, 7), (5, 4), (3, 4), (3, 6)}, bidirected={(6, 7), (5, 7), (2, 3), (3, 7), (2, 5), (2, 4), (3, 5)})
    # print(m2)
    # print(m2.is_maximal(new=True, verbose=True))
    # print(m2.is_maximal(new=False))
    # print(m2.legitimate_mark_changes(verbose=True))
    # mags = m2.get_all_mec()
    # are_minimal = [mag.is_minimal_imap(m) for mag in mags]
    # print('all minimal', all(are_minimal))
    #
    # ci_tester = MemoizedCI_Tester(msep_test, m)
    # for mag in mags:
    #     m_ = poset2mag_stable(mag2poset(mag), ci_tester)
    #     print('maximal', mag.is_maximal())
    #     print(m_ == mag)
    #
    # for mag in mags:
    #     lmcs = [(i, j) for i, j in mag._directed | mag._bidirected | set(map(reversed, mag._bidirected)) if is_legitimate(mag, i, j)]
    #     print(lmcs)
