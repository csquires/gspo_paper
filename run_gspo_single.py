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
from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_suffstat, gauss_ci_test
import time
from numpy.linalg import matrix_rank

OVERWRITE = False
GSPO_NAME = 'gspo'
MULTITHREAD = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngraphs', type=int)
    parser.add_argument('--nnodes', type=int)
    parser.add_argument('--nlatent', type=int)
    parser.add_argument('--exp_nbrs', type=float)
    parser.add_argument('--nsamples', type=int)

    parser.add_argument('--alpha', type=float)
    parser.add_argument('--strict', type=bool)
    parser.add_argument('--initial', type=str)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--max_iters', type=int)
    parser.add_argument('--nruns', type=int)
    parser.add_argument('--lmc_update', type=str)

    args = parser.parse_args()
    ngraphs, nnodes, nlatent, exp_nbrs, nsamples = args.ngraphs, args.nnodes, args.nlatent, args.exp_nbrs, args.nsamples
    alpha, strict, initial, depth, nruns, lmc_update, max_iters = args.alpha, args.strict, args.initial, args.depth, args.nruns, args.lmc_update, args.max_iters

    def run_fci(graph_num):
        results_filename = get_alg_estimate_filename(
            ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples, GSPO_NAME, alpha=alpha, initial=initial, depth=depth, max_iters=max_iters
        )
        time_filename = get_alg_time_filename(
            ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples, GSPO_NAME, alpha=alpha, initial=initial, depth=depth, max_iters=max_iters
        )

        if OVERWRITE or not os.path.exists(results_filename):
            samples = get_mag_samples(ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples)
            start = time.time()
            suffstat = gauss_ci_suffstat(samples)
            ci_tester = MemoizedCI_Tester(gauss_ci_test, suffstat, alpha=alpha)
            est_mag = gspo(set(range(nnodes)), ci_tester, initial_imap=initial, depth=depth, nruns=nruns, max_iters=max_iters, make_minimal=lmc_update)
            time_used = time.time() - start

            os.makedirs(os.path.dirname(results_filename), exist_ok=True)
            np.save(results_filename, est_mag.to_amat())
            np.save(time_filename, time_used)
            return est_mag, time_used
        else:
            time_used = np.load(time_filename)
            return cd.AncestralGraph.from_amat(np.load(results_filename)), time_used

    if MULTITHREAD:
        with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
            graph_nums = list(range(ngraphs))
            results = list(tqdm(pool.imap(run_fci, graph_nums), total=ngraphs))
            est_mags, times = zip(*results)
    else:
        results = list(tqdm((run_fci(n) for n in range(ngraphs)), total=ngraphs))
        est_mags, times = zip(*results)

    evaluate(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, GSPO_NAME, est_mags, alpha=alpha, initial=initial, depth=depth)
