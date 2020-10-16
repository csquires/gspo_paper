import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from create_mags_and_samples import get_mag_samples, get_alg_estimate_filename, get_alg_time_filename
import argparse
import multiprocessing
from tqdm import tqdm
import os
from R_algs.fci_wrapper import fci
import numpy as np
import causaldag as cd
from evaluate import evaluate
import time

OVERWRITE = False
MULTITHREAD = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngraphs', type=int)
    parser.add_argument('--nnodes', type=int)
    parser.add_argument('--nlatent', type=int)
    parser.add_argument('--exp_nbrs', type=float)
    parser.add_argument('--nsamples', type=int)

    parser.add_argument('--alpha', type=float)
    parser.add_argument('--fci_plus', type=str)

    args = parser.parse_args()
    ngraphs, nnodes, nlatent, exp_nbrs, nsamples = args.ngraphs, args.nnodes, args.nlatent, args.exp_nbrs, args.nsamples
    alpha, fci_plus = args.alpha, args.fci_plus

    FCI_NAME = 'fci_plus' if fci_plus == 'True' else 'fci'

    def run_fci(graph_num):
        results_filename = get_alg_estimate_filename(
            ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples, FCI_NAME, alpha=alpha
        )
        skeleton_results_filename = get_alg_estimate_filename(
            ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples, FCI_NAME, skeleton=True, alpha=alpha
        )
        time_filename = get_alg_time_filename(
            ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples, FCI_NAME, alpha=alpha
        )
        if not os.path.exists(results_filename) or OVERWRITE:
            print(results_filename)
            samples = get_mag_samples(ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples)

            start_time = time.time()
            est_mag, est_skeleton = fci(samples, alpha, graph_num, fci_plus=fci_plus)
            time_used = time.time() - start_time

            os.makedirs(os.path.dirname(results_filename), exist_ok=True)
            np.save(skeleton_results_filename, est_skeleton.to_amat())
            np.save(results_filename, est_mag.to_amat())
            np.save(time_filename, time_used)
            return est_mag, est_skeleton, time_used
        else:
            time_used = np.load(time_filename)
            skeleton = cd.UndirectedGraph.from_amat(np.load(skeleton_results_filename))
            try:
                mag = cd.AncestralGraph.from_amat(np.load(results_filename))
            except cd.NeighborError as e:
                mag = cd.AncestralGraph(nodes=set(range(nnodes)))  # TODO: get to the bottom of why pag2mag produces something w/ an error
            return mag, skeleton, time_used

    if MULTITHREAD:
        with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
            graph_nums = list(range(ngraphs))
            results = list(tqdm(pool.imap(run_fci, graph_nums), total=ngraphs))
            est_mags, skeletons, times = zip(*results)
    else:
        results = list(tqdm((run_fci(n) for n in range(ngraphs)), total=ngraphs))
        est_mags, skeletons, times = zip(*results)

    evaluate(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, FCI_NAME, est_mags, skeletons=skeletons, alpha=alpha)
