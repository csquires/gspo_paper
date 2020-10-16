"""
Go through the graphs output by some algorithm and calculate:
- skeleton SHDs
- skeleton false positives/true positives
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from create_mags_and_samples import get_true_mags, get_alg_results_folder, get_alg_times
import multiprocessing
from tqdm import tqdm
import os
import numpy as np
import causaldag as cd

MULTITHREAD = False


def evaluate(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, nsamples: int, alg: str, est_mags, skeletons=None, **kwargs):
    true_mags = get_true_mags(ngraphs, nnodes, nlatent, exp_nbrs)
    alg_results_folder = get_alg_results_folder(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
    os.makedirs(alg_results_folder, exist_ok=True)

    # === SHD
    if skeletons is None:
        shd_skeletons = [true_mag.shd_skeleton(est_mag) for true_mag, est_mag in zip(true_mags, est_mags)]
    else:
        shd_skeletons = [true_mag.shd_skeleton(est_skel) for true_mag, est_skel in zip(true_mags, skeletons)]
    shd_skeletons = np.array(shd_skeletons)
    shd_skeletons_filename = os.path.join(alg_results_folder, 'shd_skeleton.npy')
    np.save(shd_skeletons_filename, shd_skeletons)

    # === TPR
    num_positives_list = [len(true_mag.skeleton) for true_mag in true_mags]
    if skeletons is None:
        true_positive_rates = [
            len(true_mag.skeleton & est_mag.skeleton)/num_positives if num_positives != 0 else 1
            for true_mag, est_mag, num_positives in zip(true_mags, est_mags, num_positives_list)
        ]
    else:
        true_positive_rates = [
            len(true_mag.skeleton & est_skel.edges) / num_positives if num_positives != 0 else 1
            for true_mag, est_skel, num_positives in zip(true_mags, skeletons, num_positives_list)
        ]
    true_positive_rates = np.array(true_positive_rates)
    true_positive_rates_filename = os.path.join(alg_results_folder, 'tprs_skeleton.npy')
    np.save(true_positive_rates_filename, true_positive_rates)

    # === FPR
    num_negatives_list = [nnodes*(nnodes-1)/2 - len(true_mag.skeleton) for true_mag in true_mags]
    if skeletons is None:
        false_positive_rates = [
            len(est_mag.skeleton - true_mag.skeleton) / num_negatives if num_negatives != 0 else 0
            for true_mag, est_mag, num_negatives in zip(true_mags, est_mags, num_negatives_list)
        ]
    else:
        false_positive_rates = [
            len(est_skel.edges - true_mag.skeleton) / num_negatives if num_negatives != 0 else 0
            for true_mag, est_skel, num_negatives in zip(true_mags, skeletons, num_negatives_list)
        ]
    false_positive_rates = np.array(false_positive_rates)
    false_positive_rates_filename = os.path.join(alg_results_folder, 'fprs_skeleton.npy')
    np.save(false_positive_rates_filename, false_positive_rates)

    # === PRECISION
    if skeletons is None:
        num_est_positives_list = [len(est_mag.skeleton) for est_mag in est_mags]
        precisions = np.array([
            len(true_mag.skeleton & est_mag.skeleton) / num_est_positives if num_est_positives != 0 else 1
            for true_mag, est_mag, num_est_positives in zip(true_mags, est_mags, num_est_positives_list)
        ])
    else:
        num_est_positives_list = [len(est_skel.edges) for est_skel in skeletons]
        precisions = np.array([
            len(true_mag.skeleton & est_skel.edges)/num_est_positives if num_est_positives != 0 else 1
            for true_mag, est_skel, num_est_positives in zip(true_mags, skeletons, num_est_positives_list)
        ])
    precisions_filename = os.path.join(alg_results_folder, 'precisions.npy')
    np.save(precisions_filename, precisions)

    # === CORRECT MEC
    same_mec = [true_mag.markov_equivalent(est_mag) for true_mag, est_mag in zip(true_mags, est_mags)]
    same_mec = np.array(same_mec)
    same_mec_filename = os.path.join(alg_results_folder, 'same_mec.npy')
    np.save(same_mec_filename, same_mec)


def get_shd_skeletons(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, nsamples: int, alg: str, **kwargs):
    alg_results_folder = get_alg_results_folder(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
    shd_skeletons_filename = os.path.join(alg_results_folder, 'shd_skeleton.npy')
    return np.load(shd_skeletons_filename)


def get_fprs_skeletons(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, nsamples: int, alg: str, **kwargs):
    alg_results_folder = get_alg_results_folder(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
    fprs_filename = os.path.join(alg_results_folder, 'fprs_skeleton.npy')
    return np.load(fprs_filename)


def get_tprs_skeletons(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, nsamples: int, alg: str, **kwargs):
    alg_results_folder = get_alg_results_folder(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
    tprs_filename = os.path.join(alg_results_folder, 'tprs_skeleton.npy')
    return np.load(tprs_filename)


def get_same_mec(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, nsamples: int, alg: str, **kwargs):
    alg_results_folder = get_alg_results_folder(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
    same_mec_filename = os.path.join(alg_results_folder, 'same_mec.npy')
    return np.load(same_mec_filename)


def get_precision_skeletons(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, nsamples: int, alg: str, **kwargs):
    alg_results_folder = get_alg_results_folder(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
    precisions_filename = os.path.join(alg_results_folder, 'precisions.npy')
    return np.load(precisions_filename)


