import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import random
import numpy as np
from config import PROJECT_FOLDER
import causaldag as cd


# === FUNCTIONS DEFINING THE DIRECTORY STRUCTURE
# /data
#   /nnodes=5,nlatent=3,exp_nbrs=2,ngraphs=100
#       /graph0
#           /nsamples=100
#               samples.npy
#               /estimates
#                   /fci
#                       alpha=1.00e-01.npy
#                       ...
#                   /gspo
#           /nsamples=500
#           ...
#       /graph1
#       ...
#   /nnodes=10,nlatent=3,exp_nbrs=2,ngraphs=100
#   ...
# /results
#   /nnodes=5,nlatent=3,exp_nbrs=2,ngraphs=100
#       /nsamples=100
#           /fci
#               /alpha=1.00e-01.npy
#                   shds_skeleton.npy
#               ...
#           /gspo
#           ...
#       /nsamples=500
#       ...
#   /nnodes=10,nlatent=3,exp_nbrs=2,ngraphs=100
#   ...
def get_graphs_string(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float):
    return f"nnodes={nnodes},nlatent={nlatent},exp_nbrs={exp_nbrs},ngraphs={ngraphs}"


def get_graphs_folder(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float):
    return os.path.join(PROJECT_FOLDER, 'data', get_graphs_string(ngraphs, nnodes, nlatent, exp_nbrs))


def get_graph_folder(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, graph_num: int):
    graphs_folder = get_graphs_folder(ngraphs, nnodes, nlatent, exp_nbrs)
    return os.path.join(graphs_folder, f"graph{graph_num}")


def get_samples_folder(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, graph_num: int, nsamples: int):
    graph_folder = get_graph_folder(ngraphs, nnodes, nlatent, exp_nbrs, graph_num)
    return os.path.join(graph_folder, f"nsamples={nsamples}")


def get_alg_estimate_folder(ngraphs, nnodes, nlatent, exp_nbrs: float, graph_num, nsamples, alg):
    samples_folder = get_samples_folder(ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples)
    return os.path.join(samples_folder, "estimates", alg)


def to_str(v):
    if isinstance(v, float):
        return f"{v:.2e}"
    else:
        return str(v)


def dict2str(d):
    keys_and_values = sorted(d.items(), key=lambda item: item[0])
    return ','.join([f"{k}={to_str(v)}" for k, v in keys_and_values])


def get_alg_estimate_filename(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, graph_num: int, nsamples: int, alg: str, skeleton=False, **kwargs):
    results_folder = get_alg_estimate_folder(ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples, alg)
    skeleton_str = '_skeleton' if skeleton else ''
    return os.path.join(results_folder, dict2str(kwargs)+skeleton_str+'.npy')


def get_alg_time_filename(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, graph_num: int, nsamples: int, alg: str, **kwargs):
    results_folder = get_alg_estimate_folder(ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples, alg)
    return os.path.join(results_folder, dict2str(kwargs)+'_time.npy')


def get_graphs_results_folder(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, nsamples: int):
    sample_str = f"samples={nsamples}"
    return os.path.join(PROJECT_FOLDER, 'results', get_graphs_string(ngraphs, nnodes, nlatent, exp_nbrs), sample_str)


def get_alg_results_folder(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, nsamples: int, alg: str, **kwargs):
    graphs_results_folder = get_graphs_results_folder(ngraphs, nnodes, nlatent, exp_nbrs, nsamples)
    return os.path.join(graphs_results_folder, alg, dict2str(kwargs))


# === GENERATING SAMPLES
def generate_mags_and_samples(ngraphs, nnodes, nlatent, exp_nbrs: float, nsamples):
    """
    Generates [ngraphs] MAGs with [nnodes] nodes, marginalized from an Erdos-Renyi DAG with [nlatent]
    additional variables and [exp_nbrs] expected neighbors.

    Generates [nsamples] samples from each MAG.

    A random seed is set so that the same MAGs are generated for each setting of (ngraphs,nnodes,nlatent,exp_nbrs).
    """
    # === SKIP IF SAMPLES HAVE ALREADY BEEN GENERATED (assume generated for 1st means generated for all)
    if os.path.exists(get_samples_folder(ngraphs, nnodes, nlatent, exp_nbrs, 0, nsamples)):
        return

    # === SET SEEDS FOR REPRODUCIBILITY
    random.seed(9889772)
    np.random.seed(9898725)

    # === GENERATE DAGS AND MAGS
    dags = cd.rand.directed_erdos(nlatent+nnodes, exp_nbrs/(nnodes-1), size=ngraphs, as_list=True)
    gdags = [cd.rand.rand_weights(dag) for dag in dags]
    mags = [dag.marginal_mag(set(range(nlatent)), relabel='default') for dag in dags]

    # === GENERATE SAMPLES
    samples_list = [gdag.sample(nsamples)[:, nlatent:] for gdag in gdags]

    # === SAVE GRAPHS AND SAMPLES
    graph_folders = [get_graph_folder(ngraphs, nnodes, nlatent, exp_nbrs, n) for n in range(ngraphs)]
    samples_folders = [get_samples_folder(ngraphs, nnodes, nlatent, exp_nbrs, n, nsamples) for n in range(ngraphs)]
    for graph_folder, samples_folder, mag, samples in zip(graph_folders, samples_folders, mags, samples_list):
        os.makedirs(samples_folder, exist_ok=True)
        np.save(os.path.join(graph_folder, 'mag_amat.npy'), mag.to_amat())
        np.save(os.path.join(samples_folder, "samples.npy"), samples)


def get_mag_samples(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, graph_num: int, nsamples: int):
    samples_folder = get_samples_folder(ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples)
    if not os.path.exists(samples_folder):
        generate_mags_and_samples(ngraphs, nnodes, nlatent, exp_nbrs, nsamples)
    return np.load(os.path.join(samples_folder, "samples.npy"))


def get_true_mags(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float):
    graph_folders = [get_graph_folder(ngraphs, nnodes, nlatent, exp_nbrs, n) for n in range(ngraphs)]
    mag_filenames = [os.path.join(graph_folder, 'mag_amat.npy') for graph_folder in graph_folders]
    mags = [cd.AncestralGraph.from_amat(np.load(mag_filename)) for mag_filename in mag_filenames]
    return mags


def get_alg_times(ngraphs: int, nnodes: int, nlatent: int, exp_nbrs: float, nsamples: int, alg: str, **kwargs):
    times_filename = get_alg_results_folder(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs) + 'times.npy'

    overwrite = True
    if overwrite or not os.path.exists(times_filename):
        alg_time_filenames = [
            get_alg_time_filename(ngraphs, nnodes, nlatent, exp_nbrs, graph_num, nsamples, alg, **kwargs)
            for graph_num in range(ngraphs)
        ]
        times = np.array([np.load(fn) for fn in alg_time_filenames])
        np.save(times_filename, times)
        return times
    else:
        return np.load(times_filename)

