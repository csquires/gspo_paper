"""
Plot number of samples vs. percent correct.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from plotting.plot_utils import get_nsample_handles, get_alg_handles, algs2linestyles, \
    get_fig_name_vary_samples, algs2colors, linestyles, get_alg_color_handles, custom_style
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import get_same_mec
from create_mags_and_samples import get_alg_results_folder
sns.set(rc=custom_style)
import itertools as itr
import numpy as np

ngraphs = 100
nnodes = 10
nlatent = 3
exp_nbrs = 3.  # must be a float
algs = [
    'fci',
    'fci_plus',
    ('gspo', 'empty'),
    ('gspo', 'permutation')
]
alg2kwargs = {
    'fci': [
        dict(alpha=1e-10),
        dict(alpha=1e-8),
        dict(alpha=1e-5),
        # dict(alpha=1e-4),
        dict(alpha=1e-3),
        # dict(alpha=1e-2),
        dict(alpha=1e-1),
        # dict(alpha=2e-1),
        dict(alpha=3e-1),
        # dict(alpha=4e-1),
        dict(alpha=5e-1),
        # dict(alpha=6e-1),
        dict(alpha=7e-1),
        dict(alpha=9e-1),
    ],
    'fci_plus': [
        dict(alpha=1e-10),
        dict(alpha=1e-8),
        dict(alpha=1e-5),
        # dict(alpha=1e-4),
        dict(alpha=1e-3),
        # dict(alpha=1e-2),
        dict(alpha=1e-1),
        # # dict(alpha=2e-1),
        dict(alpha=3e-1),
        # dict(alpha=4e-1),
        dict(alpha=5e-1),
        # dict(alpha=6e-1),
        dict(alpha=7e-1),
        dict(alpha=9e-1),
    ],
    'gspo': [
        dict(alpha=1e-20, depth=100),
        dict(alpha=1e-8, depth=100),
        # dict(alpha=1e-7, depth=100),
        # dict(alpha=1e-6, depth=100),
        dict(alpha=1e-5, depth=100),
        # dict(alpha=1e-4, depth=100),
        dict(alpha=1e-3, depth=100),
        # dict(alpha=1e-2, depth=100),
        dict(alpha=1e-1, depth=100),
        # dict(alpha=2e-1, depth=100),
        dict(alpha=3e-1, depth=100),
        # dict(alpha=5e-1, depth=100),
    ]

}
nsamples_list = [1000, 10000]
alg_handles = get_alg_color_handles(algs)
nsample_handles, nsamples2color = get_nsample_handles(nsamples_list)

FCI_PLUS = True
FCI = False
GSPO_EMPTY = True
GSPO_PERM = True
GSPO_GSP = True

if FCI:
    alg = 'fci'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    for kwargs, ls in zip(kwargs_list, linestyles):
        same_mec = np.array([
            get_same_mec(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
            for nsamples in nsamples_list
        ])
        print(alg, same_mec.mean(axis=1))
        plt.plot(nsamples_list, same_mec.mean(axis=1), linestyle=ls, color=algs2colors[alg], marker='.')


if FCI_PLUS:
    alg = 'fci_plus'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    for kwargs, ls in zip(kwargs_list, linestyles):
        same_mec = np.array([
            get_same_mec(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg,  **kwargs)
            for nsamples in nsamples_list
        ])
        print(alg, same_mec.mean(axis=1))
        plt.plot(nsamples_list, same_mec.mean(axis=1), color=algs2colors[alg], linestyle=ls, marker='.')


if GSPO_EMPTY:
    alg = 'gspo'
    initial = 'empty'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    for kwargs, ls in zip(kwargs_list, linestyles):
        same_mec = np.array([
            get_same_mec(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for nsamples in nsamples_list
        ])
        print(alg, same_mec.mean(axis=1))
        plt.plot(nsamples_list, same_mec.mean(axis=1), color=algs2colors[(alg, initial)], linestyle=ls, marker='.')


if GSPO_PERM:
    alg = 'gspo'
    initial = 'permutation'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    for kwargs, ls in zip(kwargs_list, linestyles):
        same_mec = np.array([
            get_same_mec(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for nsamples in nsamples_list
        ])
        print(alg, same_mec.mean(axis=1))
        plt.plot(nsamples_list, same_mec.mean(axis=1), linestyle=ls, color=algs2colors[(alg, initial)], marker='.')


# for alg, kwargs_list in alg2kwargs.items():
#     for kwargs, color in zip(kwargs_list, sns.color_palette()):
#         same_mec = np.array([
#             get_same_mec(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
#             for nsamples in nsamples_list
#         ])
#         plt.plot(nsamples_list, same_mec.mean(axis=1), color=color, linestyle=algs2linestyles[alg], marker='.')


plt.title(f"nnodes={nnodes},exp_nbrs={exp_nbrs},nlatent={nlatent},ngraphs={ngraphs}")
plt.xlabel('Number of Samples')
plt.ylabel('Percent Correct MEC')
plt.legend(handles=alg_handles)
plt.savefig(get_fig_name_vary_samples(ngraphs, nnodes, nlatent, exp_nbrs, 'same_mec'))
