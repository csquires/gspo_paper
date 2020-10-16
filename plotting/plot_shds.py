"""
Plot number of samples vs. average SHD.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from plotting.plot_utils import get_nsample_handles, get_alg_handles, algs2linestyles, \
    get_fig_name_vary_samples, linestyles, algs2colors, get_alg_color_handles, custom_style
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import get_fprs_skeletons, get_tprs_skeletons, get_shd_skeletons
from create_mags_and_samples import get_alg_results_folder
sns.set(rc=custom_style)
import itertools as itr
import numpy as np
from matplotlib import rcParams

plt.clf()

ngraphs = 100
nnodes = 10
nlatent = 3
exp_nbrs = 3.  # must be a float
algs = [
    'fci',
    'fci_plus',
    ('gspo', 'empty'),
    ('gspo', 'permutation'),
    ('gspo', 'gsp')
]
alg2kwargs = {
    'fci': [
        dict(alpha=1e-10),
        dict(alpha=1e-8),
        dict(alpha=1e-5),
        dict(alpha=1e-3),
        dict(alpha=1e-1),
        # dict(alpha=5e-1),
        dict(alpha=7e-1),
    ],
    'fci_plus': [
        dict(alpha=1e-10),
        dict(alpha=1e-8),
        dict(alpha=1e-5),
        dict(alpha=1e-3),
        dict(alpha=1e-1),
        # dict(alpha=5e-1),
        dict(alpha=7e-1),
    ],
    'gspo': [
        dict(alpha=1e-20, depth=4),
        dict(alpha=1e-8, depth=4),
        dict(alpha=1e-5, depth=4),
        dict(alpha=1e-3, depth=4),
        dict(alpha=1e-1, depth=4),
        dict(alpha=3e-1, depth=4),
        dict(alpha=5e-1, depth=4)
    ]

}
nsamples_list = [500, 1000, 2500, 5000, 10000]
alg_handles = get_alg_color_handles(algs)

FCI_PLUS = True
FCI = True
GSPO_EMPTY = True
GSPO_PERM = True
GSPO_GSP = True

if FCI_PLUS:
    alg = 'fci_plus'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best

    avg_shds_list = np.array([
        np.array([
            get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
            for nsamples in nsamples_list
        ]).mean(axis=1)
        for kwargs in kwargs_list
    ])
    best_avg_shd_ix = np.argmin(avg_shds_list.mean(axis=1))
    print(kwargs_list[best_avg_shd_ix])
    plt.plot(nsamples_list, avg_shds_list[best_avg_shd_ix], color=algs2colors[alg])
    # for kwargs, ls in zip(kwargs_list, linestyles):
    #     shds = np.array([
    #         get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
    #         for nsamples in nsamples_list
    #     ])
    #     print(alg, kwargs)
    #     plt.plot(nsamples_list, shds.mean(axis=1), linestyle=ls, color=algs2colors[alg], marker='.')


if FCI:
    alg = 'fci'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    avg_shds_list = np.array([
        np.array([
            get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
            for nsamples in nsamples_list
        ]).mean(axis=1)
        for kwargs in kwargs_list
    ])
    best_avg_shd_ix = np.argmin(avg_shds_list.mean(axis=1))
    print(kwargs_list[best_avg_shd_ix])
    plt.plot(nsamples_list, avg_shds_list[best_avg_shd_ix], color=algs2colors[alg])

    # for kwargs, ls in zip(kwargs_list, linestyles):
    #     shds = np.array([
    #         get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
    #         for nsamples in nsamples_list
    #     ])
    #     print(alg, kwargs)
    #     plt.plot(nsamples_list, shds.mean(axis=1), linestyle=ls, color=algs2colors[alg], marker='.')

if GSPO_EMPTY:
    alg = 'gspo'
    initial = 'empty'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best

    avg_shds_list = np.array([
        np.array([
            get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for nsamples in nsamples_list
        ]).mean(axis=1)
        for kwargs in kwargs_list
    ])
    best_avg_shd_ix = np.argmin(avg_shds_list.mean(axis=1))
    print(kwargs_list[best_avg_shd_ix])
    plt.plot(nsamples_list, avg_shds_list[best_avg_shd_ix], color=algs2colors[(alg, initial)])

    # for kwargs, ls in zip(kwargs_list, linestyles):
    #     shds = np.array([
    #         get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
    #         for nsamples in nsamples_list
    #     ])
    #     print(alg, kwargs)
    #     plt.plot(nsamples_list, shds.mean(axis=1), linestyle=ls, color=algs2colors[(alg, initial)], marker='.')

if GSPO_PERM:
    alg = 'gspo'
    initial = 'permutation'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best

    avg_shds_list = np.array([
        np.array([
            get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for nsamples in nsamples_list
        ]).mean(axis=1)
        for kwargs in kwargs_list
    ])
    best_avg_shd_ix = np.argmin(avg_shds_list.mean(axis=1))
    print(kwargs_list[best_avg_shd_ix])
    plt.plot(nsamples_list, avg_shds_list[best_avg_shd_ix], color=algs2colors[(alg, initial)])

    # for kwargs, ls in zip(kwargs_list, linestyles):
    #     shds = np.array([
    #         get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
    #         for nsamples in nsamples_list
    #     ])
    #     print(alg, kwargs)
    #     plt.plot(nsamples_list, shds.mean(axis=1), linestyle=ls, color=algs2colors[(alg, initial)], marker='.')


if GSPO_PERM:
    alg = 'gspo'
    initial = 'gsp'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best

    avg_shds_list = np.array([
        np.array([
            get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for nsamples in nsamples_list
        ]).mean(axis=1)
        for kwargs in kwargs_list
    ])
    best_avg_shd_ix = np.argmin(avg_shds_list.mean(axis=1))
    print(kwargs_list[best_avg_shd_ix])
    plt.plot(nsamples_list, avg_shds_list[best_avg_shd_ix], color=algs2colors[(alg, initial)])

    # for kwargs, ls in zip(kwargs_list, linestyles):
    #     shds = np.array([
    #         get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
    #         for nsamples in nsamples_list
    #     ])
    #     print(alg, kwargs)
    #     plt.plot(nsamples_list, shds.mean(axis=1), linestyle=ls, color=algs2colors[(alg, initial)], marker='.')

# for alg, kwargs_list in alg2kwargs.items():
#     for kwargs, color in zip(kwargs_list, sns.color_palette()):
#         print(kwargs)
#         shds = np.array([get_shd_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs) for nsamples in nsamples_list])
#
#         plt.plot(nsamples_list, shds.mean(axis=1), color=color, linestyle=algs2linestyles[alg], marker='.')

FINAL = True
if not FINAL:
    plt.title(f"nnodes={nnodes},exp_nbrs={exp_nbrs},nlatent={nlatent},ngraphs={ngraphs}")
plt.xticks(nsamples_list, ['', *nsamples_list[1:]])
plt.xlabel('Number of Samples')
plt.ylabel('Average SHD')
plt.legend(handles=alg_handles, ncol=2)
plt.tight_layout(pad=.3)
plt.savefig(get_fig_name_vary_samples(ngraphs, nnodes, nlatent, exp_nbrs, 'shds'))
