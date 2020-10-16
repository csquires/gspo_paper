"""
Plot skeleton ROC curves for multiple algorithms and multiple numbers of samples.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from plotting.plot_utils import get_alg_handles, algs2linestyles, get_fig_name_vary_nodes, algs2colors, \
    linestyles, get_alg_color_handles, custom_style
import matplotlib.pyplot as plt
import seaborn as sns
from create_mags_and_samples import get_alg_times
sns.set(rc=custom_style)
import numpy as np

ngraphs = 100
nlatent = 3
exp_nbrs = 3.  # must be a float
algs = [
    'fci',
    'fci_plus',
    ('gspo', 'empty'),
    ('gspo', 'permutation'),
    ('gspo', 'gsp'),
]
alg2kwargs = {
    'fci': [
        # dict(alpha=1e-6),
        # dict(alpha=1e-5),
        dict(alpha=1e-3),
        # dict(alpha=1e-3),
    ],
    'fci_plus': [
        # dict(alpha=1e-5),
        # dict(alpha=1e-3),
        dict(alpha=1e-1)
    ],
    'gspo': [
        # dict(alpha=1e-20, depth=4, max_iters=1000),
        dict(alpha=1e-1, depth=4, max_iters=1000),
        # dict(alpha=1e-1, depth=4, max_iters=1000),
    ]
}
nsamples = 1000
nnodes_list = [10, 20, 30, 40, 50]

FCI_PLUS = True
FCI = True
GSPO_EMPTY = True
GSPO_PERM = True
GSPO_GSP = True

algs = []
if FCI: algs.append('fci')
if FCI_PLUS: algs.append('fci_plus')
if GSPO_EMPTY: algs.append(('gspo', 'empty'))
if GSPO_PERM: algs.append(('gspo', 'permutation'))
if GSPO_GSP: algs.append(('gspo', 'gsp'))
alg_handles = get_alg_color_handles(algs)

MEDIAN = False

if FCI_PLUS:
    alg = 'fci_plus'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    for kwargs, ls in zip(kwargs_list, linestyles):
        times = np.array([
            get_alg_times(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
            for nnodes in nnodes_list
        ])
        print(alg, kwargs)
        print(times.shape)
        points = times.mean(axis=1) if not MEDIAN else np.median(times, axis=1)
        plt.plot(nnodes_list, points, linestyle=ls, color=algs2colors[alg])


if FCI:
    alg = 'fci'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    for kwargs, ls in zip(kwargs_list, linestyles):
        times = []
        for nnodes in nnodes_list:
            # try:
            times.append(get_alg_times(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs))
            # except Exception as e:
            #     times.append([np.nan]*ngraphs)
        times = np.array(times)
        points = times.mean(axis=1) if not MEDIAN else np.median(times, axis=1)
        plt.plot(nnodes_list, points, linestyle=ls, color=algs2colors[alg])


if GSPO_EMPTY:
    alg = 'gspo'
    initial = 'empty'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    for kwargs, ls in zip(kwargs_list, linestyles):
        times = np.array([
            get_alg_times(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for nnodes in nnodes_list
        ])
        print(alg, kwargs)
        print(times.shape)
        print(times)
        points = times.mean(axis=1) if not MEDIAN else np.median(times, axis=1)
        plt.plot(nnodes_list, points, linestyle=ls, color=algs2colors[(alg, initial)])


if GSPO_PERM:
    alg = 'gspo'
    initial = 'permutation'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    for kwargs, ls in zip(kwargs_list, linestyles):
        times = np.array([
            get_alg_times(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for nnodes in nnodes_list
        ])
        print(alg, kwargs)
        print(times.shape)
        points = times.mean(axis=1) if not MEDIAN else np.median(times, axis=1)
        plt.plot(nnodes_list, points, linestyle=ls, color=algs2colors[(alg, initial)])

if GSPO_GSP:
    alg = 'gspo'
    initial = 'gsp'
    kwargs_list = alg2kwargs[alg]  # TODO: consider only picking best
    for kwargs, ls in zip(kwargs_list, linestyles):
        times = np.array([
            get_alg_times(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for nnodes in nnodes_list
        ])
        print(alg, kwargs)
        print(times.shape)
        points = times.mean(axis=1) if not MEDIAN else np.median(times, axis=1)
        plt.plot(nnodes_list, points, linestyle=ls, color=algs2colors[(alg, initial)])


# for alg, kwargs_list in alg2kwargs.items():
#     for kwargs, color in zip(kwargs_list, sns.color_palette()):
#         times = np.array([
#             get_alg_times(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
#             for nnodes in nnodes_list
#         ])
#         print(alg, kwargs)
#         print(times.shape)
#         plt.plot(nnodes_list, times.mean(axis=1), linestyle=algs2linestyles[alg], color=color)

plt.tight_layout()
FINAL = True
if not FINAL:
    plt.title(f"nsamples={nsamples},exp_nbrs={exp_nbrs},nlatent={nlatent},ngraphs={ngraphs}")
plt.xticks(nnodes_list)
plt.xlabel('Number of nodes')
plt.ylabel('Average computation time (seconds)')
plt.legend(handles=alg_handles, fontsize=14)
TIME_STR = 'time_median' if MEDIAN else 'time_mean'
plt.tight_layout(pad=.3)
plt.savefig(get_fig_name_vary_nodes(ngraphs, nlatent, exp_nbrs, nsamples, TIME_STR))
