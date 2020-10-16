"""
Plot skeleton ROC curves for multiple algorithms and multiple numbers of samples.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from plotting.plot_utils import get_nsample_handles, get_alg_handles, algs2linestyles2, \
    get_fig_name_vary_samples, get_alg_handles2, get_alg_color_handles, get_nsample_line_handles, algs2colors, custom_style
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import get_fprs_skeletons, get_tprs_skeletons
from create_mags_and_samples import get_alg_results_folder
sns.set(rc=custom_style)
import itertools as itr
import numpy as np

ngraphs = 100
nnodes = 50
nlatent = 12
exp_nbrs = 3.  # must be a float
algs = [
    'fci',
    ('gspo', 'empty'),
    ('gspo', 'gsp'),
    ('gspo', 'permutation'),
    'fci_plus',
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
        # dict(alpha=2e-1),
        dict(alpha=3e-1),
        # dict(alpha=4e-1),
        # dict(alpha=5e-1),
        # # dict(alpha=6e-1),
        # dict(alpha=7e-1),
        # dict(alpha=9e-1),
    ],
    # 'gspo': [
    #     dict(alpha=1e-20),
    #     dict(alpha=1e-8),
    #     # dict(alpha=1e-7),
    #     # dict(alpha=1e-6),
    #     dict(alpha=1e-5),
    #     # dict(alpha=1e-4),
    #     dict(alpha=1e-3),
    #     # dict(alpha=1e-2),
    #     dict(alpha=1e-1),
    #     # dict(alpha=2e-1),
    #     dict(alpha=3e-1),
    #     dict(alpha=5e-1),
    # ],
    'gspo': [
        dict(alpha=1e-20, depth=4),
        dict(alpha=1e-8, depth=4),
        # dict(alpha=1e-7, depth=4),
        # dict(alpha=1e-6, depth=4),
        dict(alpha=1e-5, depth=4),
        # dict(alpha=1e-4, depth=4),
        dict(alpha=1e-3, depth=4),
        # dict(alpha=1e-2, depth=4),
        dict(alpha=1e-1, depth=4),
        # dict(alpha=2e-1, depth=4),
        dict(alpha=3e-1, depth=4),
        # dict(alpha=5e-1, depth=4),
    ]

}
nsamples_list = [1000, 10000]
alg_handles = get_alg_color_handles(algs)
nsample_handles, nsamples2ls = get_nsample_line_handles(nsamples_list)

MARKED_ALPHAS_FCI = [7e-1]
MARKED_ALPHAS_FCI_PLUS = [1e-1]
MARKED_ALPHAS_GSPO = [1e-1]

max_fpr = 0

FCI_PLUS = True
FCI = False
GSPO_EMPTY = False
GSPO_PERM = True
GSPO_GSP = True

STAR_SIZE = 100

if FCI:
    alg = 'fci'
    for nsamples in nsamples_list:
        tprs = np.array([
            get_tprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        fprs = np.array([
            get_fprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        avg_tprs = np.mean(tprs, axis=1)
        avg_fprs = np.mean(fprs, axis=1)

        marked_point_ixs = [ix for ix, kwargs in enumerate(alg2kwargs[alg]) if kwargs['alpha'] in MARKED_ALPHAS_FCI]
        marked_points = avg_fprs[marked_point_ixs], avg_tprs[marked_point_ixs]
        plt.scatter(*marked_points, color=algs2colors[alg], marker='*', s=STAR_SIZE)

        max_fpr = max(max_fpr, *avg_fprs)
        sort_ixs = np.argsort(avg_fprs)
        plt.plot(
            avg_fprs[sort_ixs],
            avg_tprs[sort_ixs],
            color=algs2colors[alg],
            linestyle=nsamples2ls[nsamples],
            marker='.'
        )


if FCI_PLUS:
    alg = 'fci_plus'
    for nsamples in nsamples_list:
        tprs = np.array([
            get_tprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        fprs = np.array([
            get_fprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        avg_tprs = np.mean(tprs, axis=1)
        avg_fprs = np.mean(fprs, axis=1)

        marked_point_ixs = [ix for ix, kwargs in enumerate(alg2kwargs[alg]) if kwargs['alpha'] in MARKED_ALPHAS_FCI_PLUS]
        marked_points = avg_fprs[marked_point_ixs], avg_tprs[marked_point_ixs]
        plt.scatter(*marked_points, color=algs2colors[alg], marker='*', s=STAR_SIZE)

        max_fpr = max(max_fpr, *avg_fprs)
        sort_ixs = np.argsort(avg_fprs)
        plt.plot(
            avg_fprs[sort_ixs],
            avg_tprs[sort_ixs],
            color=algs2colors[alg],
            linestyle=nsamples2ls[nsamples],
            marker='.'
        )


if GSPO_EMPTY:
    alg = 'gspo'
    initial = 'empty'
    for nsamples in nsamples_list:
        tprs = np.array([
            get_tprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        fprs = np.array([
            get_fprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        avg_tprs = np.mean(tprs, axis=1)
        avg_fprs = np.mean(fprs, axis=1)

        marked_point_ixs = [ix for ix, kwargs in enumerate(alg2kwargs[alg]) if kwargs['alpha'] in MARKED_ALPHAS_GSPO]
        marked_points = avg_fprs[marked_point_ixs], avg_tprs[marked_point_ixs]
        plt.scatter(*marked_points, color=algs2colors[(alg, initial)], marker='*', s=STAR_SIZE)

        max_fpr = max(max_fpr, *avg_fprs)
        sort_ixs = np.argsort(avg_fprs)
        plt.plot(
            avg_fprs[sort_ixs],
            avg_tprs[sort_ixs],
            color=algs2colors[(alg, initial)],
            linestyle=nsamples2ls[nsamples],
            marker='.'
        )


if GSPO_PERM:
    alg = 'gspo'
    initial = 'permutation'
    for nsamples in nsamples_list:
        tprs = np.array([
            get_tprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        fprs = np.array([
            get_fprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        avg_tprs = np.mean(tprs, axis=1)
        avg_fprs = np.mean(fprs, axis=1)

        marked_point_ixs = [ix for ix, kwargs in enumerate(alg2kwargs[alg]) if kwargs['alpha'] in MARKED_ALPHAS_GSPO]
        marked_points = avg_fprs[marked_point_ixs], avg_tprs[marked_point_ixs]
        plt.scatter(*marked_points, color=algs2colors[(alg, initial)], marker='*', s=STAR_SIZE)

        max_fpr = max(max_fpr, *avg_fprs)
        sort_ixs = np.argsort(avg_fprs)
        plt.plot(
            avg_fprs[sort_ixs],
            avg_tprs[sort_ixs],
            color=algs2colors[(alg, initial)],
            linestyle=nsamples2ls[nsamples],
            marker='.'
        )

if GSPO_GSP:
    alg = 'gspo'
    initial = 'gsp'
    for nsamples in nsamples_list:
        tprs = np.array([
            get_tprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        fprs = np.array([
            get_fprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, initial=initial, **kwargs)
            for kwargs in alg2kwargs[alg]
        ])
        avg_tprs = np.mean(tprs, axis=1)
        avg_fprs = np.mean(fprs, axis=1)

        marked_point_ixs = [ix for ix, kwargs in enumerate(alg2kwargs[alg]) if kwargs['alpha'] in MARKED_ALPHAS_GSPO]
        marked_points = avg_fprs[marked_point_ixs], avg_tprs[marked_point_ixs]
        plt.scatter(*marked_points, color=algs2colors[(alg, initial)], marker='*', s=STAR_SIZE)

        max_fpr = max(max_fpr, *avg_fprs)
        sort_ixs = np.argsort(avg_fprs)
        plt.plot(
            avg_fprs[sort_ixs],
            avg_tprs[sort_ixs],
            color=algs2colors[(alg, initial)],
            linestyle=nsamples2ls[nsamples],
            marker='.'
        )

# for alg, nsamples in itr.product(algs, nsamples_list):
#     tprs = np.array([
#         get_tprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
#         for kwargs in alg2kwargs[alg]
#     ])
#     fprs = np.array([
#         get_fprs_skeletons(ngraphs, nnodes, nlatent, exp_nbrs, nsamples, alg, **kwargs)
#         for kwargs in alg2kwargs[alg]
#     ])
#     print(alg, tprs.mean(axis=1), fprs.mean(axis=1))
#     avg_tprs = np.mean(tprs, axis=1)
#     avg_fprs = np.mean(fprs, axis=1)
#     max_fpr = max(max_fpr, *avg_fprs)
#     sort_ixs = np.argsort(avg_fprs)
#     plt.plot(avg_fprs[sort_ixs], avg_tprs[sort_ixs], color=nsamples2color[nsamples], linestyle=algs2linestyles[alg], marker='.')

FINAL = True
if not FINAL:
    plt.title(f"nnodes={nnodes},exp_nbrs={exp_nbrs},nlatent={nlatent},ngraphs={ngraphs}")
plt.plot([0, max_fpr], [0, max_fpr], alpha=.5, color='grey')
plt.xlabel('Average False Positive Rate (Skeleton)')
plt.ylabel('Average True Positive Rate (Skeleton)')
plt.legend(handles=alg_handles+nsample_handles)
plt.tight_layout(pad=.3)
plt.savefig(get_fig_name_vary_samples(ngraphs, nnodes, nlatent, exp_nbrs, 'roc'))
