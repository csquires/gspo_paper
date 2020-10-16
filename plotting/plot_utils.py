from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from seaborn import color_palette
from config import FIG_FOLDER
import os
from create_mags_and_samples import get_graphs_string
# import matplotlib.pyplot as plt
import matplotlib as mpl

algs2linestyles = {
    'gspo': '-',
    'fci': '--',
    'fci_plus': ':'
}

algs2linestyles2 = {
    ('gspo', 'empty'): '-',
    ('gspo', 'permutation'): (0, (3, 1, 1, 1, 1, 1)),
    ('gspo', 'gsp'): '-.',
    'fci_plus': ':',
    'fci': '--'
}

algs = [
    ('gspo', 'empty'),
    ('gspo', 'permutation'),
    ('gspo', 'gsp'),
    'fci_plus',
    'fci'
]
algs2colors = dict(zip(algs, color_palette()))

alg2label = {
    'fci': 'FCI',
    'gspo': 'GSPo',
    'fci_plus': 'FCI+'
}
alg2label2 = {
    'fci': 'FCI',
    ('gspo', 'empty'): 'GSPo (empty)',
    ('gspo', 'permutation'): 'GSPo (MD)',
    ('gspo', 'gsp'): 'GSPo (GSP)',
    'fci_plus': 'FCI+'
}

linestyles = ['-', ':', '--', '-.']

custom_style = {
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.labelsize': 16,
}


def get_nsample_handles(nsamples_list):
    handles = [Patch(color=c, label=nsamples) for c, nsamples in zip(color_palette(), nsamples_list)]
    nsamples2color = dict(zip(nsamples_list, color_palette()))
    return handles, nsamples2color


def get_nsample_line_handles(nsamples_list):
    handles = [Line2D([0], [0], linestyle=ls, label=f"$n={nsamples}$", color='k') for ls, nsamples in zip(linestyles, nsamples_list)]
    nsamples2linestyles = dict(zip(nsamples_list, linestyles))
    return handles, nsamples2linestyles


def get_alg_handles(algs):
    return [Line2D([0], [0], linestyle=ls, label=alg2label[alg], color='k') for alg, ls in algs2linestyles.items() if alg in algs]


def get_alg_handles2(algs):
    return [Line2D([0], [0], linestyle=ls, label=alg2label2[alg], color='k') for alg, ls in algs2linestyles2.items() if alg in algs]


def get_alg_color_handles(algs):
    return [Patch(color=c, label=alg2label2[alg]) for alg, c in algs2colors.items() if alg in algs]


def get_fig_name_vary_samples(ngraphs, nnodes, nlatent, exp_nbrs, figtype, nsamples=None):
    samples_str = f"nsamples={nsamples}" if nsamples is not None else ''
    fig_name = os.path.join(FIG_FOLDER, get_graphs_string(ngraphs, nnodes, nlatent, exp_nbrs), samples_str, figtype+'.png')
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    return fig_name


def get_fig_name_vary_nodes(ngraphs, nlatent, exp_nbrs, nsamples, figtype):
    setting_str = f"ngraphs={ngraphs},nlatent={nlatent},exp_nbrs={exp_nbrs},nsamples={nsamples}"
    fig_name = os.path.join(FIG_FOLDER, setting_str, figtype+'.png')
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    return fig_name
