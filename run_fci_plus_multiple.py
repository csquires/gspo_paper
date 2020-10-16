import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import itertools as itr
import os
from simulation_configs.roc import ngraphs, nnodes_list, nlatent, exp_nbrs, nsamples_list, fci_alphas
from config import SERVER

for nnodes, nsamples, alpha in itr.product(nnodes_list, nsamples_list, fci_alphas):
    print(nsamples, alpha)

    graph_setting = f"--ngraphs {ngraphs} --nnodes {nnodes} --nlatent {nlatent} --exp_nbrs {exp_nbrs}"
    sample_setting = f"--nsamples {nsamples}"
    alg_setting = f"--alpha {alpha} --fci_plus {True}"
    full_command = f"python3 run_fci_single.py {graph_setting} {sample_setting} {alg_setting}"
    if SERVER:
        pass  # TODO
    else:
        os.system(full_command)
