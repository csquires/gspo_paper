import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import itertools as itr
import os
from simulation_configs.roc import ngraphs, nnodes_list, nlatent, exp_nbrs, nsamples_list
from simulation_configs.roc import gspo_alphas, gspo_strict_lmcs, gspo_initial_list, gspo_depth, gspo_nruns, lmc_update, gspo_max_iters
from config import SERVER

for nnodes, nsamples, alpha, initial in itr.product(nnodes_list, nsamples_list, gspo_alphas, gspo_initial_list):
    graph_setting = f"--ngraphs {ngraphs} --nnodes {nnodes} --nlatent {nlatent} --exp_nbrs {exp_nbrs}"
    sample_setting = f"--nsamples {nsamples}"
    print(graph_setting, sample_setting)
    alg_setting = f"--alpha {alpha} --strict {gspo_strict_lmcs} --initial {initial} --depth {gspo_depth}"
    alg_setting += f" --nruns {gspo_nruns} --lmc_update {lmc_update} --max_iters {gspo_max_iters}"
    full_command = f"python3 run_gspo_single.py {graph_setting} {sample_setting} {alg_setting}"
    if SERVER:
        pass  # TODO
    else:
        os.system(full_command)
