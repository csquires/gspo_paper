ngraphs = 100
nnodes_list = [10]
nlatent = 3
exp_nbrs = 3.
nsamples_list = [1000, 10000]

fci_alphas = [1e-10, 1e-8, 1e-5, 1e-3, 1e-1, 3e-1, 5e-1, 7e-1]
gspo_alphas = [1e-20, 1e-8, 1e-5, 1e-3, 1e-1, 3e-1, 5e-1]
gspo_initial_list = ['gsp', 'empty', 'permutation']
gspo_nruns = 5
gspo_depth = 4
gspo_strict_lmcs = True
lmc_update = 'construction'
gspo_max_iters = 1000
