import causaldag as cd
import numpy as np
import random
import time

np.random.seed(987)
random.seed(987)

p = 10
K = 3
exp_nbrs = 2
ngraphs = 100
dags = cd.rand.directed_erdos(p+K, exp_nbrs/(p+K-1), ngraphs)
dag_num_edges = np.array([dag.num_arcs for dag in dags])

start = time.time()
mags1 = [dag.marginal_mag(set(range(K)), new=False) for dag in dags]
print(time.time() - start)
mag1_num_edges = np.array([mag.num_edges for mag in mags1])

start = time.time()
mags2 = [dag.marginal_mag(set(range(K)), new=True) for dag in dags]
print(time.time() - start)
mag2_num_edges = np.array([mag.num_edges for mag in mags2])

ancestral1 = [mag._check_ancestral() for mag in mags1]
ancestral2 = [mag._check_ancestral() for mag in mags2]

# print(all([mag1 == mag2 for mag1, mag2 in zip(mags1, mags2)]))
# print(mags1[1])
# print(mags2[1])
#
# print(mag1_num_edges >= mag2_num_edges)
# print(dag_num_edges == mag2_num_edges)
