import causaldag as cd
import numpy as np
import itertools as itr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

p = 50
Ks = [12]
exp_nbrs_list = [3]
ngraphs = 100
max_num_edges = p*(p-1)/2
np.random.seed(987982365)

for K, exp_nbrs in itr.product(Ks, exp_nbrs_list):
    dags = cd.rand.directed_erdos(p+K, exp_nbrs/(p+K-1), ngraphs)
    mags = [dag.marginal_mag(set(range(K))) for dag in dags]
    num_edges = np.array([mag.num_edges for mag in mags])
    num_bidirected = np.array([mag.num_bidirected for mag in mags])
    percent_bidirected = num_bidirected/num_edges
    print(K, exp_nbrs, num_edges.mean(), max_num_edges)
    prob_edge = num_edges.mean()/max_num_edges
    print(prob_edge*(p-1))
    print(percent_bidirected.mean())
    print(percent_bidirected.min())
    print(percent_bidirected.max())
    plt.scatter(num_edges, num_bidirected)
    plt.plot([0, num_edges.max()], [0, num_edges.max()], color='grey', alpha=.5)
    plt.plot([0, num_edges.max()], [0, num_edges.max()/2], color='grey', alpha=.5, linestyle='dashed')
    plt.xlabel('Total number of edges')
    plt.ylabel('Number of bidirected edges')
    plt.savefig('figures/sparsity.png')



