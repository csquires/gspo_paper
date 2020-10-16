This is the code base to reproduce the experimental results of the paper [Ordering-Based Causal Structure Learning in the Presence of Latent Variables
](https://arxiv.org/abs/1910.09014). For an installable version of the Greedy Sparsest Poset (GSPo) algorithm, please refer to the [causaldag](https://github.com/uhlerlab/causaldag) package.

To start, run `setup.sh`

To change graph generation/number of samples/algorithm parameters, edit
```
simulation_configs/test.py
```

To run algorithms:
```
python3 run_fci_multiple.py
python3 run_fci_plus_multiple.py
python3 run_gspo_multiple.py
```

To plot results:
```
python3 plotting/plot_roc.py
python3 plotting/plot_times.py
python3 plotting/plot_shds.py
python3 plotting/plot_percent_correct.py
```
