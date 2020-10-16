import numpy as np
import subprocess
import os
import causaldag as cd
FCI_FILENAME = os.path.join(os.path.dirname(__file__), 'fci.R')


def fci(samples, alpha, dag_num, fci_plus=False):
    p = samples.shape[1]

    # === SAVE SAMPLES
    samples_filename = f'tmp_file_{dag_num}.npy'
    np.save(samples_filename, samples)

    # === RUN FCI AND CONVERT OUTPUT
    r_output = subprocess.check_output(['Rscript', FCI_FILENAME, samples_filename, str(alpha), str(fci_plus)])
    r_output = r_output.split(b'\n')[-1]
    amat = np.array(list(map(int, r_output.decode().split(' ')))).reshape([p, p]).T
    try:
        mag = cd.AncestralGraph.from_amat(amat)
    except Exception as e:  # sometimes returns non-ancestral graph??
        mag = cd.AncestralGraph(nodes=set(range(p)))
    skeleton = cd.UndirectedGraph.from_amat(amat)
    # if not np.alltrue(pag.to_amat() == amat):
    #     print(pag.to_amat())
    #     print(amat)

    # === CLEAN UP AND RETURN
    os.remove(samples_filename)
    return mag, skeleton



