import math
import multiprocessing as mp
from multiprocessing.dummy import Pool
import os
from subprocess import Popen
from multiprocessing import Process, Queue

import numpy as np

from msibi.utils.general import backup_file
from msibi.utils.exceptions import UnsupportedEngine


HARDCODE_N_GPUS = 4
def run_query_simulations(states, engine='hoomd'):
    """Run all query simulations for a single iteration. """
    # TODO: GPU count and proper "cluster management"
    pool = Pool(HARDCODE_N_GPUS)   # should be max number concurrent simulations
    print('Changed pool to Pool(4) for ACCRE. See https://github.com/ctk3b/msibi/issues/5')
    print("Launching {0:d} threads...".format(HARDCODE_N_GPUS))
    if engine.lower() == 'hoomd':
        worker = _hoomd_worker
    else:
        raise UnsupportedEngine(engine)
    chunk_size = max(1, len(states) // HARDCODE_N_GPUS)
    pool.imap(worker, zip(states, range(len(states))), chunk_size)
    print('Also added chunksize into imap with hard-coded values')
    pool.close()
    pool.join()


def _hoomd_worker(state):
    """Worker for managing a single HOOMD-blue simulation. """
    idx = state[1]
    state = state[0]  # so i don't have to rename below
    log_file = os.path.join(state.state_dir, 'log.txt')
    err_file = os.path.join(state.state_dir, 'err.txt')
    with open(log_file, 'w') as log, open(err_file, 'w') as err:
        print('running state %s on gpu %d' % (state.name, idx % HARDCODE_N_GPUS))
        card = idx % HARDCODE_N_GPUS
        proc = Popen(['hoomd', 'run.py', '--gpu=%d' % (card)],
                cwd=state.state_dir, stdout=log, stderr=err,
                universal_newlines=True)
        print("    Launched HOOMD in {0}...".format(state.state_dir))
        proc.communicate()
        print("    Finished in {0}.".format(state.state_dir))
    _post_query(state)


def _post_query(state):
    state.reload_query_trajectory()
    backup_file(os.path.join(state.state_dir, 'log.txt'))
    backup_file(os.path.join(state.state_dir, 'err.txt'))
    if state.backup_trajectory:
        backup_file(state.traj_path)

def calc_query_rdfs(pairs, rdf_cutoff, dr, n_rdf_points):
    procs = []
    queue = Queue()
    for pair in pairs:
        for state in pair.states:
            r_range = np.array([0.0, rdf_cutoff + dr])
            n_bins = n_rdf_points + 1
            p = Process(target=_calc_pair_rdf_at_state, 
                        args=(queue, pair, state, r_range, n_bins))
            p.start()
            procs.append(p)
            #pair.compute_current_rdf(state, r_range, n_bins=self.n_rdf_points+1)

    for p in procs:
        p.join()

    for pair in pairs:
        for state in pair.states:
            pair.states[state]['current_rdf'], f_fit = queue.get()
            pair.states[state]['f_fit'].append(f_fit)

def _calc_pair_rdf_at_state(queue, pair, state, r_range, n_bins):
    print pair.name, state.name, state.state_dir
    queue.put(pair.compute_current_rdf(state, r_range, n_bins))
