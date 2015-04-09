import math
import multiprocessing as mp
from multiprocessing.dummy import Pool
import os
from subprocess import Popen

import numpy as np

from msibi.utils.general import backup_file
from msibi.utils.exceptions import UnsupportedEngine

import logging
logging.basicConfig(level=logging.DEBUG,
        format='%(threadName)-10s %(message)s')

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
        #proc = Popen(['hoomd', 'run.py', '--gpu=%d' % (card)],
        proc = Popen(['hoomd', 'run.py'],
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

def calc_query_rdfs(pairs, rdf_cutoff, dr, n_rdf_points, nprocs):
    procs = []
    queue = mp.Queue()
    new_rdfs = {}

    # make a list of rdfs to calculate in form of (pair, state) to make 
    # iterating easier
    pair_states = []
    for pair in pairs:
        for state in pair.states.keys():
            pair_states.append((pair, state))
            logging.debug('in initializer %d' % id(pair))

    # divide the rdfs to calculate roughly equally among processors
    chunksize = int(math.ceil(len(pair_states) / float(nprocs)))
    
    r_range = np.array([0.0, rdf_cutoff + dr])
    n_bins = n_rdf_points + 1
    for i in range(nprocs):  # each processor will add a list to queue
        p = mp.Process(target=_rdf_worker,
                       args=(queue, 
                             pair_states[chunksize * i:chunksize * (i+1)],
                             r_range, n_bins))
        procs.append(p)
        p.start()

    # now collect new rdfs
    for i in range(nprocs):
        proc_data = queue.get()  # a list of tuples of (pair, state, rdf, f_fit)
        for data in proc_data:  # a tuple of (pair, state, rdf, f_fit)
            # this should be more pythonic somehow
            pair, state, rdf, f_fit = data[0], data[1], data[2], data[3]
            for tpair in pairs:
                for tstate in tpair.states.keys():
                    if pair.name == tpair.name and state.name == tstate.name:
                        logging.debug('in updater %d' % id(pair))
                        tpair.states[tstate]['current_rdf'] = rdf
                        tpair.states[tstate]['f_fit'].append(f_fit)

    # wait for all processes to finish (not quite sure why this is here)
    for i, p in enumerate(procs):
        p.join()

def _rdf_worker(out_q, pair_states, r_range, n_bins):
    # each process will write a list of (pair, state, rdf, f_fit)
    data = []
    for pair, state in pair_states:
        logging.debug('in worker %d' % id(pair))
        rdf, f_fit = pair.compute_current_rdf(state, r_range, n_bins)
        data.append((pair, state, rdf, f_fit))
    if pair_states:
        logging.debug('first item in worker list %d' % id(data[0][0]))
    out_q.put(data)
