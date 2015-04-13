from __future__ import division

import logging
import math
import multiprocessing as mp
import os

import numpy as np

from msibi.potentials import tail_correction
from msibi.workers import run_query_simulations


class MSIBI(object):
    """Management class for orchestrating an MSIBI optimization.

    Parameters
    ----------
    rdf_cutoff : float
        The upper cutoff value for the RDF calculation.
    n_points : int
        The number of radius values.
    pot_cutoff : float, optional, default=rdf_cutoff
        The upper cutoff value for the potential.
    r_switch : float, optional, default=pot_r[-5]
        The radius after which a tail correction is applied.
    status_filename : str, optional, default='f_fits.log'
        A log file for tracking the quality of fits at every iteration.
    smooth_rdfs : bool, optional, default=False
        Use a smoothing function to reduce the noise in the RDF data.

    Attributes
    ----------
    states : list of States
        All states to be used in the optimization procedure.
    pairs : list of Pairs
        All pairs to be used in the optimization procedure.
    n_iterations : int, optional, default=10
        The number of MSIBI iterations to perform.
    rdf_cutoff : float
        The upper cutoff value for the RDF calculation.
    n_rdf_points : int
        The number of radius values used in the RDF calculation.
    dr : float, default=rdf_cutoff / (n_points - 1)
        The spacing of radius values.
    pot_cutoff : float, optional, default=rdf_cutoff
        The upper cutoff value for the potential.
    pot_r : np.ndarray, shape=(int((rdf_cutoff + dr) / dr),)
        The radius values at which the potential is computed.
    r_switch : float, optional, default=pot_r[-1] - 5 * dr
        The radius after which a tail correction is applied.

    """

    def __init__(self, rdf_cutoff, n_rdf_points, pot_cutoff=None, r_switch=None,
                 status_filename='f_fits.log', smooth_rdfs=False, base_dir=None):
        self.states = []
        self.pairs = []
        self.n_iterations = 10  # Can be overridden in optimize().

        self.rdf_cutoff = rdf_cutoff
        self.n_rdf_points = n_rdf_points
        self.dr = rdf_cutoff / (n_rdf_points - 1)
        self.smooth_rdfs = smooth_rdfs
        self.rdf_r_range = np.array([0.0, self.rdf_cutoff + self.dr])
        self.rdf_n_bins = self.n_rdf_points + 1


        # TODO: Description of use for pot vs rdf cutoff.
        if not pot_cutoff:
            pot_cutoff = rdf_cutoff
        self.pot_cutoff = pot_cutoff
        # TODO: Describe why potential needs to be messed with to match the RDF.
        self.pot_r = np.arange(0.0, self.pot_cutoff + self.dr, self.dr)

        if not r_switch:
            r_switch = self.pot_r[-5]
        self.r_switch = r_switch

        if base_dir is None:
            self.base_dir = os.getcwd()
        else:
            self.base_dir = base_dir

        status_filename = os.path.join(self.base_dir, status_filename)
        logging.basicConfig(filename=status_filename, level=logging.INFO,
                            format='%(message)s', filemode='a')

    def optimize(self, states, pairs, n_iterations=10, engine='hoomd',
                 start_iteration=0):
        """
        """
        self.states = states
        self.pairs = pairs
        if n_iterations:
            self.n_iterations = n_iterations
        self.initialize(engine=engine)

        for n in range(start_iteration + self.n_iterations):
            logging.info("-------- Iteration {n} --------".format(**locals()))
            run_query_simulations(self.states, engine=engine)
            self._update_potentials(n, engine)

    def _update_potentials(self, iteration, engine):
        """Update the potentials for each pair. """
        updated_rdf_f_fit = self._recompute_rdfs(iteration)

        state_id = 0
        for pair in self.pairs:
            for state in pair.states:
                # Gather all the RDF information.
                rdf, f_fit = updated_rdf_f_fit[state_id]
                pair.states[state]['current_rdf'] = rdf
                pair.states[state]['f_fit'].append(f_fit)
                state_id += 1
            # Update the potential.
            pair.update_potential(self.pot_r, self.r_switch)
            pair.save_table_potential(self.pot_r, self.dr, iteration=iteration,
                                      engine=engine)

    def _recompute_rdfs(self, iteration):
        """Recompute all RDFs after a given iteration. """
        # Give each state an ID and store it in a manager's dict which is used
        # by _rdf_worker to lookup the pair and state from the main process.
        manager = mp.Manager()
        manager_dict = manager.dict()
        state_id = 0
        for pair in self.pairs:
            for state in pair.states:
                manager_dict[state_id] = (pair, state)
                state_id += 1

        # Chunk and launch RDF processes.
        # CTK: Everything below should be "more correctly" accomplishable with
        # a pool but I couldn't get it to work properly.
        n_procs = mp.cpu_count()
        state_ids = manager_dict.keys()
        chunk_size = int(math.ceil(len(state_ids) / n_procs))
        procs = list()
        for n in range(n_procs):
            state_ids_chunk = state_ids[n * chunk_size: (n + 1) * chunk_size]
            p = mp.Process(target=self._rdf_worker,
                           args=(manager_dict, state_ids_chunk, iteration))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        return manager_dict

    def _rdf_worker(self, manager_dict, state_ids, iteration):
        """Recompute the current RDFs for a chunk of (pair, state) tuples. """
        for state_id in state_ids:
            pair, state = manager_dict[state_id]
            rdf, f_fit = pair.compute_current_rdf(state, self.rdf_r_range,
                                                  n_bins=self.rdf_n_bins,
                                                  smooth=self.smooth_rdfs)

            # Save RDF to a file for post-processing.
            filename = 'rdfs/pair_{0}-state_{1}-step{2}.txt'.format(
                pair.name, state.name, iteration)
            filepath = os.path.join(self.base_dir, filename)
            rdf[:, 0] -= self.dr / 2.0
            np.savetxt(filepath, rdf)
            logging.info('pair {0}, state {1}, iteration {2}: {3:f}'.format(
                         pair.name, state.name, iteration, f_fit))

            # Store the RDF and fitness function in the shared dict.
            manager_dict[state_id] = (rdf, f_fit)
            # NOTE: Originally this tuple was (pair, state). We are overwriting
            # it here both to pass the information back out and signify that
            # this (pair, state) has been re-computed. There may be a better way
            # to do this.
            #
            # For more info see:
            # https://docs.python.org/2/library/multiprocessing.html#multiprocessing.managers.SyncManager.list

    def initialize(self, engine='hoomd', potentials_dir=None):
        """Create initial table potentials and the simulation input scripts.

        Parameters
        ----------
        engine : str, optional, default='hoomd'
        potentials_dir : path, optional, default="'working_dir'/potentials"

        """
        if not potentials_dir:
            self.potentials_dir = os.path.join(self.base_dir, 'potentials')
        else:
            self.potentials_dir = potentials_dir
        try:
            os.mkdir(self.potentials_dir)
        except OSError:
            # TODO: Warning and maybe a make backups.
            pass

        table_potentials = []
        for pair in self.pairs:
            potential_file = os.path.join(self.potentials_dir,
                                          'pot.{0}.txt'.format(pair.name))
            pair.potential_file = potential_file
            table_potentials.append((pair.type1, pair.type2, potential_file))

            V = tail_correction(self.pot_r, pair.potential, self.r_switch)
            pair.potential = V
            pair.save_table_potential(self.pot_r, self.dr, iteration=-1,
                                      engine=engine)

        for state in self.states:
            state.save_runscript(table_potentials, table_width=len(self.pot_r),
                                 engine=engine)
