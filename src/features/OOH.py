import numpy as np
import mdtraj as md
from sklearn.base import BaseEstimator, TransformerMixin
from .utils import get_square_distances
import copy

class OOH(BaseEstimator, TransformerMixin):
    """
    Compute the O-O and O-H distances for every water molecule

    Each water vector will look like:
        [d(O1, O2), d(O1, O3), ..., d(O1, ON), 
            d(O1, H2), d(O1, H2), ..., d(O1, H(2N))]

    Parameters
    ----------
    n_waters : int, optional
        Limit the feature vectors to the closest n_waters.
        If None, then all waters are included.
    sortH : str {'local', 'global'}
        The distance from each oxygen to it's neighboring
        hydrogens can be sorted in one of two ways:
            - local: The vector is sorted by the O-O distances
                and for each water, the two O-H distances are
                sorted by increasing distance
            - global: The O-H distances are sorted by 
                increasing distance. The Hydrogens are then
                disassociated from their water atom 
    """
    def __init__(self, n_waters=None, sortH='local', remove_selfH=False):
        if n_waters is None:
            self.n_waters = n_waters
        else:
            self.n_waters = int(n_waters)

        if not sortH.lower() in ['local', 'global']:
            raise ValueError("%s not in %s" % (sortH, str(['local', 'global'])))

        self.sort_locally = False
        if sortH.lower() == 'local':
            self.sort_locally = True

        self.remove_selfH = bool(remove_selfH)

    def transform(self, traj):
        """
        Transform a trajectory.
        
        Parameters
        ----------
        traj : mdtraj.Trajectory
            trajectory to compute distances for

        Returns
        -------
        Xnew : np.ndarray
            distances for each water molecule
        distances : np.ndarray
            distance between each water molecule in the simulation
        """

        oxygens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'O'])
        hydrogens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'H'])
        n_oxygens = len(oxygens)

        OOdistances = get_square_distances(traj, oxygens)

        if self.n_waters is None:
            n_waters = n_oxygens - 1
        else:
            n_waters = self.n_waters

        x = 0
        if self.remove_selfH:
            x = 1
        OHdistances = []
        for frame_ind in xrange(traj.n_frames):
            # compute H's based on closest n oxygens
            OHpairs = []
            for Oind in xrange(n_oxygens):
                water_inds = [traj.top.atom(i).residue.index for i in np.argsort(OOdistances[frame_ind, Oind])[x:(n_waters + 1)]]
                # NOTE: This will include the hydrogens on the same molecule as well

                OHpairs.extend([(Oind, a.index) for i in water_inds 
                                    for a in traj.top.residue(i).atoms if a.element.symbol == 'H'])

            tempD = md.compute_distances(traj[frame_ind], OHpairs).reshape((1, n_oxygens, 2 * (n_waters + 1 - x)))

            if self.sort_locally:
                # ugh.
                # this might not work with the remove_selfH stuff.
                d = np.array([np.concatenate([np.sort(tempD[0, oind, i:i+2]) for i in xrange(0, 2 * n_oxygens, 2)]) for oind in xrange(n_oxygens)])
                OHdistances.append(d)
            else:
                OHdistances.append(tempD[0])
                
        OHdistances = np.array(OHdistances)
        # right now, OHdistances is ordered by the water molecule's O-O distance
        # I can either sort them globally, or sort the pairs associated with each water molecule
        if not self.sort_locally:
            OHdistances.sort()
        XnewOH = OHdistances # don't need to worry about changing these

        XnewOO = copy.copy(OOdistances)
        # sort the last index
        XnewOO.sort() 
        
        if self.n_waters is None:
            XnewOO = XnewOO[:, :, 1:]
        else:
            XnewOO = XnewOO[:, :, 1:(self.n_waters + 1)]

        Xnew = np.concatenate([XnewOO, XnewOH], axis=2) # concatenate for each water

        return Xnew, OOdistances


