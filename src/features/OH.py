import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import mdtraj as md
from .utils import get_square_distances
import copy

class OH(BaseEstimator, TransformerMixin):
    """
    Compute the OO distances and sort them for each water molecule
    
    Parameters
    ----------
    n_waters : int, optional
        Limit the feature vectors to the closest n_waters. If None, 
        then all waters are included.
    """
    def __init__(self, n_waters=None):
        if n_waters is None:
            self.n_waters = n_waters
        else:
            self.n_waters = int(n_waters)


    def transform(self, traj):
        """
        Transform a trajectory into the OO features

        Parameters
        ----------
        traj : mdtraj.Trajectory

        Returns
        -------
        Xnew : np.ndarray
            sorted distances for each water molecule
        distances : np.ndarray
            distances between each water molecule
        """
        oxygens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'O'])
        hydrogens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'H'])

        pairsOH = np.array([(i, j) for i in oxygens for j in hydrogens])
        distances = md.compute_distances(traj, pairsOH)
        distances = distances.reshape((traj.n_frames, len(oxygens), len(hydrogens)))

        Xnew = copy.copy(distances)
        Xnew.sort()

        distances = get_square_distances(traj, oxygens)

        if not self.n_waters is None:
            Xnew = Xnew[:, :, :(2 * self.n_waters)]

        return Xnew, distances

