import numpy as np
from .base import BaseTransformer
import mdtraj as md
from .utils import get_square_distances
import copy
import IPython

class OOshells(BaseTransformer):
    """
    Compute the OO distances and sort them for each water molecule.
    Then each water molecule is composed of the distance to each of
    the waters in its many solvation shells
    
    Parameters
    ----------
    n_waters : int, optional
        Limit the feature vectors to the closest n_waters. If None, 
        then all waters are included.
    """
    def __init__(self, n_shells=2, n_per_shell=4):
        if n_shells != 2:
            print "I only have 2 shells implemented"
        self.n_shells = 2
        self.n_per_shell = n_per_shell
        

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

        distances = get_square_distances(traj, oxygens)
        Xnew = copy.copy(distances)
        Xnew.sort()
        Xnew = Xnew[:, :, 1:self.n_per_shell]
        # this is the first solvation shell for each water, we will now add
        # to each water's representation based on the first solvation shell
        # of the water's first solvation shell

        sorted_waters = np.argsort(distances, axis=-1)
        # sorted_waters[t, i, k] contains the k'th closest water index to water i at time t
        # k==0 is clearly i

        ind0 = np.array([np.arange(Xnew.shape[0])] * Xnew.shape[1]).T

        Xnew0 = copy.copy(Xnew)

        for k in xrange(1, self.n_per_shell + 1):
            Xnew = np.concatenate([Xnew, Xnew0[ind0, sorted_waters[:, :, k]]], axis=2)

        return Xnew, distances



