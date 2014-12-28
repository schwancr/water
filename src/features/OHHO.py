import numpy as np
from .base import BaseTransformer
import mdtraj as md
from .utils import get_square_distances
import copy

class OHHO(BaseTransformer):
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
        oxygens = np.array([i for i in xrange(traj.n_atoms) 
                            if traj.top.atom(i).element.symbol == 'O'])
        hydrogens = np.array([i for i in xrange(traj.n_atoms) 
                              if traj.top.atom(i).element.symbol == 'H'])

        OOdistances = get_square_distances(traj, oxygens)
        Xnew = copy.copy(OOdistances)

        OOsorted_inds = np.argsort(Xnew) # sort the last axis
        OOsorted_inds = OOsorted_inds[:, :, 1:(self.n_waters + 1)]

        frame_inds = np.arange(traj.n_frames).reshape((-1, 1, 1))
        water_inds = np.arange(len(oxygens)).reshape((1, -1, 1))
        # have to reshape it so that the broadcasting works

        Xnew = Xnew[frame_inds, water_inds, OOsorted_inds]
        
        # these pairs are wrong b/c i'm an idiot... Actually I don't know
        # why they're wrong, but they are wrong
        pairs = np.array([(i, j) for i in oxygens for j in hydrogens])

        OHdistances = md.compute_distances(traj, pairs)
        OHdistances = OHdistances.reshape((traj.n_frames, len(oxygens), len(hydrogens)), order='C')

        H1 = []
        H2 = []
        for oind in oxygens:
            hinds = [a.index for a in traj.top.atom(oind).residue.atoms 
                     if a.element.symbol == 'H']
            H1.append(np.where(hydrogens == hinds[0])[0][0])
            H2.append(np.where(hydrogens == hinds[1])[0][0])
        H1 = np.array(H1)
        H2 = np.array(H2)

        # something is wrong. There are a bunch of zeros...
        # in both the OH and HO parts.

        OHpart1 = OHdistances[frame_inds, water_inds, H1[OOsorted_inds]]
        OHpart2 = OHdistances[frame_inds, water_inds, H2[OOsorted_inds]]

        OHmin = np.min([OHpart1, OHpart2], axis=0)
        OHmax = np.max([OHpart1, OHpart2], axis=0)
        # get the min distance from each water to the hydrogens

        HOdistances = OHdistances.transpose((0, 2, 1))
        H1Odistances = HOdistances[:, H1, :]
        H2Odistances = HOdistances[:, H2, :]

        HOpart1 = H1Odistances[frame_inds, water_inds, OOsorted_inds]
        HOpart2 = H2Odistances[frame_inds, water_inds, OOsorted_inds]

        HOmin = np.min([HOpart1, HOpart2], axis=0)
        HOmax = np.max([HOpart1, HOpart2], axis=0)

        Xnew = np.concatenate([Xnew, OHmin, OHmax, HOmin, HOmax], axis=2)

        return Xnew, OOdistances
        

