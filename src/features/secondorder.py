import numpy as np
import mdtraj as md
from .base import BaseTransformer
from .utils import get_square_distances
import copy


class SecondOrder(BaseTransformer):
    """
    Compute the second order distances in oxygens by including the 
    distances within the solvation shells around a water molecule

    Parameters
    ----------
    n_waters : int, optional
        use the n_waters closest waters
    sort : str {'local', 'global'}
        Sort the second order distances in one of two ways:
            - local: sort the second order distances ordered by
                the first order distances
            - global: sort the second order distances in order by
                increasing distance. I think this is a bad idea
    """
    def __init__(self, n_waters=None, sort='local'):

        if n_waters is None:
            self.n_waters = None
        else:
            self.n_waters = int(n_waters)

        if not sort.lower() in ['local', 'global']:
            raise ValueError("invalid sort.")

        self.sort_locally = False
        if sort.lower() == 'local':
            self.sort_locally = True

    
    def transform(self, traj):
        """
        Transform the trajectory

        Parameters
        ----------
        traj : mdtraj.Trajectory
            trajectory to transform
        
        Returns
        -------
        Xnew : np.ndarray
            transformed trajectory
        distances : np.ndarray
            distances between all of the oxygens
        """

        oxygens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'O'])
        distances = get_square_distances(traj, oxygens)

        n_oxygens = len(oxygens)

        if self.n_waters is None:
            n_waters = len(oxygens) - 1

        else:
            n_waters = self.n_waters

        Xnew = copy.copy(distances)
        Xnew.sort()
        Xnew = Xnew[:, :, 1:(n_waters + 1)]

        # now add the distances between the waters in the solvation shell
        closest_waters = np.argsort(distances)[:, :, 1:(n_waters + 1)]

        upper_diag_inds = np.array([(i, j) for i in xrange(n_waters) for j in xrange(i + 1, n_waters)])
        other_distances = []
        for frame_ind in xrange(traj.n_frames):
            temp = []
            D = distances[frame_ind]
            for Oind in xrange(n_oxygens):
                inds = closest_waters[frame_ind, Oind]
                templine = D[inds, :][:, inds]
                templine = templine[upper_diag_inds[:, 0], upper_diag_inds[:, 1]]
                temp.append(templine)

            other_distances.append(temp)

        other_distances = np.array(other_distances)

        Xnew = np.concatenate([Xnew, other_distances], axis=2)

        return Xnew, distances

