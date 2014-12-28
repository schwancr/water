import numpy as np
from .base import BaseTransformer
import mdtraj as md
from .utils import get_square_distances
import copy

class Dipole(BaseTransformer):
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
        Transform a trajectory into the dipole features

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
        
        hydrogens = np.array([[a.index for a in traj.top.atom(Oind).residue.atoms if a.element.symbol == 'H'] for Oind in oxygens])
        # this ^^^ is the same as that vvv
        #hydrogens = []
        #for Oind in oxygens:
        #    res = traj.top.atom(Oind).residue
        #    for atom in res.atoms:
        #        temp = []
        #        if atom.element.symbol == 'H':
        #            temp.append(atom.index)
        #        hydrogens.append(temp)
        #hydrogens = np.array(hydrogens)

        distances = get_square_distances(traj, oxygens)

        Xnew = copy.copy(distances)
        Xnew.sort()

        if self.n_waters is None:
            Xnew = Xnew[:, :, 1:]
            n_waters = len(oxygens) - 1
        else:
            Xnew = Xnew[:, :, 1:(self.n_waters + 1)]
            n_waters = self.n_waters

        raise Exception("I'm pretty sure there's a bug here where the line below should do the mean on axis=1")
        dipoles = traj.xyz[:, oxygens, :] - np.mean(traj.xyz[:, hydrogens, :], axis=2)
        dipoles /= np.sqrt(np.square(dipoles).sum(2, keepdims=True)) # make them unit vectors

        dipole_dots = []
        for frame_ind in xrange(traj.n_frames):
            temp = []
            for water in xrange(len(oxygens)):
                water_inds = np.argsort(distances[frame_ind, water])[1:(n_waters + 1)]
                temp.append(np.dot(dipoles[frame_ind, water], dipoles[frame_ind, water_inds].T))
            dipole_dots.append(temp)

        dipole_dots = np.array(dipole_dots)
        Xnew = np.concatenate([Xnew, dipole_dots], axis=2)

        return Xnew, distances

