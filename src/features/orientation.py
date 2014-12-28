import numpy as np
from .base import BaseTransformer
import mdtraj as md
from .utils import get_square_distances
import copy

class Orientation(BaseTransformer):
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


        OHvectors = traj.xyz[:, oxygens, :] - traj.xyz[:, hydrogens, :].mean(axis=2)
        # this is right because hydrogens is shaped as (n_atoms, 2)
        OHmags = np.sqrt(np.square(OHvectors).sum(axis=2, keepdims=True))
        OHvectors = OHvectors / OHmags

        norm_vectors = traj.xyz[:, hydrogens[:, 1], :] - traj.xyz[:, hydrogens[:, 0]]
        # because of symmetry, there is a degeneracy, but I will account
        # for that by computing absolute dot products as opposed to
        # the signed one
        norm_mags = np.sqrt(np.square(norm_vectors).sum(axis=2, keepdims=True))

        # gram-schmidt up in here
        norm_vectors = norm_vectors - np.sum(OHvectors * norm_vectors, axis=2, keepdims=True) * OHvectors

        # I don't think this magnitude is very meaningful
        temp_mags = np.sqrt(np.square(norm_vectors).sum(axis=2, keepdims=True))
        
        norm_vectors = norm_vectors / temp_mags

        # Now compute the angle between the two reference frames
        # but for the normal vectors, I want the absolute value of
        # the dot product, due to the water molecule's symmetry

        norm_mags = norm_mags.squeeze()
        OHmags = OHmags.squeeze()

        OH_dots = []
        norm_dots = []
        neighbor_OHmags = []
        neighbor_norm_mags = []
        for frame_ind in xrange(traj.n_frames):

            OH_temp = []
            norm_temp = []
            neighbor_OH_temp = []
            neighbor_norm_temp = []
            for water in xrange(len(oxygens)):

                water_inds = np.argsort(distances[frame_ind, water])[1:(n_waters + 1)]

                OH_temp.append(np.dot(OHvectors[frame_ind, water], 
                                      OHvectors[frame_ind, water_inds].T))

                norm_temp.append(np.abs(np.dot(norm_vectors[frame_ind, water], 
                                               norm_vectors[frame_ind, water_inds].T)))

                neighbor_OH_temp.append(np.concatenate([OHmags[frame_ind, water : (water + 1)], 
                                                        OHmags[frame_ind, water_inds]]))

                neighbor_norm_temp.append(np.concatenate([norm_mags[frame_ind, water : (water + 1)], 
                                                          norm_mags[frame_ind, water_inds]]))

            OH_dots.append(OH_temp)
            norm_dots.append(norm_temp)
            neighbor_OHmags.append(neighbor_OH_temp)
            neighbor_norm_mags.append(neighbor_norm_temp)

        OH_dots = np.array(OH_dots)
        norm_dots = np.array(norm_dots)
        neighbor_OHmags = np.array(neighbor_OHmags)
        neighbor_norm_mags = np.array(neighbor_norm_mags)

        Xnew = np.concatenate([Xnew, OH_dots, norm_dots], axis=2) 
                              # neighbor_OHmags, neighbor_norm_mags], axis=2)

        return Xnew, distances

