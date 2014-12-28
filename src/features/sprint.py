
from .base import BaseTransformer
import numpy as np
import mdtraj as md
from .utils import get_square_distances

class Sprint(BaseTransformer):
    r"""
    Use the SPRINT representation from

    [1] Pietrucci, F and Andreoni, W. Graph Theory Meets
        Ab Initio Molecular Dynamics: Atomic Structures
        and Transformations at the Nanoscale. (2011) PRL,
        504, pp 085504.

    There is a slight adaptation, since we will define a
    bond between waters as the minimum O-H and H-O distances

    Parameters
    ----------
    r0 : float
        radius to use as the 'typical' (hydrogen) bond radius
    n : int, optional
        numerator exponent in the representation (see below)
    m : int, optional
        denominator exponent in the representation (see below)

    Notes
    -----
    The representation consists of the eigenvector of an
    adjacency matrix with entries corresponding to:
    
    .. math ::

        a_{ij} = \frac{1 - (r_{ij} / r0)^n}{1 - (r_{ij} / r0)^m}
    """
    def __init__(self, r0, n=6, m=12):
        self.r0 = np.float(r0)
        self.n = np.int(n)
        self.m = np.int(m)

    def transform(self, traj):
        """
        Transform a trajectory into its SPRINT representations

        Parameters
        ----------
        traj : mdtraj.Trajectory
            trajectory containing 
        """
        oxygens = np.array([a.index for a in traj.top.atoms if a.element.symbol == 'O'])
        OOdistances = get_square_distances(traj, oxygens)
        
        numer = 1 - np.power(OOdistances / self.r0, self.n)
        denom = 1 - np.power(OOdistances / self.r0, self.m)

        adjacencies = numer / denom

        all_vals, all_vecs = np.linalg.eig(adjacencies)
        dec_vals_inds = np.argsort(all_vals, axis=1)[:, ::-1]

        axis0 = np.arange(all_vals.shape[0])
        axis1 = np.arange(all_vecs.shape[1])

        all_vals = all_vals[axis0.reshape((-1, 1)), dec_vals_inds]
        all_vecs = all_vecs[axis0.reshape((-1, 1, 1)), axis1.reshape((1, -1, 1)), dec_vals_inds.reshape((dec_vals_inds.shape[0], 1, dec_vals_inds.shape[1]))]

        # need to reconcile the changing signs between frames
        # really these spectral representations are a bit wierd 
        # to be using...
        for i in xrange(all_vecs.shape[0]):
            for j in xrange(all_vecs.shape[2]):
                sgn = np.sign(all_vecs[i, np.abs(all_vecs[i, :, j]).argmax(), j])
                all_vecs[i, :, j] *= sgn

        return all_vecs, OOdistances
