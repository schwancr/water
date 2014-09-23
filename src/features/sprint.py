
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import mdtraj as md

class Sprint(BaseEstimator, TransformerMixin):
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
        dec_vals_inds = np.argsort(all_vals, axis=1)

        axis0 = np.arange(all_vals.shape[0])
        axis1 = np.arange(all_vecs.shape[1])

        all_vals = all_vals[axis0, dec_vals_inds]
        all_vecs = all_vecs[axis0, axis1, dec_vals_inds]

        return all_vecs, OOdistances
