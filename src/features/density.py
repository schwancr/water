
import numpy as np
from .base import BaseTranformer
import mdtraj as md


class GridDensity(BaseTransformer):
    """
    Class for discretizing a KDE around each water molecule. 
    The representation consists of creating a KDE for each
    frame in a trajectory. Then for each water molecule, 
    computing the density in neighboring voxels as given
    by the KDE (without the current water molecule's contribution)

    The voxels are averaged across 4 regions since the water 
    molecule has C2v symmetry

    Parameters
    ----------
    sigma : float
        KDE parameter for setting the standard deviation
        in the KDE
    radius : float
        Each representation will include points within some distance
        of the central Oxygen. 
    sigma_hydrogen : float, optional
        KDE parameter for the standard deviation in the 
        Hydrogen KDE. If None, then both Oxygen and Hydrogen
        will use ```sigma```
    n_pixels : int, optional
        number of pixels to use to define the representation

    """
    def __init__(self, sigma, radius, sigma_hydrogen=None,
                 n_pixels=25):

        self.oxy_sigma2 = np.float(sigma) ** 2

        if self.sigma_hydrogen is None:
            self.hyd_sigma2 = self.oxy_sigma2
        else:
            self.hyd_sigma2 = np.float(sigma_hydrogen) ** 2

        self.radius = np.float(radius)
        
        self.n_pixels = np.int(self.n_pixels)

    
    def _prep_kde(points, sigma2):
        """
        internal function for setting up the kde's
        """
        self._hyd_kde = scipy.stats.gaussian_kde(points.T, 
        
