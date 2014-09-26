import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import mdtraj as md
from .utils import get_square_distances
import copy
from joblib import Parallel, delayed

def _analyze_frame(frame, min, max, n): 
    hists = np.array([np.histogram(frame[i], range=(min, max), bins=n)[0] for i in xrange(len(frame))])

    radii2 = np.square(np.linspace(min, max, n) + (max - min) / n / 2.0)
    
    return hists / radii2.reshape((1, -1))

class OOdensity(BaseEstimator, TransformerMixin):
    """
    Compute the OO distances and sort them for each water molecule
    
    Parameters
    ----------
    max_distance : float, optional
    """
    def __init__(self, min_distance=0.0, max_distance=2.5, n_bins=25, n_procs=1):
        self.min_distance = np.float(min_distance)
        self.max_distance = np.float(max_distance)
        self.n_bins = np.int(n_bins)

        self.n_procs = np.int(n_procs)


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

        Xnew = Parallel(self.n_procs)(delayed(_analyze_frame)(frame, self.min_distance, self.max_distance, self.n_bins) for frame in distances)
        #for i in xrange(traj.n_frames):
        #    frame = []
        #    for j in xrange(len(oxygens)):
        #        frame.append(np.histogram(distances[i, j])[0])
        #    Xnew.append(frame)
        Xnew = np.array(Xnew)

        return Xnew, distances



