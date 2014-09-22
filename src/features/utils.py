import mdtraj as md
import numpy as np

def get_square_distances(traj, aind=None):
    """
    The the atom distances in for a subset of a trajectory and 
    return it as a square distance matrix

    Parameters
    ----------
    traj : mdtraj.Trajectory 
        trajectory object
    aind : np.ndarray, optional
        atom indices to compute distances between

    Returns
    -------
    distances : np.ndarray, shape = [traj.n_frames, aind.shape[0], aind.shape[0]]
        distances between the atoms in aind
    """

    if aind is None:
        aind = np.arange(traj.n_atoms)

    pairs_ind = np.array([(i, j) for i in xrange(len(aind)) for j in xrange(i + 1, len(aind))])
    pairs = np.array([(aind[i], aind[j]) for i, j in pairs_ind])

    print traj.n_frames
    print pairs.shape[0]
    distances = md.compute_distances(traj, pairs)
    distances = md.geometry.squareform(distances, pairs_ind)

    return distances

