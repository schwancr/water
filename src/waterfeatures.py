
import numpy as np
import mdtraj as md
from sklearn.base import TransformerMixin, BaseEstimator
import copy


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
    
# featurizers:

class OO(BaseEstimator, TransformerMixin):
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
        oxygens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'O'])

        distances = get_square_distances(traj, oxygens)
        Xnew = copy.copy(distances)
        Xnew.sort()

        if self.n_waters is None:
            Xnew = Xnew[:, :, 1:]
        else:
            Xnew = Xnew[:, :, 1:(self.n_waters + 1)]

        return Xnew, distances


class OOH(BaseEstimator, TransformerMixin):
    """
    Compute the O-O and O-H distances for every water molecule

    Each water vector will look like:
        [d(O1, O2), d(O1, O3), ..., d(O1, ON), 
            d(O1, H2), d(O1, H2), ..., d(O1, H(2N))]

    Parameters
    ----------
    n_waters : int, optional
        Limit the feature vectors to the closest n_waters.
        If None, then all waters are included.
    sortH : str {'local', 'global'}
        The distance from each oxygen to it's neighboring
        hydrogens can be sorted in one of two ways:
            - local: The vector is sorted by the O-O distances
                and for each water, the two O-H distances are
                sorted by increasing distance
            - global: The O-H distances are sorted by 
                increasing distance. The Hydrogens are then
                disassociated from their water atom 
    """
    def __init__(self, n_waters=None, sortH='local'):
        if n_waters is None:
            self.n_waters = n_waters
        else:
            self.n_waters = int(n_waters)

        if not sortH.lower() in ['local', 'global']:
            raise ValueError("%s not in %s" % (sortH, str(['local', 'global'])))

        self.sort_locally = False
        if sortH.lower() == 'local':
            self.sort_locally = True

    def transform(self, traj):
        """
        Transform a trajectory.
        
        Parameters
        ----------
        traj : mdtraj.Trajectory
            trajectory to compute distances for

        Returns
        -------
        Xnew : np.ndarray
            distances for each water molecule
        distances : np.ndarray
            distance between each water molecule in the simulation
        """

        oxygens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'O'])
        hydrogens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'H'])
        n_oxygens = len(oxygens)

        OOdistances = get_square_distances(traj, oxygens)

        if self.n_waters is None:
            n_waters = n_oxygens - 1
        else:
            n_waters = self.n_waters

        OHdistances = []
        for frame_ind in xrange(traj.n_frames):
            # compute H's based on closest n oxygens
            OHpairs = []
            for Oind in xrange(n_oxygens):
                water_inds = [traj.top.atom(i).residue.index for i in np.argsort(OOdistances[frame_ind, Oind])[:(n_waters + 1)]]
                # NOTE: This will include the hydrogens on the same molecule as well

                OHpairs.extend([(Oind, a.index) for i in water_inds 
                                    for a in traj.top.residue(i).atoms if a.element.symbol == 'H'])

            tempD = md.compute_distances(traj[frame_ind], OHpairs).reshape((1, n_oxygens, 2 * (n_waters + 1)))

            if self.sort_locally:
                # ugh.
                d = np.array([np.concatenate([np.sort(tempD[0, oind, i:i+2]) for i in xrange(0, 2 * n_oxygens, 2)]) for oind in xrange(n_oxygens)])
                OHdistances.append(d)
            else:
                OHdistances.append(tempD[0])
                
        OHdistances = np.array(OHdistances)
        # right now, OHdistances is ordered by the water molecule's O-O distance
        # I can either sort them globally, or sort the pairs associated with each water molecule
        if not self.sort_locally:
            OHdistances.sort()
        XnewOH = OHdistances # don't need to worry about changing these

        XnewOO = copy.copy(OOdistances)
        # sort the last index
        XnewOO.sort() 
        
        if self.n_waters is None:
            XnewOO = XnewOO[:, :, 1:]
        else:
            XnewOO = XnewOO[:, :, 1:(self.n_waters + 1)]

        Xnew = np.concatenate([XnewOO, XnewOH], axis=2) # concatenate for each water

        return Xnew, OOdistances


class SecondOrder(BaseEstimator, TransformerMixin):
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