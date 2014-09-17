
import numpy as np
import mdtraj as md
from sklearn.base import TransformerMixin, BaseEstimator
import copy
from simtk.openmm import app
from simtk import openmm as mm
from simtk import unit
import time


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


class OH(BaseEstimator, TransformerMixin):
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
        hydrogens = np.array([i for i in xrange(traj.n_atoms) if traj.top.atom(i).element.symbol == 'H'])

        pairsOH = np.array([(i, j) for i in oxygens for j in hydrogens])
        distances = md.compute_distances(traj, pairsOH)
        distances = distances.reshape((traj.n_frames, len(oxygens), len(hydrogens)))

        Xnew = copy.copy(distances)
        Xnew.sort()

        distances = get_square_distances(traj, oxygens)

        if not self.n_waters is None:
            Xnew = Xnew[:, :, :(2 * self.n_waters)]

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
    def __init__(self, n_waters=None, sortH='local', remove_selfH=False):
        if n_waters is None:
            self.n_waters = n_waters
        else:
            self.n_waters = int(n_waters)

        if not sortH.lower() in ['local', 'global']:
            raise ValueError("%s not in %s" % (sortH, str(['local', 'global'])))

        self.sort_locally = False
        if sortH.lower() == 'local':
            self.sort_locally = True

        self.remove_selfH = bool(remove_selfH)

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

        x = 0
        if self.remove_selfH:
            x = 1
        OHdistances = []
        for frame_ind in xrange(traj.n_frames):
            # compute H's based on closest n oxygens
            OHpairs = []
            for Oind in xrange(n_oxygens):
                water_inds = [traj.top.atom(i).residue.index for i in np.argsort(OOdistances[frame_ind, Oind])[x:(n_waters + 1)]]
                # NOTE: This will include the hydrogens on the same molecule as well

                OHpairs.extend([(Oind, a.index) for i in water_inds 
                                    for a in traj.top.residue(i).atoms if a.element.symbol == 'H'])

            tempD = md.compute_distances(traj[frame_ind], OHpairs).reshape((1, n_oxygens, 2 * (n_waters + 1 - x)))

            if self.sort_locally:
                # ugh.
                # this might not work with the remove_selfH stuff.
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


class Dipole(BaseEstimator, TransformerMixin):
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


class InducedDipole(BaseEstimator, TransformerMixin):
    """
    Compute the OO distances and sort them for each water molecule
    Then compute the induced dipole on neighboring waters, and 
    compute the dot product between the central molecules induced
    dipole and its neighbors
    
    Parameters
    ----------
    n_waters : int, optional
        Limit the feature vectors to the closest n_waters. If None, 
        then all waters are included.
    """
    def __init__(self, n_waters=None, topology=None):

        if hasattr(topology, 'to_openmm'):
            self._top = topology.to_openmm()
        else:
            self._top = topology

        if n_waters is None:
            self.n_waters = n_waters
        else:
            self.n_waters = int(n_waters)

        forcefield = app.ForceField('iamoeba.xml')

        self._top.setUnitCellDimensions([10, 10, 10])
        system = forcefield.createSystem(self._top, nonbondedMethod=app.PME)
        self._amoeba_force = [f for f in system.getForces() if isinstance(f, mm.openmm.AmoebaMultipoleForce)][0]
        integrator = mm.LangevinIntegrator(3000*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
        platform = mm.Platform.getPlatformByName('Reference')

        self._sim = app.Simulation(self._top, system, integrator, platform)
        self._context = self._sim.context


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

        a = time.time()
        distances = get_square_distances(traj, oxygens)

        Xnew = copy.copy(distances)
        Xnew.sort()

        if self.n_waters is None:
            Xnew = Xnew[:, :, 1:]
            n_waters = len(oxygens) - 1
        else:
            Xnew = Xnew[:, :, 1:(self.n_waters + 1)]
            n_waters = self.n_waters

        b = time.time()
        distance_time = b - a

        induced_time = 0
        rotate_time = 0

        # this is probably pretty slow ... 
        alldipoles = []
        for i in xrange(traj.n_frames):
            a = time.time()
            dipoles = []
            pos = traj.xyz[i]
            box = traj.unitcell_vectors[i]
            self._context.setPeriodicBoxVectors(box[0], box[1], box[2])
            self._context.setPositions(pos)

            induced = self._amoeba_force.getInducedDipoles(self._context)

            b = time.time()

            for ai in xrange(traj.n_atoms):
                params = self._amoeba_force.getMultipoleParameters(ai)

                molec_dipole = np.array(params[1])
                atomZ = params[4]
                atomX = params[5]

                posX = traj.xyz[i, atomX]
                posZ = traj.xyz[i, atomZ]
                
                myPos = traj.xyz[i, ai]

                Z = posZ - myPos
                X = posX - myPos

                Z /= np.sqrt(np.square(Z).sum())
                X /= np.sqrt(np.square(X).sum())

                Z = Z + X
                Z /= np.sqrt(np.square(Z).sum())
                
                X = X - X.dot(Z) * Z
                X /= np.sqrt(np.square(X).sum())

                Y = np.cross(Z, X)

                lab_dipole = molec_dipole.dot(np.vstack([X, Y, Z]))

                dipoles.append(lab_dipole + induced[ai])

            c = time.time()

            induced_time += b - a
            rotate_time += c - b

            alldipoles.append(dipoles)

        dipoles = np.array(alldipoles)
        print dipoles.shape
        # add the dipoles for each atom
        dipoles = dipoles[:, ::3] + dipoles[:, 1::3] + dipoles[:, 2::3]

        a = time.time()
        dipole_dots = []
        for frame_ind in xrange(traj.n_frames):
            temp = []
            for water in xrange(len(oxygens)):
                water_inds = np.argsort(distances[frame_ind, water])[1:(n_waters + 1)]
                temp.append(np.dot(dipoles[frame_ind, water], dipoles[frame_ind, water_inds].T))
                print temp[-1]
            dipole_dots.append(temp)

        dipole_dots = np.array(dipole_dots)
        Xnew = np.concatenate([Xnew, dipole_dots], axis=2)

        b = time.time()

        print "distances : %.2f | induced dipoles : %.2f | rotate frame : %.2f | dot products : %.2f" % (distance_time, induced_time, rotate_time, b - a)

        return Xnew, distances

