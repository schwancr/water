import mdtraj as md
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from simtk.openmm import app
from simtk import openmm as mm
from simtk import unit
import time
import copy
from .utils import get_square_distances

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
        system = forcefield.createSystem(self._top, nonbondedMethod=app.PME, polarization='direct')
        self._amoeba_force = [f for f in system.getForces() if isinstance(f, mm.openmm.AmoebaMultipoleForce)][0]
        integrator = mm.LangevinIntegrator(3000*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
        platform = mm.Platform.getPlatformByName('Reference')

        self._sim = app.Simulation(self._top, system, integrator, platform)
        self._context = self._sim.context


    def _compute_dipoles(self, traj):
        """compute the dipoles given box vectors and coordinates
        """

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

        print "induced dipoles: %.2f | rotations: %.2f" % (induced_time, rotate_time)

        return np.array(alldipoles)


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

        dipoles = self._compute_dipoles(traj)

        # add the dipoles for each atom
        dipoles = dipoles[:, ::3] + dipoles[:, 1::3] + dipoles[:, 2::3]

        a = time.time()
        dipole_dots = []

        mags = np.sqrt(np.square(dipoles).sum(axis=2, keepdims=True))
        dipoles = dipoles / mags
        mags = mags.sum(2)

        dipole_cos = []
        dipole_sin = []
        dipole_mags = []
        for frame_ind in xrange(traj.n_frames):
            tempdots = []
            tempcross = []
            tempmags = []
            for water in xrange(len(oxygens)):
                water_inds = np.argsort(distances[frame_ind, water])[1:(n_waters + 1)]
                tempdots.append(np.dot(dipoles[frame_ind, water], dipoles[frame_ind, water_inds].T))
                tempcross.append(np.array([np.cross(dipoles[frame_ind, water], dipoles[frame_ind, i]) for i in water_inds]))
                tempmags.append(mags[frame_ind, water_inds])

            dipole_cos.append(tempdots)
            dipole_sin.append(np.sqrt(np.square(tempcross).sum(axis=2)))
            dipole_mags.append(tempmags)

        dipole_cos = np.array(dipole_cos)
        dipole_sin = np.array(dipole_sin)
        dipole_mags = np.array(dipole_mags)
        Xnew = np.concatenate([Xnew, dipole_mags, dipole_cos, dipole_sin], axis=2)

        b = time.time()

        print "distances : %.2f | angle calculation : %.2f" % (distance_time, b - a)

        return Xnew, distances

