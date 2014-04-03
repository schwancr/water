
import mdtraj as md
import scipy.special
import tables
import numpy as np
cimport numpy as np
cimport cython
import itertools
from time import time

DTYPE_FLOAT = np.float
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t
ctypedef np.float_t DTYPE_FLOAT_t

def get_features_distances(filehandler, np.ndarray[DTYPE_INT_t, ndim=2] atom_pairs,
                           np.ndarray[DTYPE_INT_t, ndim=1] atom_types, int n_terms):
    """
    compute the features of the trajectories by computing the overlap integrals
    between k atoms and then averaging over the atom types. This makes sense in
    an example:

    Let's say there are two atom types in my trajectory (e.g. water H and O), 
    then for the 5th term in the expansion I would calculate the average 
    distances for five atoms to be, and then I would average over the atom types
    which would mean I would have individual features for the overlaps between:
    
    0 H and 5 O's
    1 H and 4 O's
    2 H's and 3 O's
    3 H's and 2 O's
    4 H's and 1 O's
    5 H's and 0 O's

    This gets more complicated as the number of atom types gets bigger, but we
    can extrapolate.

    Now determining equilivalent atoms is a bit tricky... But for now, I'm just
    computing this by saying the atoms are equal if they are the same element
    and in the same residue name. This works for water but obviously fails for
    other things. In the future, we can compute this with the topology, since we
    know all the bonds and things. It's not too hard, but it's a bit of code
    I don't want to write right now.

    Parameters
    ----------
    atom_distances : np.ndarray
        atom distances for each frame to analyze shape: (n_frames, n_atompairs)
    atom_pairs : np.ndarray
        pairs of atoms corresponding to the distances in atom_distances
        shape: (n_atompairs, 2)
    top_or_traj : mdtraj.Topology or mdtraj.Trajectory with a top attr
        topology information for determining whether two atoms are the same
    atom_types : np.ndarray, optional
        provide the equivalent atom types such that each equivalent atom is assigned
        the same integer in this array shape: (n_atoms,)
    n_terms : int, optional
        number of terms to compute in the expansion. This has a hard limit at the 
        number of atoms you have, but realistically you should use something 
        much smaller (how many atoms can really overlap at any one time anyway?)
    
    Returns
    -------
    features : np.ndarray
        shape: (n_frames, n_features) where the number of features will depend on
        the number of atom types as well as the number of atom pairs computed.
    """

    cdef unsigned int n_types = np.max(atom_types) + 1
    cdef unsigned int max_natoms = np.max(np.bincount(atom_types))
    cdef unsigned int n_atoms = len(atom_types)
    cdef unsigned int n_frames = filehandler.root.arr_0.shape[0]
    cdef np.ndarray[DTYPE_INT_t, ndim=1] unique_atoms = np.arange(n_types, dtype=DTYPE_INT)
    cdef unsigned int n_features = np.sum([len(list(itertools.combinations_with_replacement(unique_atoms, k))) for k in xrange(2, 2 + n_terms)])

    cdef unsigned int k
    cdef unsigned int L
    cdef float C

    cdef unsigned int feature_ind = 0
    cdef unsigned int chunk_size = 10

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=3] E = np.empty((chunk_size, n_atoms, n_atoms))
    # temporary storage for the exponential of the distances

    cdef np.ndarray[DTYPE_INT_t, ndim=2] atom_lists = -1 * np.ones((n_types, max_natoms), dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_INT_t, ndim=2] all_combos = -1 * np.ones((n_features, n_terms), dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] features = np.zeros((n_frames, n_features), dtype=DTYPE_FLOAT)

    print n_features, (features.shape[0], features.shape[1])

    # first populate the atom_lists
    for k in range(n_types):
        L = np.where(atom_types == k)[0].shape[0]
        atom_lists[k, :L] = np.where(atom_types == k)[0]

    sigma_sqr = 0.05 ** 2 # radius of an atom in nm
    feature_ind = 0
    for k in range(2, n_terms + 2):
        C = 0.25 / (k**2) / (sigma_sqr)
        print "C=%f" % C
        combos = itertools.combinations_with_replacement(unique_atoms, k)
        # get k atom types
        for c in combos:
            print "Working on types %s" % str(c)
            # c is a tuple containing combinations of atom types
            # but we need to sum over a bunch of atom indices
            a = time()

            for atom_inds in itertools.product(*[atom_lists[i] for i in c]):
                if len(atom_inds) != len(set(atom_inds)):
                    continue
                # atom_inds is a set of k atom indices
                # we need to look through all possible combinations of these atoms
                
                for chunk_ind in xrange(0, n_frames / chunk_size + 1):

                    #E = md.geometry.contact.squareform(filehandler.root.arr_0[chunk_ind * chunk_size : (chunk_ind + 1) * chunk_size], atom_pairs).astype(DTYPE_FLOAT)
                    E = filehandler.root.arr_0[chunk_ind * chunk_size : (chunk_ind + 1) * chunk_size].astype(DTYPE_FLOAT)
                    E = np.exp(- C * E)

                    if k == 2:
                        features[chunk_ind * chunk_size : (chunk_ind + 1) * chunk_size, feature_ind] += E[:, atom_inds[0], atom_inds[1]]

                    elif k == 3:
                        features[chunk_ind * chunk_size : (chunk_ind + 1) * chunk_size, feature_ind] += E[:, atom_inds[0], atom_inds[1]] * E[:, atom_inds[0], atom_inds[2]] * E[:, atom_inds[1], atom_inds[2]]

                    else:
                        pairs = np.array(list(itertools.combinations(atom_inds, 2)))
                        features[chunk_ind * chunk_size : (chunk_ind + 1) * chunk_size, feature_ind] += np.product(E[:, pairs[:, 0], pairs[:, 1]], axis=1)

            all_combos[feature_ind, :len(c)] = c

            feature_ind += 1
                
            b = time()
            print "finish: %.4f" % (b - a)

    return features, all_combos

