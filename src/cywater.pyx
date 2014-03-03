
import mdtraj as md
import numpy as np
cimport numpy as np
cimport cython
import itertools
from time import time

DTYPE_FLOAT = np.float
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t
ctypedef np.float_t DTYPE_FLOAT_t

def get_features_distances(np.ndarray[DTYPE_FLOAT_t, ndims=2] atom_distances, 
                           np.ndarray[DTYPE_INT_t, ndims=1] atom_types, 
                           int n_terms):
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

    cdef unsigned int num_types = np.max(atom_types) + 1
    cdef unsigned int max_natoms = np.max(np.bincount(atom_types))
    cdef unsigned int k
    cdef unsigned int L
    cdef float C
    cdef np.ndarray[DTYPE_FLOAT_t, ndims=2] E
    cdef np.ndarray[DTYPE_INT_t, ndims=1] unique_atoms = np.arange(atom_distances.shape[0], dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_INT_t, ndims=2] atom_lists = -1 * np.ones((num_types, max_natoms), dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_INT_t, ndims=2] all_combos = -1 * np.ones((num_features, n_terms), dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndims=1] features = -1 * np.ones(num_features, dtype=DTYPE_FLOAT)
    # first populate the atom_lists

    for k in range(num_types):
        L = np.where(atom_types == k)[0].shape[0]
        atom_lists[k, :L] = np.where(atom_types == k)[0]

    sigma_sqr = 0.05 ** 2 # radius of an atom in nm
    for k in range(2, n_terms + 1):
        C = 0.25 / (k**2) / (sigma_sqr)
        E = np.exp(- C * atom_distances)
        print "C=%f" % C
        combos = itertools.combinations_with_replacement(unique_atoms, k)
        # get k atom types
        current_features = []
        for c in combos:
            print "Working on types %s" % str(c)
            # c is a tuple containing combinations of atom types
            # but we need to sum over a bunch of atom indices
            a = time()
            temp = itertools.product(*[atom_lists[i] for i in c])

            running_sum = 0.0
            for i in temp:
                if len(i) != len(np.unique(i)):
                    continue

                pairs = np.array(list(itertools.combinations(i, 2)))

                running_sum += np.product(E[:, pairs[:, 0], pairs[:, 1]], axis=1)
                
            #zipped_atom_pairs = np.concatenate([list(itertools.combinations(i, 2)) for i in temp if len(i) == len(np.unique(i))])
            b = time()
            print "finish: %.4f" % (b - a)
            #temp_exp_terms = np.exp(- C * atom_distances[:, zipped_atom_pairs[:, 0], zipped_atom_pairs[:, 1]])
            #print "exponentiate: %.4f" % (c - b)
            #temp_exp_terms = temp_exp_terms.reshape((len(atom_distances), -1, k))
            #d = time()
            #print "reshape: %.4f" % (d - c)
            #feature = np.product(temp_exp_terms, axis=2).sum(axis=1)
            feature = running_sum

            features.append(np.float(feature))
            all_combos.append(c)
            e = time()

    return features, all_combos

