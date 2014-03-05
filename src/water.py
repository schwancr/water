
import mdtraj as md
import numpy as np
import itertools
from time import time
from pyxutils import mult

def get_atom_distances(traj):
    """
    compute all atom distances for each frame in traj
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        trajectory to analyze

    Returns
    -------
    atom_distances : np.ndarray
        numpy array containing all atom distances
    atom_pairs : np.ndarray
        numpy array containing the atom pairs corresponding to
        each feature of atom_distances
    """

    n_atoms = traj.n_atoms

    atom_pairs = np.array([(i, j) for i in xrange(n_atoms) for j in xrange(i + 1, n_atoms)])

    atom_distances = md.compute_distances(traj, atom_pairs)

    return atom_distances, atom_pairs
   

def get_features_distances(atom_distances, atom_pairs, top_or_traj, atom_types=None, n_terms=10):
#def get_features(traj, atom_types=None, n_terms=10):
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

    unique_atoms = np.unique(atom_types)
    atom_lists = [np.where(atom_types==a)[0] for a in unique_atoms]
    features = [] # this will be a list that looks like a stair case. Ok?

    atom_distances = md.geometry.contact.squareform(np.square(atom_distances), atom_pairs)
    print atom_distances.shape
    # store the square of the distances

    all_combos = []
    sigma_sqr = 0.05 ** 2 # radius of an atom in nm
    C = 1. / (sigma_sqr)
    E = np.exp(- C * atom_distances)
    diag_ind = np.arange(E.shape[1])
    E[:, diag_ind, diag_ind] = 0.0
    temp = np.zeros(E.shape)
    for i in xrange(n_terms):
        # remove the diagonal so we don't count self distances
        k = i + 2 # first term is the two-body term
        combos = itertools.combinations_with_replacement(unique_atoms, k)
        # get k atom types
        current_features = []
        for c in combos:
            print "Working on types %s" % str(c)
            current_mat = np.zeros(E.shape)
            current_mat[:] = np.eye(E.shape[1])
            for j in range(len(c) - 1):
                t0 = time()
                temp *= 0.0 # reset the temp container
                type1 = np.where(atom_types==c[j])[0]
                type2 = np.where(atom_types==c[j + 1])[0]
                # need to only include distances associated with these pairs of points
                t1 = time()

                print "setup: %f" % (t1 - t0)
                all_ind = np.array(list(itertools.product(type1, type2)))
                t2 = time()

                print "iter: %f" % (t2 - t1)
                temp[:, all_ind[:,0], all_ind[:,1]] = E[:, all_ind[:, 0], all_ind[:, 1]]
                temp[:, diag_ind, diag_ind] = 0.0
                t3 = time()
                print "set: %f" % (t3 - t2)

                mult(current_mat, temp)
                #for l in xrange(current_mat.shape[0]):
                #    current_mat[l] = current_mat[l].dot(temp[l])
                t4 = time()
                print "multiply: %f" % (t4 - t3)

            current_mat[:, diag_ind, diag_ind] = 0.0
            features.append(current_mat.sum(2).sum(1))
            all_combos.append(c)

    return features, all_combos


def _moveto(mobile, target, traj):
    """
    move the mobile atom to the target atom's frame according to the PBCs in top
    """

    displacement = target - mobile
    projections = np.array([np.dot(displacement, v) for v in traj.unitcell_vectors.T])
        # I think the vectors are in the columns. I need to confirm that first
    n_periods_away = np.round(projections / traj.unitcell_lengths, 0)
    new_mobile = mobile + n_periods_away.dot(traj.unitcell_vectors.T) 

    return new_mobile
    

def _compute_sums(zipped_atoms, traj):
    """
    compute the sum of the exponential of the variance between sets of atoms
    
    This is really just a library function
    """
    C = 1.0
    running_sum = 0.0
    #for atom_set in zipped_atoms:
    #    # vvvv This is wrong.. "atom_set" are sets of indices, and I would need
    #    # to loop over all of the frames, which will be prohibitively expensive
    #    c_atom_set = np.array([_moveto(a, atom_set[0], traj) for a in atom_set])
    #    running_sum += np.exp(- C * (np.var(c_atom_set, axis=0).sum()))

    #return running_sum


    

