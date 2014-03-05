
import mdtraj as md
import numpy as np
import itertools
from time import time

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
    for k in xrange(2, n_terms + 1):
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


    

