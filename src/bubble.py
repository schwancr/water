
import numpy as np
import itertools

def get_features(atom_distances, atom_types, n_terms=2, bubble_radius=1.0):
    """
    Parameters
    ----------
    atom_distances : np.ndarray
        a single frame's atom-atom distances
    atom_types : np.ndarray
        the atom type for each atom in the atom distance matrix
    n_terms : int, optional
        number of terms
    bubble_radius : float, optional
        radius to use when calculating the terms here. To be 
        honest it would be better to actually look at atoms
        that are all within this radius of ALL atoms in the
        k-tuple. But this isn't precisely how I have it implemnted        

    Returns
    -------
    features
    """

    n_atoms = len(atom_types)
    n_types = len(np.unique(atom_types))

    C = lambda k : 0.5 / k / (0.05)**2

    E_all = np.exp(- C(2) * atom_distances)

    max_num = len(list(itertools.combinations_with_replacement(np.arange(n_types), n_terms + 2)))
    features = np.zeros((n_terms, max_num)) - 1 
    # Eventually, we should store this as a sparse matrix for cases 
    # when there are a lot of atom types

    all_atom_lists = []
    for i in range(n_types):
        all_atom_lists.append(np.where(atom_types == i)[0])
    
    all_features = []
    all_combos = []
    for k in xrange(2, n_terms + 2):

        if k != 2:
            E = np.power(E_all, (C(k) / C(2)))

        else:
            E = E_all

        kbody_features = []
        kbody_combos = []
        for atom_type_combos in itertools.combinations_with_replacement(np.arange(n_types), k):
            #print "k=%d, %s" % (k, str(atom_type_combos))
            kbody_combos.append(atom_type_combos)
            kbody_features.append(0.0)
            # will update as we go.
            N = len(all_atom_lists[atom_type_combos[0]])
            n = 0
            for aind in all_atom_lists[atom_type_combos[0]]:
                n += 1
                #print "\tworking on atom %d / %d" % (n, N)
                # these are the atoms of this first type
                neighbors = np.where(atom_distances[aind] < bubble_radius)[0]
                if len(neighbors) < k:
                    continue

                atom_lists = []
                for i in xrange(n_types):
                    atom_lists.append(np.where(atom_types[neighbors] == i)[0])

                # If there's an atom type we need that we don't have in this bubble
                # then just go to the next atom. It turns out that itertools.product
                # returns an empty list if one of the iterables is empty!
                for temp_atom_inds in itertools.product(*[atom_lists[i] for i in atom_type_combos[1:]]):
                    #atom_inds is a tuple corresponding to the index in neighbors
                    atom_inds = [aind] + list(neighbors[list(temp_atom_inds)])

                    if len(set(atom_inds)) != len(atom_inds):
                        continue

                    temp = 1.0
                    for i in xrange(k):
                        for j in xrange(i + 1, k):
                            temp *= E[atom_inds[i], atom_inds[j]]

                    kbody_features[-1] += temp

        all_features.extend(kbody_features)
        all_combos.extend(kbody_combos)

    return all_features, all_combos
