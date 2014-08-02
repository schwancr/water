
import numpy as np
import itertools
import time

def get_features(atom_distances, atom_types, n_terms=2, bubble_radius=1.0,
    sigma=0.05):
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
    sigma : float, optional
        standard deviation of each atom's gaussian density

    Returns
    -------
    features : np.ndarray
        features is an array of shape (n_atoms, n_features), where
        n_features is given by the number of different k-tuples
        which comes from the number of atom types
    """

    n_atoms = len(atom_types)
    n_types = len(np.unique(atom_types))

    C = lambda k : 0.5 / k / float(sigma)**2

    E_all = np.exp(- C(2) * atom_distances)

    features = []  # will append features per atom every loop
    combo_strs = []
    for k in xrange(2, n_terms + 2):
        combos = itertools.combinations_with_replacement(np.unique(atom_types), k)
        for c in combos:
            combo_strs.append('.'.join([str(i) for i in np.sort(c)]))
    combo_strs = np.array(combo_strs)
    n_features = len(combo_strs)

    for aind in xrange(n_atoms):
        neighbors = np.where(atom_distances[aind] <= bubble_radius)[0]
        neighbors = neighbors[np.where(neighbors != aind)]

        atom_features = np.zeros(n_features)
        
        for k in xrange(2, n_terms + 2):
            a = time.time()
            combo_tensor = np.zeros(tuple([n_types] * k))

            other_ainds = itertools.combinations(neighbors, k - 1)
            k_tuples = ((aind,) + others for others in other_ainds)
            for ainds in k_tuples:
                temp = 1.0
                for i in xrange(k):
                    for j in xrange(i + 1, k):
                        temp *= E_all[ainds[i], ainds[j]]
                combo_tensor[tuple([atom_types[i] for i in ainds])] += temp
            
            for indices in itertools.product(*[np.arange(n_types)] * k):
                combo_str = '.'.join([str(i) for i in np.sort([atom_types[i] for i in indices])])
                feature_ind = np.where(combo_str == combo_strs)[0][0]

                atom_features[feature_ind] += combo_tensor[indices]

        features.append(atom_features)

    features = np.array(features)
    
    return features, combo_strs
