
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import mdtraj as md

class Sprint(BaseEstimator, TransformerMixin):
    r"""
    Use the SPRINT representation from

    [1] Pietrucci, F and Andreoni, W. Graph Theory Meets
        Ab Initio Molecular Dynamics: Atomic Structures
        and Transformations at the Nanoscale. (2011) PRL,
        504, pp 085504.

    There is a slight adaptation, since we will define a
    bond between waters as the minimum O-H and H-O distances

    Parameters
    ----------
    r0 : float
        radius to use as the 'typical' (hydrogen) bond radius
    n : int, optional
        numerator exponent in the representation (see below)
    m : int, optional
        denominator exponent in the representation (see below)

    Notes
    -----
    The representation consists of the eigenvector of an
    adjacency matrix with entries corresponding to:
    
    .. math ::

        a_{ij} = \frac{1 - (r_{ij} / r0)^n}{1 - (r_{ij} / r0)^m}

    """
