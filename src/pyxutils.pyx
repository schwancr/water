
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float_t DTYPE_FLOAT_t

@cython.boundscheck(False)
def mult(np.ndarray[DTYPE_FLOAT_t, ndim=3] A, 
         np.ndarray[DTYPE_FLOAT_t, ndim=3] B):

    cdef unsigned int i

    for i in range(len(A)):
        A[i] = np.dot(A[i], B[i])

    return
