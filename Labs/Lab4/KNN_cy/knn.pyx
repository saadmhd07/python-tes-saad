import numpy as np
cimport numpy as np
from scipy.stats import mode
 

cdef double[:] distance (double[:,:] x_train_view, double[:] x_test_view ):

    cdef Py_ssize_t j, d
    cdef Py_ssize_t N_train = x_train_view.shape[0]
    cdef Py_ssize_t N_features = x_train_view.shape[1]
    cdef double dist, diff
    cdef double[:] distances = np.zeros(N_train, dtype=np.double)
  
    for j in range(N_train):
        dist = 0
        for d in range(N_features):
            diff = x_train_view[j, d] - x_test_view[d]
            dist += diff * diff
        distances[j] = dist # stocker la distance
    return distances


cpdef int[:] knn(double[:,:] x_train, int[:] class_train, double[:,:] x_test, int k):

    cdef Py_ssize_t N_test = x_test.shape[0]
    cdef Py_ssize_t N_train = x_train.shape[0]
    cdef Py_ssize_t i
    cdef int mode_class
    cdef int[:] knn_indices
    cdef int[:] class_test = np.zeros(N_test, dtype=np.int32)
    cdef double[:] distances
    cdef double[:,:] x_train_view = x_train
    cdef double[:,:] x_test_view = x_test
    cdef np.ndarray[int, ndim=1] nearest_classes

    for i in range(N_test):
        # Renvoie la distance 
        distances = distance(x_train_view, x_test_view[i])
        knn_indices = np.argpartition(distances, k)[:k]
        nearest_classes = np.take(class_train, knn_indices)
        mode_class = int(mode(nearest_classes, keepdims=False)[0])
        class_test[i] = mode_class
    return class_test