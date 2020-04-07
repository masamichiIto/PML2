from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """ kernel pca with radial basis function kernel
    
    parameters:
    * X: shape = (n_sample, n_features)
    * gamma: float, tuning parameter
    * n_components: int, number of principal components
    
    return:
    X_pc: shape = (n_samples, n_components)
    
    """
    sq_dists = pdist(X, "sqeuclidean")
    mat_sq_dists = squareform(sq_dists)
    K = exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    
    X_pc = np.column_stack((eigvecs[:, i] for i in range(n_components)))
    
    return X_pc