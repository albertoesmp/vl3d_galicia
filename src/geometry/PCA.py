from scipy.linalg import svd  # type: ignore
import numpy as np

def compute_PCA(X):
    """Compute the principal components of the structure matrix of a set of points.

    # TODO: Add more information about the PCA algorithm.
    """

    m, dim = X.shape

    if not m >= dim:
        raise ValueError("The number of observations must be greater than the number of points.")

    # Compute centroid of the structure matrix
    centroid = np.mean(X, axis=0)
    Y = X - centroid

    # Compute covariance matrix of the structure matrix
    C = np.cov(Y.T)
    eigenvalues, eigenvectors = np.linalg.eig(C)

    return centroid[0], eigenvectors, eigenvalues
