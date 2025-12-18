#!/bin/python3

# ---   IMPORTS   --- #
# ------------------- #
import pandas as pd
import numpy as np
from scipy.spatial import KDTree as KDT


# ---  DATA SPECIFICATION  --- #
# ---------------------------- #
# Path to the input point cloud
X_path = 'Octree_X.xyz'
# Path to the input supports points defining the neighborhoods (e.g., centers)
Xsup_path = 'Octree_Xsup.xyz'


# ---   INPUT/OUTPUT   --- #
# ------------------------ #
def write_output(Y, Ypath):
    with open(Ypath, 'w') as outf:
        for yi in Y:
            outf.write(f'{",".join([str(i) for i in yi])}\n')


# ---   ANALYSIS   --- #
# -------------------- #
def analyze_knn(X, Xsup, kdt, k, Npath, Dpath):
    D, N = kdt.query(Xsup, k, workers=-1)
    write_output(N, Npath)
    write_output(D, Dpath)


def analyze_boundedKnn(X, Xsup, kdt, k, maxDistanceBound, Npath, Dpath):
    D, N = kdt.query(Xsup, k, workers=-1)  # Query KNN
    D, N = D.tolist(), N.tolist()
    for i in range(len(N)):  # Filter out those out of bounds
        di = D[i]
        for j, dij in enumerate(di):
            if dij > maxDistanceBound:
                D[i] = di[:j]
                N[i] = N[i][:j]
                break
    write_output(N, Npath)
    write_output(D, Dpath)


def analyze_sphere(X, Xsup, kdt, r, Npath):
    N = kdt.query_ball_point(Xsup, r, workers=-1)
    write_output(N, Npath)


def analyze_boundedCylinder(X, Xsup, kdt, r, zmin, zmax, Npath):
    N = kdt.query_ball_point(Xsup[:, :2], r, workers=-1)  # Query cylinder
    # Filter out points out of bounds (on vertical coordinate, i.e., z)
    for i, ni in enumerate(N):
        zi = Xsup[i, 2]
        z = X[ni][:, 2]-zi
        # Compute boolean mask for point inside boundaries (True)
        mask = [zj >= zmin and zj <= zmax for zj in z]
        N[i] = np.array(ni)[mask].tolist()  # Preserve only neighbors inside
    write_output(N, Npath)


def analyze_rectangular(X, Xsup, kdt, r, Npath):
    # Obtain neighborhood in the sphere that contains the bounding box
    n = X.shape[1]  # Space dimensionality
    rout = np.sqrt(n) * r  # Radius of S s.t. BBox \subseteq S
    N = kdt.query_ball_point(Xsup, rout, workers=-1)
    # Obtain bounding box
    for i, ni in enumerate(N):
        Xsupi = Xsup[i]
        Xni = X[ni]-Xsupi
        # Boolean mask for points inside the bounding box
        mask = np.all(Xni >= -r, axis=1)*np.all(Xni <= r, axis=1)
        N[i] = np.array(ni)[mask].tolist()
    if Npath is not None:
        write_output(N, Npath)
    return N


def analyze_boundedRectangular(X, Xsup, kdt, r, zmin, zmax, Npath):
    # Obtain 2D rectangular neighborhood
    N = analyze_rectangular(X[:, :2], Xsup[:, :2], kdt, r, None)
    # Filter out points out of bounds (on vertical coordinate, i.e., z)
    for i, ni in enumerate(N):
        zi = Xsup[i, 2]
        z = X[ni][:, 2]-zi
        # Compute boolean mask for point inside boundaries (True)
        mask = [zj >= zmin and zj <= zmax for zj in z]
        N[i] = np.array(ni)[mask].tolist()  # Preserve only neighbors inside
    write_output(N, Npath)


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    # Load data
    X = pd.read_csv(X_path, sep=',', header=None).to_numpy()
    Xsup = pd.read_csv(Xsup_path, sep=',', header=None).to_numpy()
    # Build KDTree
    kdt3D = KDT(X)
    kdt2D = KDT(X[:, :2])
    # Do the many analysis
    analyze_knn(X, Xsup, kdt3D, 16, 'Octree_knn3D_N.txt', 'Octree_knn3D_D.txt')
    analyze_knn(X[:, :2], Xsup[:, :2], kdt2D, 16, 'Octree_knn2D_N.txt', 'Octree_knn2D_D.txt')
    analyze_boundedKnn(X, Xsup, kdt3D, 16, 0.21, 'Octree_boundedKnn3D_N.txt', 'Octree_boundedKnn3D_D.txt')
    analyze_boundedKnn(X[:, :2], Xsup[:, :2], kdt2D, 16, 0.21, 'Octree_boundedKnn2D_N.txt', 'Octree_boundedKnn2D_D.txt')
    analyze_sphere(X, Xsup, kdt3D, 0.30, 'Octree_sphere3D_N.txt')
    analyze_sphere(X[:, :2], Xsup[:, :2], kdt2D, 0.3, 'Octree_sphere2D_N.txt')
    analyze_boundedCylinder(X, Xsup, kdt2D, 0.30, -1.01, 1.01, 'Octree_boundedCylinder_N.txt')
    analyze_rectangular(X, Xsup, kdt3D, 0.30, 'Octree_rectangular3D_N.txt')
    analyze_rectangular(X[:, :2], Xsup[:, :2], kdt2D, 0.30, 'Octree_rectangular2D_N.txt')
    analyze_boundedRectangular(X, Xsup, kdt2D, 0.30, -1.01, 1.01, 'Octree_boundedRectangular_N.txt')
