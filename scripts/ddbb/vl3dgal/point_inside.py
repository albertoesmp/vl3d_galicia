# ----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Utils to check whether points are inside certain geometries
# ----------------------------------------------------------------------------

# ---   IMPORTS   --- #
# ------------------- #
from shapely.geometry import Point
import open3d
from scipy.spatial import KDTree as KDT
import numpy as np
import joblib


# --------------------------------------- #
# ---   P U B L I C   M E T H O D S   --- #
# --------------------------------------- #
def point_inside_geometry(x, geom):
    """
    :param x: The point to be checked.
    :param geom: The geometry shape.
    :return: True if inside, False otherwise.
    """
    p = Point(x)
    return geom.contains(p)


def points_inside_geometry(
    X, geom, n_jobs=10, support_size=1000, Xsup=None
):
    """
    :param X: The points to be checked.
    :param geom: The geometry shape.
    :return: Point-wise mask with True if inside, False otherwise.
    """
    # Generate support by FPS
    if not isinstance(Xsup, np.ndarray):
        o3d_pcloud = open3d.geometry.PointCloud()
        o3d_pcloud.points = open3d.utility.Vector3dVector(np.hstack(
            [X, np.zeros((X.shape[0], 1))],
        ))
        o3d_pcloud = o3d_pcloud.farthest_point_down_sample(support_size)
        if isinstance(Xsup, list):
            _Xsup = np.asarray(o3d_pcloud.points)[:, :2]
            Xsup.append(_Xsup)
            Xsup = _Xsup
        elif Xsup is None:
            Xsup = np.asarray(o3d_pcloud.points)[:, :2]
    o3d_pcloud = None
    # Check if support points are inside the geometry
    P = [Point(x) for x in Xsup]
    mask = np.array(joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(geom.contains)(p) for p in P
    ))
    # Propagate checks from support points to original points
    kdt_sup = KDT(Xsup)
    I = kdt_sup.query(X, k=1, workers=n_jobs)[1]
    # Return
    return mask[I]

