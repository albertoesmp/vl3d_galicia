# ---   IMPORTS   --- #
# ------------------- #

import numpy as np
import scipy as sp

from scipy.spatial import ConvexHull
from src.geometry.PCA import compute_PCA

class OrientedBoundingBox():
    """
    :author: Miguel Yermo

    Class that computes the oriented bounding box of a point cloud.
    """
    def __init__(self):
        self.bbox = None
        self.volume = None

    @classmethod
    def create_from_points(cls, points):
        """
        Create an oriented bounding box from a structure space matrix.
        
        :param points: The structure space matrix representing the point cloud
            whose oriented bounding box must be found.
        :type points: :class:`np.ndarray`
        :return: The oriented bounding box of the point cloud as a list of 8
            points with 3 coordinates each the volume of the box.
        """
        hull = ConvexHull(points)
        xyz = points[hull.vertices]
        boxes = []

        for simplex in hull.simplices:
            a, b, c = points[simplex]
            uvw = local_axes(a, b, c)
            if not np.linalg.norm(uvw[0]) or not np.linalg.norm(uvw[1]):
                continue
            frame = [a, uvw[0], uvw[1]]
            rst = world_to_local_coordinates(frame, xyz)
            rmin, smin, tmin = np.amin(rst, axis=0)
            rmax, smax, tmax = np.amax(rst, axis=0)
            dr = rmax - rmin
            ds = smax - smin
            dt = tmax - tmin
            v = dr * ds * dt

            bbox = [
                [rmin, smin, tmin],
                [rmax, smin, tmin],
                [rmax, smax, tmin],
                [rmin, smax, tmin],
                [rmin, smin, tmax],
                [rmax, smin, tmax],
                [rmax, smax, tmax],
                [rmin, smax, tmax],
            ]

            boxes.append([frame, bbox, v])

        frame, bbox, volume = min(boxes, key=lambda b: b[2])
        bbox = local_to_world_coordinates(frame, bbox)

        obj = cls()
        obj.bbox = bbox
        obj.volume = volume

        return obj

    def get_box_points(self):
        """
        Returns the oriented bounding box as a list of 8 points with 3 coordinates each.

        :return: The oriented bounding box as a list of 8 points with 3 coordinates each.
        :rtype: list
        """
        return self.bbox

    def get_volume(self):
        """
        Returns the volume of the bounding box.

        :return: The volume of the bounding box.
        :rtype: float
        """
        return self.volume

def world_to_local_coordinates(frame, xyz):
    """Convert global coordinates to local coordinates. """
    origin = frame[0]
    uvw = [frame[1], frame[2], np.cross(frame[1], frame[2])]
    uvw = np.asarray(uvw).T
    xyz = np.asarray(xyz).T - np.asarray(origin).reshape((-1, 1))
    rst = sp.linalg.solve(uvw, xyz)
    return rst.T


def local_to_world_coordinates(frame, rst):
    """Convert local coordinates to global (world) coordinates. """
    origin = frame[0]
    uvw = [frame[1], frame[2], np.cross(frame[1], frame[2])]

    uvw = np.asarray(uvw).T
    rst = np.asarray(rst).T
    xyz = uvw.dot(rst) + np.asarray(origin).reshape((-1, 1))
    return xyz.T

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def local_axes(a, b, c):
    u = b - a
    v = c - a
    w = np.cross(u, v)
    v = np.cross(w, u)
    return normalize_vector(u), normalize_vector(v), normalize_vector(w)

