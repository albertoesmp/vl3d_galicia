# ----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Utils to handle LAS data
# ----------------------------------------------------------------------------

# ---   IMPORTS   --- #
# ------------------- #
import numpy as np
import laspy
import time


# --------------------------------------- #
# ---   P U B L I C   M E T H O D S   --- #
# --------------------------------------- #
def read_las(fpath, print_time=True):
    """
    :param fpath: Path to the LAS/LAZ file to be read
    :return: The read LAS data
    """
    if print_time:
        print(f'Reading "{fpath}" ...')
        start = time.perf_counter()
    las = laspy.read(fpath)
    if print_time:
        end = time.perf_counter()
        print(f'"{fpath}" read in {end-start:.3f} seconds.')
    return las


def coordinates_from_las(las):
    """
    :param las: The LAS data.
    :return: The matrix of 3D coordinates representing the points.
    """
    scales, offsets = las.header.scales, las.header.offsets
    return np.array([
        las.X * scales[0] + offsets[0],
        las.Y * scales[1] + offsets[1],
        las.Z * scales[2] + offsets[2]
    ]).T


def coordinates2D_from_las(las):
    """
    :param las: The LAS data.
    :return: The matrix of 2D coordinates representing the points.
    """
    scales, offsets = las.header.scales, las.header.offsets
    return np.array([
        las.X * scales[0] + offsets[0],
        las.Y * scales[1] + offsets[1]
    ]).T


def labels_from_las(las):
    """
    :param las: The LAS data.
    :return: The vector of classes.
    """
    return las['classification']
