"""
:author: Alberto M. Esmoris Pena
:brief: Python script to count the ocurrences of a given class in a LAS point
    cloud.
    It is mostly used to count the unlabeled points.

:arguments:

    -- arg1: The directory whose LAS files must be considered (it will be
        explored recursively if the RECURSIVE_RECOUNT constant flag is enabled).

    -- arg2: The integer representing the class to be counted.
"""

# ---   IMPORTS   --- #
# ------------------- #
import numpy as np
import laspy
import os
import sys

# ---   CONSTANTS   --- #
# --------------------- #
RECURSIVE_RECOUNT=False


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    # Parse args
    root_dirpath = sys.argv[1]
    ref_class = int(sys.argv[2])
    # Validate dir
    if not os.path.isdir(root_dirpath):
        raise NotADirectoryError(f'{root_dirpath} is not a directory')
    # Recount LAS files (also includes LAZ files)
    dirs = [root_dirpath]
    points_count, class_count, files_count = 0, 0, 0
    while len(dirs) > 0:
        dirpath = dirs.pop()
        children = os.listdir(dirpath)
        for child in children:
            child = os.path.join(dirpath, child)
            if os.path.isdir(child) and RECURSIVE_RECOUNT:
                dirs.append(child)
            elif os.path.isfile(child) and child[-3:] in ['las', 'laz']:
                print(f'Reading {child} ...')
                files_count += 1
                las = laspy.read(child)
                points_count += len(las.classification)
                class_count += np.count_nonzero(
                    np.array(las.classification) == ref_class
                )
    # Report results
    print(f'Total number of points: {points_count}')
    print(
        f'Total number of points in class {ref_class}: {class_count} '
        f'({(100*class_count/points_count):.3f} %)'
    )
    print(f'Total read point clouds: {files_count}')


