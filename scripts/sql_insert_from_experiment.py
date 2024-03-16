# ----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Analyze the outputs generated during a single experiment.
#   The output will be printed through the standard output as a set of
#   SQL insert-like sentences.
# ----------------------------------------------------------------------------


# ---  IMPORTS  --- #
# ----------------- #
import vl3dgal.las_utils as lasu
import vl3dgal.assets as assets
import vl3dgal.classes as classes
from vl3dgal.point_inside import points_inside_geometry
import laspy
import numpy as np
import sys
import os
import time
import gc


# ---  CONSTANTS  --- #
# ------------------- #
PATHS = {  # Paths relative to the root directory
    'class_distribution': 'report/class_distribution.log',
    'class_eval': 'report/class_eval.log',
    'global_eval': 'report/global_eval.log',
    'confusion_matrix': 'report/confusion_matrix.log',
    'predicted': 'predicted.laz',
    'uncertainty': 'uncertainty/uncertainty.laz',
    'class_distribution_plot': 'plot/class_distribution.svg',
    'confusion_matrix_plot': 'plot/confusion_matrix.svg',
    'pwise_entropy_plot': 'uncertainty/point_wise_entropy_figure.svg',
    'class_ambiguity_plot': 'uncertainty/class_ambiguity_figure.svg',
    'weighted_entropy_plot': 'uncertainty/weighted_entropy_figure.svg',
    'cwise_entropy_plot': 'uncertainty/cluster_wise_entropy_figure.svg',
    'rf_distribution_plot': 'training_eval/receptive_fields_distribution.svg',
    'class_reduce_plot': 'class_reduction.svg'
}



# ---  METHODS  --- #
# ----------------- #
def print_help():
    print(
'''USAGE of analyze_dataset.py

        1: Path to a root directory containing the data of the experiments

'''
    )


def printerr(x):
    print(x, file=sys.stderr)


def parse_args():
    # Validate args
    if len(sys.argv) < 2:
        printerr('Not enough arguments were given.\n\n')
        print_help()
        sys.exit(1)
    # Validate path to directory
    dirpath = sys.argv[1]
    if not os.path.isdir(dirpath):
        printerr(f'Given path "{dirpath}" does not point to a directory.\n\n')
        print_help()
        sys.exit(1)
    # Return path to directory
    return dirpath



# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    start = time.perf_counter()
    experiment_dir = parse_args()
    analysis = analyze_experiment(experiment_dir)
    print_sql_inserts(analysis, pclouds_dir)
    end = time.perf_counter()
    printerr(
        '\n\n'
        f'SQL insert from experiment script run in {end-start:.3f} seconds',
    )

