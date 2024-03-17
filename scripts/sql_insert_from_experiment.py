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
import re
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
MODEL_ID = 1  # The id of the model in the models table of the database


# ---  METHODS  --- #
# ----------------- #
def print_help():
    print(
'''USAGE of analyze_dataset.py

        1: Path to a root directory containing the data of the experiments

        2: Name of the dataset on which the experiments were computed

'''
    )


def printerr(x):
    print(x, file=sys.stderr)


def parse_args():
    # Validate args
    if len(sys.argv) < 3:
        printerr('Not enough arguments were given.\n\n')
        print_help()
        sys.exit(1)
    # Validate path to directory
    dirpath = sys.argv[1]
    if not os.path.isdir(dirpath):
        printerr(f'Given path "{dirpath}" does not point to a directory.\n\n')
        print_help()
        sys.exit(1)
    # Return path to directory and dataset name
    return dirpath, sys.argv[2]


def analyze_experiment(experiment_dir):
    return {
        'class_distribution': analyze_class_distribution(experiment_dir),
        'class_eval': analyze_class_eval(experiment_dir),
        'global_eval': analyze_global_eval(experiment_dir),
        'confusion_matrix': analyze_confusion_matrix(experiment_dir),
        'predicted': analyze_predictions(experiment_dir),
        'uncertainty': analyze_uncertainties(experiment_dir),
        'class_distribution_plot': load_class_distribution_plot(experiment_dir),
        'confusion_matrix_plot': load_confusion_matrix_plot(experiment_dir),
        'pwise_entropy_plot': load_pwise_entropy_plot(experiment_dir),
        'class_ambiguity_plot': load_class_ambiguity_plot(experiment_dir),
        'weighted_entropy_plot': load_weighted_entropy_plot(experiment_dir),
        'cwise_entropy_plot': load_cwise_entropy_plot(experiment_dir),
        'rf_distribution_plot': load_rf_distribution_plot(experiment_dir),
        'class_reduce_plot': load_class_reduction_plot(experiment_dir)
    }


def analyze_class_distribution(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'class_distribution')
    distributions = []
    with open(inpath, 'r') as infile:
        for i in range(2):
            readline(infile)  # Skip first two lines
        line = readline(infile)
        while len(line) > 0:
            record = line.split(',')
            distributions.append({
                'class_name': record[0].strip(" "),
                'recount': int(record[1])
            })
            line = readline(infile)
    return distributions


def analyze_class_eval(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'class_eval')
    evals = []
    with open(inpath, 'r') as infile:
        for i in range(2):
            readline(infile)  # Skip first two lines
        line = readline(infile)
        while len(line) > 0:
            record = [line[:22].strip(' '), line[22:]]
            scores = [float(x) for x in record[1].split(',')]
            evals.append({
                'class_name': record[0],
                'p': scores[0],
                'r': scores[1],
                'f1': scores[2],
                'iou': scores[3]
            })
            line = readline(infile)
    return evals


def analyze_global_eval(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'global_eval')
    geval = {}
    with open(inpath, 'r') as infile:
        readline(infile)  # Skip first line
        keys = readline(infile).split(',')  # Header row
        values = readline(infile).split(',')  # Values row
        for k, v in zip(keys, values):
            k = k.strip(' ').lower()
            v = float(v)
            geval[k] = v
    return geval


def analyze_confusion_matrix(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'confusion_matrix')
    cmat = []
    with open(inpath, 'r') as infile:
        readline(infile)  # Skip first line
        line = readline(infile)
        while len(line) > 0:
            cmat.append([float(x) for x in line.split(',')])
            line = readline(infile)
    return np.array(cmat)


def analyze_predictions(experiment_dir):
    return None  # TODO Rethink : Implement


def analyze_uncertainties(experiment_dir):
    return None  # TODO Rethink : Implement


def load_class_distribution_plot(experiment_dir):
    return None  # TODO Rethink : Implement


def load_confusion_matrix_plot(experiment_dir):
    return None  # TODO Rethink : Implement


def load_pwise_entropy_plot(experiment_dir):
    return None  # TODO Rethink : Implement


def load_class_ambiguity_plot(experiment_dir):
    return None  # TODO Rethink : Implement


def load_weighted_entropy_plot(experiment_dir):
    return None  # TODO Rethink : Implement


def load_cwise_entropy_plot(experiment_dir):
    return None  # TODO Rethink : Implement


def load_rf_distribution_plot(experiment_dir):
    return None  # TODO Rethink : Implement


def load_class_reduction_plot(experiment_dir):
    return None  # TODO Rethink : Implement


def handle_input_file(experiment_dir, key):
    inpath = os.path.join(experiment_dir, PATHS[key])
    if not os.path.isfile(inpath):
        raise FileNotFoundError(
            f'Cannot find input file at: "inpath"'
        )
    return inpath


def readline(f):
    return f.readline().rstrip("\n")

def print_sql_inserts(analysis, dataset_name):
    # Insert global evaluation
    geval = analysis['global_eval']
    print(
        'INSERT INTO global_resultsets '
        f'({",".join([key for key in geval.keys()])}) VALUES\n'
        f'\t({",".join([str(v) for v in geval.values()])})\n'
        '\tON CONFLICT DO NOTHING;\n'
    )
    # Insert resultset
    print(
        'INSERT INTO resultsets '
        '(model_id, dataset_id, global_resultset_id, result_date) VALUES\n'
        '\t(\n'
        f'\t\t{MODEL_ID},\n'
        f"\t\t(SELECT id FROM datasets WHERE name = '{dataset_name}'),\n"
        "\t\t(SELECT currval(pg_get_serial_sequence('global_resultsets', 'id'))),\n"
        '\t\tnow()\n'
        '\t) ON CONFLICT DO NOTHING;\n'
    )
    # Insert class distribution
    cdistr = analysis['class_distribution']
    print(
        'INSERT INTO predicted_class_distributions '
        '(resultset_id, class_id, recount) VALUES'
    )
    for i, distr in enumerate(cdistr):
        print(
            '\t(\n'
            f"\t\t(SELECT currval(pg_get_serial_sequence('resultsets', 'id'))),\n"
            f"\t\t(SELECT id FROM classes where LOWER(classes.name) like '{distr['class_name']}'),\n"
            f'\t\t{distr["recount"]}\n',
            end=''
        )
        if i < len(cdistr)-1:
            print('\t),')
        else:
            print('\t)')
    print('\tON CONFLICT DO NOTHING;\n')
    # Insert confusion matrix
    cmat = analysis['confusion_matrix']
    print(
        'INSERT INTO resultset_confusions '
        '(resultset_id, ref_class_id, pred_class_id, recount) VALUES'
    )
    for i in range(cmat.shape[0]):  # Rows
        for j in range(cmat.shape[1]):  # Columns
            if i != 0 or j != 0:
                print('\n\t),(')
            else:
                print('\n\t(')
            print(
                f"\t\t(SELECT currval(pg_get_serial_sequence('resultsets', 'id'))),\n"
                f"\t\t(SELECT id FROM classes where LOWER(classes.name) like '{cdistr[i]['class_name']}'),\n"
                f"\t\t(SELECT id FROM classes where LOWER(classes.name) like '{cdistr[j]['class_name']}'),\n"
                f'\t\t{cmat[i, j]}'
            )
    print('\t) ON CONFLICT DO NOTHING;\n')
    # TODO Rethink : Implement


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    start = time.perf_counter()
    experiment_dir, dataset_name = parse_args()
    analysis = analyze_experiment(experiment_dir)
    printerr(analysis)  # TODO Remove
    print_sql_inserts(analysis, dataset_name)
    end = time.perf_counter()
    printerr(
        '\n\n'
        f'SQL insert from experiment script run in {end-start:.3f} seconds',
    )

