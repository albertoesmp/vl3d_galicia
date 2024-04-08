# ----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Analyze the outputs generated during a single experiment.
#   The output will be printed through the standard output as a set of
#   SQL insert-like sentences.
# ----------------------------------------------------------------------------


# ---  IMPORTS  --- #
# ----------------- #
import vl3dgal.las_utils as lasu
import vl3dgal.classes as classes
import numpy as np
import base64
import sys
import os
import time


# ---  CONSTANTS  --- #
# ------------------- #
PATHS = {  # Paths relative to the root directory
    'class_distribution': 'report/class_distribution.log',
    'class_eval': 'report/class_eval.log',
    'global_eval': 'report/global_eval.log',
    'confusion_matrix': 'report/confusion_matrix.log',
    'rf_distribution': 'training_eval/receptive_fields_distribution.log',
    'uncertainty': 'uncertainty/uncertainty.las',
    'class_distribution_plot': 'plot/class_distribution.svg',
    'confusion_matrix_plot': 'plot/confusion_matrix.svg',
    'pwise_entropy_plot': 'uncertainty/point_wise_entropy_figure.svg',
    'class_ambiguity_plot': 'uncertainty/class_ambiguity_figure.svg',
    'weighted_entropy_plot': 'uncertainty/weighted_entropy_figure.svg',
    'cwise_entropy_plot': 'uncertainty/cluster_wise_entropy_figure.svg',
    'rf_distribution_plot': 'training_eval/receptive_fields_distribution.svg',
    'class_reduce_plot': 'class_reduction.svg'
}
# The id of the model in the models table of the database
MODEL_ID = os.environ.get('MODEL_ID', 1)
CLASSES = [  # The classes representing the classification task
    'vegetation'
]  # See keys from vl3dgal.classes.CLASS_NAMES


# ---  METHODS  --- #
# ----------------- #
def print_help():
    print(
'''USAGE of sql_insert_from_experiment.py

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
        'rf_distribution': analyze_rf_distribution(experiment_dir),
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
                'class_name': record[0].strip(" "),
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


def analyze_rf_distribution(experiment_dir, key='rf_distribution', paths=None):
    inpath = handle_input_file(experiment_dir, key, paths=paths)
    rf_distributions = []
    with open(inpath, 'r') as infile:
        readline(infile)  # Skip first line
        line = readline(infile)
        while len(line) > 0:
            if line[:4] == 'SUM':  # Skip last row with SUM of all values
                continue
            fields = line.split(',')
            rf_distribution = {
                'class_name': fields[0].strip(" "),
                'pred_count': fields[1],
                'pred_rf_count': fields[3],
            }
            if len(fields) > 5:
                rf_distribution['ref_count'] = fields[5]
                rf_distribution['ref_rf_count'] = fields[7]
            rf_distributions.append(rf_distribution)
            line = readline(infile)
    return rf_distributions


def analyze_uncertainties(experiment_dir):
    # Read point cloud
    inpath = handle_input_file(experiment_dir, 'uncertainty')
    las = lasu.read_las(inpath, print_time=False)
    # Find classes
    task, classes = find_classes_from_las(las)
    # Extract entropies
    pwe = las['PointWiseEntropy']
    we = las['WeightedEntropy']
    cwe = las['ClusterWiseEntropy']
    ca = las['ClassAmbiguity']
    # Entropy of point i given it is labeled as class x
    y = las.classification
    pwe_by_class = make_class_wise_uncertainty(pwe, y, classes)
    we_by_class = make_class_wise_uncertainty(we, y, classes)
    cwe_by_class = make_class_wise_uncertainty(cwe, y, classes)
    ca_by_class = make_class_wise_uncertainty(ca, y, classes)
    # Entropy of point i given it is predicted as x
    yhat = las['Prediction']
    pwe_by_pred = make_class_wise_uncertainty(pwe, yhat, classes)
    we_by_pred = make_class_wise_uncertainty(we, yhat, classes)
    cwe_by_pred = make_class_wise_uncertainty(cwe, yhat, classes)
    ca_by_pred = make_class_wise_uncertainty(ca, yhat, classes)
    # Extract likelihoods
    lkhd = [make_uncertainty_dict(las[classi]) for classi in classes]
    # Extract likelihood given points belong to class x
    lkhd_by_class = [
        make_class_wise_uncertainty(las[classi], y, classes)
        for classi in classes
    ]
    # Extract likelihood given points were predicted as x
    lkhd_by_pred = [
        make_class_wise_uncertainty(las[classi], yhat, classes)
        for classi in classes
    ]
    # Return
    return {
        'pwise_entropy': make_uncertainty_dict(pwe),
        'weighted_entropy': make_uncertainty_dict(we),
        'cwise_entropy': make_uncertainty_dict(cwe),
        'class_ambiguity': make_uncertainty_dict(ca),
        'pwise_entropy_by_class': pwe_by_class,
        'weighted_entropy_by_class': we_by_class,
        'cwise_entropy_by_class': cwe_by_class,
        'class_ambiguity_by_class': ca_by_class,
        'pwise_entropy_by_pred': pwe_by_pred,
        'weighted_entropy_by_pred': we_by_pred,
        'cwise_entropy_by_pred': cwe_by_pred,
        'class_ambiguity_by_pred': ca_by_pred,
        'classif_type': task,
        'classes': classes,
        'likelihood': lkhd,
        'likelihood_by_class': lkhd_by_class,
        'likelihood_by_pred': lkhd_by_pred
    }


def load_class_distribution_plot(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'class_distribution_plot')
    return digest_figure(inpath)


def load_confusion_matrix_plot(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'confusion_matrix_plot')
    return digest_figure(inpath)


def load_pwise_entropy_plot(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'pwise_entropy_plot')
    return digest_figure(inpath)


def load_class_ambiguity_plot(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'class_ambiguity_plot')
    return digest_figure(inpath)


def load_weighted_entropy_plot(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'weighted_entropy_plot')
    return digest_figure(inpath)


def load_cwise_entropy_plot(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'cwise_entropy_plot')
    return digest_figure(inpath)


def load_rf_distribution_plot(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'rf_distribution_plot')
    return digest_figure(inpath)


def load_class_reduction_plot(experiment_dir):
    inpath = handle_input_file(experiment_dir, 'class_reduce_plot')
    return digest_figure(inpath)


def handle_input_file(experiment_dir, key, paths=None):
    if paths is None:
        paths = PATHS
    inpath = os.path.join(experiment_dir, paths[key])
    if not os.path.isfile(inpath):
        raise FileNotFoundError(
            f'Cannot find input file at: "{inpath}"'
        )
    return inpath


def readline(f):
    return f.readline().rstrip("\n")


def digest_figure(inpath):
    try:
        frmat = inpath[inpath.rindex('.')+1:]
        fmt_low = frmat.lower()
        if fmt_low not in ['svg', 'png', 'gif', 'bmp', 'svg', 'geotiff', 'tiff']:
            raise ValueError(
                'Given path points to a file with an unexpected extension '
                f'"{frmat}"'
            )
        with open(inpath, 'rb') as infile:
            bytea = infile.read()
        return {
            'format': frmat,
            'bytea': base64.b64encode(bytea).decode('utf-8')
        }
    except Exception as ex:
        print(f'Failed to read figure at "{inpath}"')
        raise ex


def find_classes_from_las(las):
    # Check LMH
    has_midveg = False
    try:
        x = las['midveg']
        has_midveg = True
    except Exception as ex:
        pass
    if has_midveg:
        return 'LHM_VEGETATION', ['lowveg', 'midveg', 'highveg', 'other', 'ignore']
    # Check building
    has_building = False
    try:
        x = las['building']
        has_building = True
    except Exception as ex:
        pass
    # Check vegetation
    has_vegetation = False
    try:
        x = las['vegetation']
        has_vegetation = True
    except Exception as ex:
        pass
    # Return
    if has_building:
        if has_vegetation:
            return 'BUILD_VEG', ['vegetation', 'building', 'other', 'ignore']
        return 'BUILDING', ['building', 'other', 'ignore']
    elif has_vegetation:
        return 'VEGETATION', ['vegetation', 'other', 'ignore']
    else:
        raise ValueError('Unexpected combination of classes.')


def make_uncertainty_dict(x):
    return {
        'mean': np.mean(x),
        'stdev': np.std(x),
        'Q': np.quantile(x, [i/10 for i in range(1, 10)])
    }

def make_class_wise_uncertainty(x, y, classes):
    cwu = {}
    for i, classi in enumerate(classes):
        I = y == i
        if np.count_nonzero(I) > 0:
            cwu[classi] = make_uncertainty_dict(x[I])
    return cwu


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
    # Insert classwise evaluation
    cevals = analysis['class_eval']
    print(
        'INSERT INTO classwise_resultsets '
        '(resultset_id, class_id, p, r, f1, iou) VALUES\n'
    )
    for i, ceval in enumerate(cevals):
        print(
            f"\t(\n"
            f"\t\t(SELECT currval(pg_get_serial_sequence('resultsets', 'id'))),\n"
            f"\t\t(SELECT id FROM classes WHERE LOWER(classes.name) like '{ceval['class_name']}'),\n"
            f"\t\t{ceval['p']}, {ceval['r']}, {ceval['f1']}, {ceval['iou']}"
        )
        if i < len(cevals)-1:
            print('\t),')
        else:
            print('\t)')
    print('\tON CONFLICT DO NOTHING;\n')
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
            f"\t\t(SELECT id FROM classes WHERE LOWER(classes.name) like '{distr['class_name']}'),\n"
            f'\t\t{distr["recount"]}\n',
            end=''
        )
        if i < len(cdistr)-1:
            print('\t),')
        else:
            print('\t)')
    print('\tON CONFLICT DO NOTHING;\n')
    # Insert receptive field distribution
    rfdistr = analysis['rf_distribution']
    print(
        'INSERT INTO receptive_field_distributions '
        '(resultset_id, class_id, pred_count, pred_rf_count) VALUES'
    )
    for i, distr in enumerate(rfdistr):
        print(
            '\t(\n'
            f"\t\t(SELECT currval(pg_get_serial_sequence('resultsets', 'id'))),\n"
            f"\t\t(SELECT id FROM classes WHERE LOWER(classes.name) like '{distr['class_name']}'),\n"
            f'\t\t{distr["pred_count"]},\n'
            f'\t\t{distr["pred_rf_count"]}',
            end=''
        )
        if i < len(rfdistr)-1:
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
    # Insert uncertainties and likelihoods
    uncertainty = analysis['uncertainty']
    print(
        'INSERT INTO uncertainty_resultsets '
        '(resultset_id, metric_id, mean, stdev, '
        'q1, q2, q3, q4, q5, q6, q7, q8, q9) VALUES'
    )
    print_uncertainty_sql_values(
        'Point-wise entropy', uncertainty['pwise_entropy'], last=False
    )
    print_uncertainty_sql_values(
        'Weighted point-wise entropy', uncertainty['weighted_entropy'], last=False
    )
    print_uncertainty_sql_values(
        'Cluster-wise entropy', uncertainty['cwise_entropy'], last=False
    )
    print_uncertainty_sql_values(
        'Class ambiguity', uncertainty['class_ambiguity'], last=True
    )
    print(
        'INSERT INTO likelihood_resultsets '
        '(resultset_id, class_id, mean, stdev, '
        'q1, q2, q3, q4, q5, q6, q7, q8, q9) VALUES'
    )
    for i, likelihood in enumerate(uncertainty['likelihood']):
        print_likelihood_sql_values(
            classes.CLASS_NAMES[uncertainty['classes'][i]],
            likelihood,
            last=i==len(uncertainty['likelihood'])-1
        )
    print(
        'INSERT INTO classwise_uncertainty_resultsets '
        '(resultset_id, class_id, metric_id, mean, stdev, '
        'q1, q2, q3, q4, q5, q6, q7, q8, q9) VALUES'
    )
    pwe = uncertainty['pwise_entropy_by_class']
    for i, classi in enumerate(uncertainty['classes']):
        if classi not in pwe:
            continue
        print_uncertainty_by_class_sql_values(
            'Point-wise entropy',
            classes.CLASS_NAMES[classi],
            pwe[classi],
            last=False
        )
    we = uncertainty['weighted_entropy_by_class']
    for i, classi in enumerate(uncertainty['classes']):
        if classi not in we:
            continue
        print_uncertainty_by_class_sql_values(
            'Weighted point-wise entropy',
            classes.CLASS_NAMES[classi],
            we[classi],
            last=False
        )
    cwe = uncertainty['cwise_entropy_by_class']
    for i, classi in enumerate(uncertainty['classes']):
        if classi not in cwe:
            continue
        print_uncertainty_by_class_sql_values(
            'Cluster-wise entropy',
            classes.CLASS_NAMES[classi],
            we[classi],
            last=False
        )
    ca = uncertainty['class_ambiguity_by_class']
    for i, classi in enumerate(uncertainty['classes']):
        if classi not in ca:
            continue
        print_uncertainty_by_class_sql_values(
            'Class ambiguity',
            classes.CLASS_NAMES[classi],
            ca[classi],
            last=i==len(uncertainty['class_ambiguity_by_class'])-1
        )
    print(
        'INSERT INTO predictive_uncertainty_resultsets '
        '(resultset_id, class_id, metric_id, mean, stdev, '
        'q1, q2, q3, q4, q5, q6, q7, q8, q9) VALUES'
    )
    pwe = uncertainty['pwise_entropy_by_pred']
    for i, classi in enumerate(uncertainty['classes']):
        if classi not in pwe:
            continue
        print_uncertainty_by_class_sql_values(
            'Point-wise entropy',
            classes.CLASS_NAMES[classi],
            pwe[classi],
            last=False
        )
    we = uncertainty['weighted_entropy_by_pred']
    for i, classi in enumerate(uncertainty['classes']):
        if classi not in we:
            continue
        print_uncertainty_by_class_sql_values(
            'Weighted point-wise entropy',
            classes.CLASS_NAMES[classi],
            we[classi],
            last=False
        )
    cwe = uncertainty['cwise_entropy_by_pred']
    for i, classi in enumerate(uncertainty['classes']):
        if classi not in cwe:
            continue
        print_uncertainty_by_class_sql_values(
            'Cluster-wise entropy',
            classes.CLASS_NAMES[classi],
            we[classi],
            last=False
        )
    ca = uncertainty['class_ambiguity_by_pred']
    for i, classi in enumerate(uncertainty['classes']):
        if classi not in ca:
            continue
        print_uncertainty_by_class_sql_values(
            'Class ambiguity',
            classes.CLASS_NAMES[classi],
            ca[classi],
            last=i==len(uncertainty['class_ambiguity_by_pred'])-1
        )
    print(
        'INSERT INTO classwise_likelihood_resultsets '
        '(resultset_id, class_id, ref_class_id, mean, stdev, '
        'q1, q2, q3, q4, q5, q6, q7, q8, q9) VALUES'
    )
    lbc = uncertainty['likelihood_by_class']
    for i, classi in enumerate(uncertainty['classes']):
        for j, (lbcij_class, lbcij_metrics) in enumerate(lbc[i].items()):
            print_likelihood_by_class_sql_values(
                classes.CLASS_NAMES[uncertainty['classes'][i]],
                classes.CLASS_NAMES[lbcij_class],
                lbcij_metrics,
                last=j==len(lbc[i])-1 and i==len(uncertainty['classes'])-1
            )
    print(
        'INSERT INTO predictive_likelihood_resultsets '
        '(resultset_id, class_id, pred_class_id, mean, stdev, '
        'q1, q2, q3, q4, q5, q6, q7, q8, q9) VALUES'
    )
    lbc = uncertainty['likelihood_by_pred']
    for i, classi in enumerate(uncertainty['classes']):
        for j, (lbcij_class, lbcij_metrics) in enumerate(lbc[i].items()):
            print_likelihood_by_class_sql_values(
                classes.CLASS_NAMES[uncertainty['classes'][i]],
                classes.CLASS_NAMES[lbcij_class],
                lbcij_metrics,
                last=j==len(lbc[i])-1 and i==len(uncertainty['classes'])-1
            )
    # Insert figures
    print_sql_insert_figure(
        analysis['class_distribution_plot'],
        'Class distribution'
    )
    print_sql_insert_figure(
        analysis['confusion_matrix_plot'],
        'Validation confusion matrix'
    )
    print_sql_insert_figure(
        analysis['pwise_entropy_plot'],
        'Point-wise entropy'
    )
    print_sql_insert_figure(
        analysis['class_ambiguity_plot'],
        'Class ambiguity'
    )
    print_sql_insert_figure(
        analysis['weighted_entropy_plot'],
        'Weighted entropy'
    )
    print_sql_insert_figure(
        analysis['cwise_entropy_plot'],
        'Cluster-wise entropy'
    )
    print_sql_insert_figure(
        analysis['rf_distribution_plot'],
        'Validation receptive fields distribution'
    )
    print_sql_insert_figure(
        analysis['class_reduce_plot'],
        'Class reduction distribution'
    )


def print_sql_insert_figure(figdict, plot_name):
    print(
        'INSERT INTO resultset_plots '
        '(resultset_id, plot_id, plot_bin, plot_format_id) VALUES\n'
        '\t(\n'
        f"\t\t(SELECT currval(pg_get_serial_sequence('resultsets', 'id'))),\n"
        f"\t\t(SELECT id FROM plots WHERE name like '{plot_name}'),\n"
        f"\t\t'{figdict['bytea']}'::bytea,\n"
        f"\t\t(SELECT id FROM plot_formats WHERE LOWER(name) like '%{figdict['format']}%')\n"
        '\t) ON CONFLICT DO NOTHING;\n'
    )
    # Note that plots inserted like this must be obtained with a query similar
    # to: SELECT encode(plot_bin, 'escape') FROM resultset_plot


def print_uncertainty_sql_values(metric_name, x, last=False):
    end_line = '\t),'
    if last:
        end_line = '\t) ON CONFLICT DO NOTHING;\n'
    print(
        '\t(\n'
        f"\t\t(SELECT currval(pg_get_serial_sequence('resultsets', 'id'))),\n"
        f"\t\t(SELECT id FROM uncertainty_metrics WHERE name = '{metric_name}'),\n"
        f'\t\t{x["mean"]},\n'
        f'\t\t{x["stdev"]},\n'
        f'\t\t{",".join(x["Q"].astype(str))}\n'
        f'{end_line}'
    )


def print_likelihood_sql_values(class_name, x, last=False):
    end_line = '\t),'
    if last:
        end_line = '\t) ON CONFLICT DO NOTHING;\n'
    print(
        '\t(\n'
        f"\t\t(SELECT currval(pg_get_serial_sequence('resultsets', 'id'))),\n"
        f"\t\t(SELECT id FROM classes WHERE name = '{class_name}'),\n"
        f'\t\t{x["mean"]},\n'
        f'\t\t{x["stdev"]},\n'
        f'\t\t{",".join(x["Q"].astype(str))}\n'
        f'{end_line}'
    )


def print_uncertainty_by_class_sql_values(metric_name, class_name, x, last=False):
    end_line = '\t),'
    if last:
        end_line = '\t) ON CONFLICT DO NOTHING;\n'
    print(
        '\t(\n'
        f"\t\t(SELECT currval(pg_get_serial_sequence('resultsets', 'id'))),\n"
        f"\t\t(SELECT id FROM classes WHERE name = '{class_name}'), \n"
        f"\t\t(SELECT id FROM uncertainty_metrics WHERE name = '{metric_name}'),\n"
        f'\t\t{x["mean"]},\n'
        f'\t\t{x["stdev"]},\n'
        f'\t\t{",".join(x["Q"].astype(str))}\n'
        f'{end_line}'
    )


def print_likelihood_by_class_sql_values(
    class_name, ref_class_name, x, last=False
):
    end_line = '\t),'
    if last:
        end_line = '\t) ON CONFLICT DO NOTHING;\n'
    print(
        '\t(\n'
        f"\t\t(SELECT currval(pg_get_serial_sequence('resultsets', 'id'))),\n"
        f"\t\t(SELECT id FROM classes WHERE name = '{class_name}'), \n"
        f"\t\t(SELECT id FROM classes WHERE name = '{ref_class_name}'),\n"
        f'\t\t{x["mean"]},\n'
        f'\t\t{x["stdev"]},\n'
        f'\t\t{",".join(x["Q"].astype(str))}\n'
        f'{end_line}'
    )


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    start = time.perf_counter()
    experiment_dir, dataset_name = parse_args()
    printerr(f'MODEL_ID: {MODEL_ID}')
    analysis = analyze_experiment(experiment_dir)
    print_sql_inserts(analysis, dataset_name)
    end = time.perf_counter()
    printerr(
        '\n\n'
        f'SQL insert from experiment script run in {end-start:.3f} seconds',
    )

