# ----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Analyze the outputs generated during a single training process.
#   The output will be printed through the standard output as a set of
#   SQL insert-like sentences.
# ----------------------------------------------------------------------------


# ---  IMPORTS  --- #
# ----------------- #
import vl3dgal.classes as classes
from sql_insert_from_experiment import handle_input_file,digest_figure, \
    analyze_rf_distribution
import pandas as pd
import numpy as np
import base64
import re
import sys
import os
import time


# ---  CONSTANTS  --- #
# ------------------- #
PATHS = {  # Paths relative to the root directory
    'model_summary': 'model_summary.log',
    'trf_distribution': 'training_eval/training_receptive_fields_distribution.log',
    'training_history': 'training_eval/history/training_history.csv',
    'model_graph': 'model_graph.png',
    'class_distribution_plot': 'training_eval/class_distribution.svg',
    'confusion_matrix_plot': 'training_eval/confusion.svg',
    'trf_distribution_plot': 'training_eval/training_receptive_fields_distribution.svg',
    'training_categorical_accuracy_plot': 'training_eval/history/categorical_accuracy.svg',
    'training_loss_plot': 'training_eval/history/loss.svg',
    'training_lr_plot': 'training_eval/history/lr.svg',
    'training_summary_plot': 'training_eval/history/summary.svg',
    'class_reduce_plot': 'class_reduction.svg',
    'kpconv_plots': 'training_eval/kpconv_layers/',
    'skpconv_plots': 'training_eval/skpconv_layers/'
}
CLASSES = [  # The classes representing the classification task
    'vegetation'
]  # See keys from vl3dgal.classes.CLASS_NAMES


# ---  METHODS  --- #
# ----------------- #
def print_help():
    print(
'''USAGE of sql_insert_from_training.py

        1: Path to the training JSON

        2: Path to a root directory containing the data of the training

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
    # Validate path to json
    jsonpath = sys.argv[1]
    if not os.path.isfile(jsonpath):
        printerr(f'Given path "{jsonpath}" does not point to a file.\n\n')
        print_help()
        sys.exit(1)
    # Validate path to directory
    dirpath = sys.argv[2]
    if not os.path.isdir(dirpath):
        printerr(f'Given path "{dirpath}" does not point to a directory.\n\n')
        print_help()
        sys.exit(1)
    # Return path to directory and dataset name
    return jsonpath, dirpath


def analyze_experiment(training_json_path, experiment_dir):
    def parse_model_name(regexp, input_str, model_name):
        if model_name is not None:
            return model_name
        result = regexp.findall(input_str)
        if len(result) > 0:
            return result[0][1:-1]
    # Read training json
    training_json = ''
    regexp = re.compile('/kpc_final_[A-Za-z]*/')
    model_name = None
    with open(training_json_path, 'r') as infile:
        line = infile.readline()
        while len(line) > 0:
            model_name = parse_model_name(regexp, line, model_name)
            training_json += line
            line = infile.readline()
    # Return analysis results
    return {
        'model_name': model_name,
        'training_json': training_json,
        'model_summary': analyze_model_summary(training_dir),
        'trf_distribution': analyze_rf_distribution(
            training_dir, key='trf_distribution', paths=PATHS
        ),
        'training_history': analyze_training_history(training_dir),
        # TODO Rethink : Model graph is commented due to missing lib at FT-III
        #'model_graph': load_model_graph(training_dir),
        'class_distribution_plot': load_class_distribution_plot(training_dir),
        'confusion_matrix_plot': load_confusion_matrix_plot(training_dir),
        'trf_distribution_plot': load_trf_distribution_plot(training_dir),
        'training_categorical_accuracy_plot': load_training_categorical_accuracy_plot(training_dir),
        'training_loss_plot': load_training_loss_plot(training_dir),
        'training_lr_plot': load_training_lr_plot(training_dir),
        'training_summary_plot': load_training_summary_plot(training_dir),
        'class_reduce_plot': load_class_reduce_plot(training_dir),
        'kpconv_plots': load_kpconv_plots(training_dir),
        'skpconv_plots': load_skpconv_plots(training_dir)
    }

def analyze_model_summary(training_dir):
    inpath = handle_input_file(training_dir, 'model_summary', paths=PATHS)
    summary = ''
    with open(inpath, 'r') as infile:
        line = infile.readline()
        while len(line) > 0:
            summary += line
            line = infile.readline()
    return summary


def analyze_training_history(training_dir):
    inpath = handle_input_file(training_dir, 'training_history', paths=PATHS)
    df = pd.read_csv(inpath)
    keys = [key for key in df.keys().tolist() if key != 'epoch']
    outdict = {'num_epochs': len(df['epoch']), 'metrics': []}
    for key in keys:
        v = df[key]
        dbname = '#UNKNOWN#'
        if key == 'loss':
            dbname = 'Loss'
        elif key == 'lr':
            dbname = 'Learning rate'
        elif key == 'categorical_accuracy':
            dbname = 'Categorical accuracy'
        outdict['metrics'].append({
            'name': dbname,
            'start_value': v[0],
            'end_value': v[len(v)-1],
            'min_value': v.min(),
            'min_value_epoch': v.idxmin(),
            'max_value': v.max(),
            'max_value_epoch': v.idxmax(),
            'mean_value': v.mean(),
            'stdev_value': v.std(),
            'Q': v.quantile([i/10 for i in range(1, 10)]).to_numpy()
        })
    return outdict


def load_model_graph(training_dir):
    inpath = handle_input_file(training_dir, 'model_graph', paths=PATHS)
    return digest_figure(inpath)


def load_class_distribution_plot(training_dir):
    inpath = handle_input_file(training_dir, 'class_distribution_plot', paths=PATHS)
    return digest_figure(inpath)


def load_confusion_matrix_plot(training_dir):
    inpath = handle_input_file(training_dir, 'confusion_matrix_plot', paths=PATHS)
    return digest_figure(inpath)


def load_trf_distribution_plot(training_dir):
    inpath = handle_input_file(training_dir, 'trf_distribution_plot', paths=PATHS)
    return digest_figure(inpath)


def load_training_categorical_accuracy_plot(training_dir):
    inpath = handle_input_file(training_dir, 'training_categorical_accuracy_plot', paths=PATHS)
    return digest_figure(inpath)


def load_training_loss_plot(training_dir):
    inpath = handle_input_file(training_dir, 'training_loss_plot', paths=PATHS)
    return digest_figure(inpath)


def load_training_lr_plot(training_dir):
    inpath = handle_input_file(training_dir, 'training_lr_plot', paths=PATHS)
    return digest_figure(inpath)


def load_training_summary_plot(training_dir):
    inpath = handle_input_file(training_dir, 'training_summary_plot', paths=PATHS)
    return digest_figure(inpath)


def load_class_reduce_plot(training_dir):
    inpath = handle_input_file(training_dir, 'class_reduce_plot', paths=PATHS)
    return digest_figure(inpath)


def load_kpconv_plots(training_dir):
    init_plots, end_plots = find_kpconv_plots(
        os.path.join(training_dir, PATHS['kpconv_plots'])
    )
    return init_plots, end_plots


def load_skpconv_plots(training_dir):
    init_plots, end_plots = find_kpconv_plots(
        os.path.join(training_dir, PATHS['skpconv_plots']),
        init_offset=9, end_offset=12
    )
    return init_plots, end_plots


def find_kpconv_plots(dirpath, init_offset=4, end_offset=7):
    children = os.listdir(dirpath)
    init_children = [child for child in children if child[:4].lower() == 'init']
    end_children = [child for child in children if child[:7].lower() == 'trained']
    init_children = [
        {
            'plot_prefix': child_rpath[init_offset+1:],
            'plot_W': digest_figure(os.path.join(dirpath, child_rpath, 'Winit_mat.svg')),
            'plot_Whist': digest_figure(os.path.join(dirpath, child_rpath, 'Winit_hist.svg')),
            'plot_Q': digest_figure(os.path.join(dirpath, child_rpath, 'Qinit.svg'))
        }
        for child_rpath in init_children
    ]
    end_children = [
        {
            'plot_prefix': child_rpath[end_offset+1:],
            'plot_W': digest_figure(os.path.join(dirpath, child_rpath, 'Wend_mat.svg')),
            'plot_W_diff': digest_figure(os.path.join(dirpath, child_rpath, 'Wdiff_mat.svg')),
            'plot_Whist': digest_figure(os.path.join(dirpath, child_rpath, 'Wend_hist.svg')),
            'plot_Whist_diff': digest_figure(os.path.join(dirpath, child_rpath, 'Wdiff_hist.svg')),
            'plot_Q': digest_figure(os.path.join(dirpath, child_rpath, 'Qend.svg'))
        }
        for child_rpath in end_children
    ]
    return init_children, end_children


def print_sql_inserts(analysis):
    # Insert model types
    training_json = analysis['training_json'].replace("'", "''")
    training_json_low = training_json.lower()
    family_name = 'Sequential'
    subfamily_name = 'PointNet'
    if training_json_low.find('convolutionalautoencoderpwiseclassifier'):
        family_name = 'Hierarchical autoencoder'
        if training_json_low.find('kpconv'):
            subfamily_name = 'KPConv'
    print(
        'INSERT INTO model_types '
        '(specification, family_id, subfamily_id, notes) VALUES\n'
        '\t('
        f"\t\t'{training_json}',\n"
        f"\t\t(SELECT id FROM model_families WHERE name = '{family_name}'),\n"
        f"\t\t(SELECT id FROM model_subfamilies WHERE name = '{subfamily_name}'),\n"
        f"\t\t'{analysis['model_name']}'\n"
        '\t) ON CONFLICT DO NOTHING;\n'
    )
    # Insert model
    summary = analysis['model_summary'].replace("'", "''")
    print(
        'INSERT INTO models '
        '(model_type_id, framework_id, model_summary, notes) VALUES\n'
        '\t('
        "\t\t(SELECT currval(pg_get_serial_sequence('model_types', 'id'))),\n"
        '\t\t1,\n'
        f'\t\t\'{summary}\',\n'
        f"\t\t'{analysis['model_name']}'\n"
        '\t) ON CONFLICT DO NOTHING;\n'
    )
    # Insert training_histories
    hist = analysis['training_history']
    print(
        'INSERT INTO training_histories '
        '(model_id, metric_id, num_epochs, start_value, end_value, '
        'min_value, min_value_epoch, max_value, max_value_epoch, '
        'mean_value, stdev_value, q1, q2, q3, q4, q5, q6, q7, q8, q9) VALUES'
    )
    for i, metric in enumerate(hist['metrics']):
        if i == 0:
            print('\t(')
        else:
            print('\t),(')
        print(
            "\t\t(SELECT currval(pg_get_serial_sequence('models', 'id'))),\n"
            f"\t\t(SELECT id FROM training_metrics WHERE name = '{metric['name']}'),\n"
            f'\t\t{hist["num_epochs"]},\n'
            f'\t\t{metric["start_value"]},\n'
            f'\t\t{metric["end_value"]},\n'
            f'\t\t{metric["min_value"]},\n'
            f'\t\t{metric["min_value_epoch"]},\n'
            f'\t\t{metric["max_value"]},\n'
            f'\t\t{metric["max_value_epoch"]},\n'
            f'\t\t{metric["mean_value"]},\n'
            f'\t\t{metric["stdev_value"]},\n'
            f'\t\t{",".join([str(q) for q in metric["Q"]])}'
        )
    print('\t) ON CONFLICT DO NOTHING;\n')
    # Insert training receptive field distribution
    trfdistr = analysis['trf_distribution']
    print(
        'INSERT INTO training_receptive_field_distributions '
        '(model_id, class_id, pred_count, pred_rf_count, ref_count, ref_rf_count) VALUES'
    )
    for i, distr in enumerate(trfdistr):
        print(
            '\t(\n'
            f"\t\t(SELECT currval(pg_get_serial_sequence('models', 'id'))),\n"
            f"\t\t(SELECT id FROM classes WHERE LOWER(classes.name) like '{distr['class_name']}'),\n"
            f'\t\t{distr["pred_count"]},\n'
            f'\t\t{distr["pred_rf_count"]},\n'
            f'\t\t{distr["ref_count"]},\n'
            f'\t\t{distr["ref_rf_count"]}\n',
            end=''
        )
        if i < len(trfdistr)-1:
            print('\t),')
        else:
            print('\t)')
    print('\tON CONFLICT DO NOTHING;\n')
    # Insert figures
    # TODO Rethink : Model graph is commented due to missing lib at FT-III
    #print_sql_insert_figure(
    #    analysis['model_graph'],
    #    'Model graph'
    #)
    print_sql_insert_figure(
        analysis['class_distribution_plot'],
        'Class distribution'
    )
    print_sql_insert_figure(
        analysis['confusion_matrix_plot'],
        'Training confusion matrix'
    )
    print_sql_insert_figure(
        analysis['trf_distribution_plot'],
        'Training receptive fields distribution'
    )
    print_sql_insert_figure(
        analysis['training_categorical_accuracy_plot'],
        'Categorical accuracy history'
    )
    print_sql_insert_figure(
        analysis['training_loss_plot'],
        'Loss history'
    )
    print_sql_insert_figure(
        analysis['training_lr_plot'],
        'Learning rate history'
    )
    print_sql_insert_figure(
        analysis['training_summary_plot'],
        'Training history summary'
    )
    print_sql_insert_figure(
        analysis['class_reduce_plot'],
        'Class reduction distribution'
    )
    # KPConv plots
    kpconv_init, kpconv_end = analysis['kpconv_plots']
    for kpconv_init in kpconv_init:
        print_sql_insert_figure(
            kpconv_init['plot_W'],
            f"{kpconv_init['plot_prefix']} init"
        )
        print_sql_insert_figure(
            kpconv_init['plot_Whist'],
            f"{kpconv_init['plot_prefix']} init hist"
        )
        print_sql_insert_figure(
            kpconv_init['plot_Q'],
            f"{kpconv_init['plot_prefix']} init Q"
        )
    for kpconv_end in kpconv_end:
        print_sql_insert_figure(
            kpconv_end['plot_W'],
            f"{kpconv_end['plot_prefix']} trained"
        )
        print_sql_insert_figure(
            kpconv_end['plot_Whist'],
            f"{kpconv_end['plot_prefix']} trained hist"
        )
        print_sql_insert_figure(
            kpconv_end['plot_Q'],
            f"{kpconv_end['plot_prefix']} trained Q"
        )
        print_sql_insert_figure(
            kpconv_end['plot_W_diff'],
            f"{kpconv_end['plot_prefix']} diff"
        )
        print_sql_insert_figure(
            kpconv_end['plot_Whist_diff'],
            f"{kpconv_end['plot_prefix']} diff hist"
        )
    # SKPConv plots
    skpconv_init, skpconv_end = analysis['skpconv_plots']
    for skpconv_init in skpconv_init:
        print_sql_insert_figure(
            skpconv_init['plot_W'],
            f"{skpconv_init['plot_prefix']} init"
        )
        print_sql_insert_figure(
            skpconv_init['plot_Whist'],
            f"{skpconv_init['plot_prefix']} init hist"
        )
        print_sql_insert_figure(
            skpconv_init['plot_Q'],
            f"{skpconv_init['plot_prefix']} init Q"
        )
    for skpconv_end in skpconv_end:
        print_sql_insert_figure(
            skpconv_end['plot_W'],
            f"{skpconv_end['plot_prefix']} trained"
        )
        print_sql_insert_figure(
            skpconv_end['plot_Whist'],
            f"{skpconv_end['plot_prefix']} trained hist"
        )
        print_sql_insert_figure(
            skpconv_end['plot_Q'],
            f"{skpconv_end['plot_prefix']} trained Q"
        )
        print_sql_insert_figure(
            skpconv_end['plot_W_diff'],
            f"{skpconv_end['plot_prefix']} diff"
        )
        print_sql_insert_figure(
            skpconv_end['plot_Whist_diff'],
            f"{skpconv_end['plot_prefix']} diff hist"
        )


def print_sql_insert_figure(figdict, plot_name):
    print(
        'INSERT INTO model_plots '
        '(model_id, plot_id, plot_bin, plot_format_id) VALUES\n'
        '\t(\n'
        f"\t\t(SELECT currval(pg_get_serial_sequence('models', 'id'))),\n"
        f"\t\t(SELECT id FROM plots WHERE name like '{plot_name}'),\n"
        f"\t\t'{figdict['bytea']}'::bytea,\n"
        f"\t\t(SELECT id FROM plot_formats WHERE LOWER(name) like '%{figdict['format']}%')\n"
        '\t) ON CONFLICT DO NOTHING;\n'
    )
    # Note that plots inserted like this must be obtained with a query similar
    # to: SELECT encode(plot_bin, 'escape') FROM resultset_plot


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    start = time.perf_counter()
    training_json_path, training_dir = parse_args()
    analysis = analyze_experiment(training_json_path, training_dir)
    print_sql_inserts(analysis)
    end = time.perf_counter()
    printerr(
        '\n\n'
        f'SQL insert from training script run in {end-start:.3f} seconds',
    )

