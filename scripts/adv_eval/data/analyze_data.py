"""
:author: Alberto M. Esmoris
:brief: Script for uncertainty assessment on point clouds
"""
# ---   IMPORTS   --- #
# ------------------- #
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import sys

# DISABLE WARNINGS (clean stdout)
import warnings
warnings.filterwarnings("ignore")


# ---   SPECIFICATION   --- #
# ------------------------- #
# Names for each case that must be processed
CASE_NAME = [
    'vegetation_286 (pwe)',
    'vegetation_286 (ca)',
    'lmhveg_49 (pwe)',
    'lmhveg_49 (ca)',
    'building_241 (pwe)',
    'building_241 (ca)',
    'buildveg_34 (pwe)',
    'buildveg_34 (ca)'
]

# Path to the input global eval files
INPUT_GLOBAL_EVAL = [
    '/home/uadmin/git/vl3dgal/scripts/adv_eval/data/vegetation_286/pwe/report/global_eval.log',
    '/home/uadmin/git/vl3dgal/scripts/adv_eval/data/vegetation_286/ca/report/global_eval.log',
    '/home/uadmin/git/vl3dgal/scripts/adv_eval/data/lmhveg_49/pwe/report/global_eval.log',
    '/home/uadmin/git/vl3dgal/scripts/adv_eval/data/lmhveg_49/ca/report/global_eval.log',
    '/home/uadmin/git/vl3dgal/scripts/adv_eval/data/building_241/pwe/report/global_eval.log',
    '/home/uadmin/git/vl3dgal/scripts/adv_eval/data/building_241/ca/report/global_eval.log',
    '/home/uadmin/git/vl3dgal/scripts/adv_eval/data/buildveg_34/pwe/report/global_eval.log',
    '/home/uadmin/git/vl3dgal/scripts/adv_eval/data/buildveg_34/ca/report/global_eval.log'
]

# The name of the domain (variable in the x-axis)
DOMAIN_NAME = [
    'Entropy cut',
    'Class ambiguity cut',
    'Entropy cut',
    'Class ambiguity cut',
    'Entropy cut',
    'Class ambiguity cut',
    'Entropy cut',
    'Class ambiguity cut'
]

# The metric to be considered for the quantifications
QUANTIFICATION_METRIC = [
    'F1',
    'F1',
    'F1',
    'F1',
    'F1',
    'F1',
    'F1',
    'F1'
]


# The metrics to be considered for the plots
PLOT_METRICS = [
    ['OA', 'F1', 'IoU', 'MCC'],
    ['OA', 'F1', 'IoU', 'MCC'],
    ['OA', 'F1', 'IoU', 'MCC'],
    ['OA', 'F1', 'IoU', 'MCC'],
    ['OA', 'F1', 'IoU', 'MCC'],
    ['OA', 'F1', 'IoU', 'MCC'],
    ['OA', 'F1', 'IoU', 'MCC'],
    ['OA', 'F1', 'IoU', 'MCC']
]


# ---   METHODS   --- #
# ------------------- #
def get_input_dim():
    """Obtain the number of input cases"""
    return len(INPUT_GLOBAL_EVAL)


def read_global_eval(global_eval_path, domain_name):
    """read a global evaluation text file and return a dict with 3 elements:
        x -> list with values in the domain
        m -> list with number of points
        metrics -> list of dict with K (metric name) and V (metric value)
    """
    # Open the file for reading
    with open(global_eval_path, 'r') as f:
        # Skip first 2 lines
        for i in range(2):
            f.readline()
        # Parse values
        cols = [
            f'{domain_name}', 'Number of points',
            'OA', 'P', 'R', 'F1', 'IoU',  # Evaluation metrics
            'wP', 'wR', 'wF1', 'wIoU',  # Weighted evaluation metrics
            'MCC', 'Kappa'  # Correlation metrics

        ]
        s = f.readline()  # Read first line
        x, m, metrics = [], [], []
        while len(s) > 0:
            # Parse current line
            split = s.split(',')
            x.append(float(split[0]))  # Get value in the domain
            m.append(int(split[1]))  # Get number of points
            metrics.append(dict(zip(cols[2:], [float(z) for z in split[2:]])))
            # Read next line
            s = f.readline()
    # Return output dictionary
    return {'x': x, 'm': m, 'metrics': metrics}


def read_data(i):
    """return a dictionary with the read data"""
    domain_name = DOMAIN_NAME[i]
    return {
        'global_eval': read_global_eval(INPUT_GLOBAL_EVAL[i], domain_name)
    }


def quantify(data, domain_name, metric_name):
    """quantify the data through stdout considering the requested metric"""
    # Extract data
    x = np.array(data['global_eval']['x'])
    metric = np.array([z[metric_name] for z in data['global_eval']['metrics']])
    # Standardize data
    x_std = (x - np.mean(x)) / np.std(x)
    metric_std = (metric - np.mean(metric)) / np.std(metric)
    # Start quantifications
    print(f'Quantifying {metric_name}({domain_name}):')
    # Quantify pearson correlation (linear relationship)
    pearson = ss.pearsonr(x_std, metric_std)
    print(
        f'Pearson correlation:  r = {pearson.statistic:.3f}, '
        f'p-value = {pearson.pvalue:.2E}'
    )
    # Quantify spearman correlation (monotonicity relationship)
    spearman = ss.spearmanr(x, metric)
    print(
        f'Spearman correlation: rho = {spearman.statistic:.3f}, '
        f'p-value = {spearman.pvalue:.2E}'
    )


def graph(data, domain_name, metric_names):
    # Extract data
    x = np.array(data['global_eval']['x'])
    metrics = np.array([
        [z[metric_name] for z in data['global_eval']['metrics']]
        for metric_name in metric_names
    ])
    # Initialize figure
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1)
    # Plot data
    zorder, lw, markersize, marker = 3, 18, 32, 'X'
    for name, metric in zip(metric_names, metrics):
        ax.plot(
            x, metric, lw=lw, ls='-', label=name, zorder=zorder,
            marker=marker, markersize=markersize, markeredgecolor='black'
        )
        if marker == 'X':
            zorder, lw, markersize, marker = 4, 12, 20, 'v'
        else:
            zorder, lw, markersize, marker = 3, 20, 30, 'X'
    # Format axes
    ax.grid('both', linewidth=2)
    ax.set_axisbelow(True)
    # TODO Restore : Legend below
    #legend = ax.legend(loc='best', prop={'weight': 'bold', 'size': 30}, ncol=1)
    #legend.get_frame().set_linewidth(4)
    ax.set_xlabel(domain_name, fontsize=32, fontweight='bold')
    ax.set_ylabel('Metric (%)', fontsize=32, fontweight='bold')
    ax.tick_params(axis='both', which='both', labelsize=30, width=4, length=8)
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)
    # Format figure
    fig.tight_layout()
    # Show
    plt.show()


# ---     MAIN     --- #
# -------------------- #
if __name__ == '__main__':
    # Digest each input case
    for i in range(get_input_dim()):
        # Show name of the case being processed
        print(f'Analyzing "{CASE_NAME[i]}" ...')
        # Read and process the data
        data = read_data(i)
        # Generate the output
        quantify(  # Quantify through stdout
            data,
            DOMAIN_NAME[i],
            QUANTIFICATION_METRIC[i]
        )
        graph(  # Plots with GUI
            data,
            DOMAIN_NAME[i],
            PLOT_METRICS[i]
        )
        print('\n-----------------------------------------------\n')
    # Successfully finished
    sys.exit(os.EX_OK)
