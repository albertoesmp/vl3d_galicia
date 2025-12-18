# ---------------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena
# BIREF: Analyze one point cloud per task and plot its MR vs CA curve.
# ---------------------------------------------------------------------------- #


# ---  IMPORTS  --- #
# ----------------- #
import scipy.stats as ss
import laspy
import numpy as np
import matplotlib.pyplot as plt
import time


# ---  CONSTANTS  --- #
# ------------------- #
# Path to the binary vegetation classification point cloud
VEGETATION_PATH='/oldext4/lidar_data/pnoa/second_edition/vl3d/out/cesga/vegetation/MERGE_59/uncertainty/uncertainty.laz'
# Path to the low-mid-high vegetation classification point cloud
LMHVEG_PATH='/oldext4/lidar_data/pnoa/second_edition/vl3d/out/cesga/lmhveg/MERGE_31/uncertainty/uncertainty.laz'
# Path to the binary building classification point cloud
BUILDING_PATH='/oldext4/lidar_data/pnoa/second_edition/vl3d/out/cesga/building/MERGE_225/uncertainty/uncertainty.laz'
# Path to the building and vegetation classification point cloud
BUILDVEG_PATH='/oldext4/lidar_data/pnoa/second_edition/vl3d/out/cesga/buildveg/MERGE_101/uncertainty/uncertainty.laz'
# Path to the full classification point cloud
FULL_PATH='/oldext4/lidar_data/pnoa/second_edition/vl3d/out/cesga/fullraw/sflnet/MERGE_232/uncertainty/uncertainty.laz'


# ---  FUNCTIONS  --- #
# ------------------- #
def quantify_correlations(ca_bins, mr_bins):
    # Standardize data
    ca_std = (ca_bins - np.mean(ca_bins)) / np.std(ca_bins)
    mr_std = (mr_bins - np.mean(mr_bins)) / np.std(mr_bins)
    # Quantify correlations
    pearson = ss.pearsonr(ca_std, mr_std)
    spearman = ss.spearmanr(ca_bins, mr_bins)
    # Report correlations
    print(
        f'Pearson correlation: r = {pearson.statistic:.3f}, '
        f'p-value = {pearson.pvalue:.2E}\n'
        f'Spearman correlation: r = {spearman.statistic:.3f}, '
        f'p-value = {spearman.pvalue:.2E}'
    )

def plot_mr_ca_curve(ax, path, color, label):
    # Read las file
    start = time.perf_counter()
    las = laspy.read(path)
    end = time.perf_counter()
    print(
        f'Point cloud "{path}" with {len(las.X)} points read in '
        f'{end-start:.3f} seconds.'
    )
    # Retrieve Class Ambiguity (CA) and error mask
    start = time.perf_counter()
    ca = las['ClassAmbiguity']
    success = las['Success']
    # Compute 33 CA bins
    ca_bins = np.linspace(0, 1, 33)
    # Compute Misclassification Rate (MR) from error mask and CA bins
    mr_bins = np.zeros(ca_bins.shape[0])
    for i, ca_bini in enumerate(ca_bins):
        mask = ca <= ca_bini
        total_count = np.count_nonzero(mask)
        error_count = np.count_nonzero(success[mask]==0)
        try:
            mr_bins[i] = error_count/total_count*100.0
        except ZeroDivisionError as zderr:
            mr_bins[i] = 0
    end = time.perf_counter()
    print(
        f'Point cloud "{path}" of {ca.shape[0]} points '
        f'processed in {end-start:.3f} seconds.'
    )
    # Plot CA vs MR curve
    ax.plot(ca_bins, mr_bins, color=color, label=label, lw=2)
    # Compute correlations
    start = time.perf_counter()
    quantify_correlations(ca_bins, mr_bins)
    end = time.perf_counter()
    print(f'Correlations for "{path}" computed in {end-start:.3f} seconds.')


# ---   MAIN   --- #
# ---------------- #
if __name__ == '__main__':
    # Prepare fig
    fig = plt.figure(figsize=(4, 3))
    # Prepare axes
    ax = fig.add_subplot(1, 1, 1)
    # Plot data
    plot_mr_ca_curve(ax, VEGETATION_PATH, 'tab:green', 'Vegetation')
    plot_mr_ca_curve(ax, LMHVEG_PATH, 'tab:olive', 'LMH-Veg')
    plot_mr_ca_curve(ax, BUILDING_PATH, 'tab:red', 'Building')
    plot_mr_ca_curve(ax, BUILDVEG_PATH, 'tab:orange', 'Build/Veg')
    plot_mr_ca_curve(ax, FULL_PATH, 'tab:blue', 'Full')
    # Format plot
    ax.set_axisbelow(True)
    ax.set_xlabel('Class ambiguity')
    ax.set_ylabel('Misclassification rate (%)')
    ax.grid('both')
    ax.legend(loc='best')
    ax.set_xticks(np.linspace(0, 1, 5))
    #ax.set_yticks(np.linspace(0, 100, 5))
    # Format fig
    fig.tight_layout()
    # Save and show
    plt.savefig('/tmp/ca_vs_mr.jpg', dpi=300)
    plt.show()

