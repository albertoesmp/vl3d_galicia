# ----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Analyze all the point clouds inside a given directory
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


# ---  METHODS  --- #
# ----------------- #
def print_help():
    print(
'''USAGE of analyze_dataset.py

        1: Path to a directory containing many LAS/LAZ files

'''
    )


def parse_args():
    # Validate args
    if len(sys.argv) < 2:
        print('Not enough arguments were given.\n\n')
        print_help()
        sys.exit(1)
    # Validate path to directory
    dirpath = sys.argv[1]
    if not os.path.isdir(dirpath):
        print(f'Given path "{dirpath}" does not point to a directory.\n\n')
        print_help()
        sys.exit(1)
    # Return path to directory
    return dirpath


def analyze_pclouds(pclouds_dir):
    # Load geographic assets
    start = time.perf_counter()
    galicia_geom = assets.load_shape_geometry(region='galicia')
    coruna_geom = assets.load_shape_geometry(region='coruna')
    pvedra_geom = assets.load_shape_geometry(region='pvedra')
    lugo_geom = assets.load_shape_geometry(region='lugo')
    ourense_geom = assets.load_shape_geometry(region='ourense')
    end = time.perf_counter()
    print(f'Assets loaded in {end-start:.3f} seconds.\n\n')
    # Initialize counters
    counters = init_counters()
    # Iterate over LAS/LAZ files
    fnames = os.listdir(pclouds_dir)
    for i, fname in enumerate(fnames):
        start = time.perf_counter()
        # Check LAS/LAZ extension
        fname_low = fname.lower()
        if not (fname_low.endswith('.las') or fname_low.endswith('.laz')):
            continue
        # Check path is file
        fpath = os.path.join(pclouds_dir, fname)
        if not os.path.isfile(fpath):
            continue
        # Get 2D point-wise coordinates
        las = lasu.read_las(fpath)
        X = lasu.coordinates2D_from_las(las)
        y = lasu.labels_from_las(las)
        las = None
        gc.collect()
        # Update counters
        update_counters(
            counters, X, y, coruna_geom, pvedra_geom, lugo_geom, ourense_geom
        )
        # Report end of iteration
        end = time.perf_counter()
        print(
            f'Processed pcloud {i+1}/{len(fnames)} "{fname}" '
            f'in {end-start:.3f} seconds.\n'
        )
    # Return data
    output = {
        'galicia_area': galicia_geom.area,
        'galicia_points': counters['galicia_points'],
        'galicia_noise': counters['galicia_noise'],
        'galicia_overlap': counters['galicia_overlap'],
        'galicia_west_area': coruna_geom.area+pvedra_geom.area,
        'galicia_west_points': counters['coruna_points']+counters['pvedra_points'],
        'galicia_west_noise': counters['coruna_noise']+counters['pvedra_noise'],
        'galicia_west_overlap': counters['coruna_overlap']+counters['pvedra_overlap'],
        'galicia_east_area': lugo_geom.area+ourense_geom.area,
        'galicia_east_points': counters['lugo_points']+counters['ourense_points'],
        'galicia_east_noise': counters['lugo_noise']+counters['ourense_noise'],
        'galicia_east_overlap': counters['lugo_overlap']+counters['ourense_overlap'],
        'coruna_area': coruna_geom.area,
        'coruna_points': counters['coruna_points'],
        'coruna_noise': counters['coruna_noise'],
        'coruna_overlap': counters['coruna_overlap'],
        'pvedra_area': pvedra_geom.area,
        'pvedra_points': counters['pvedra_points'],
        'pvedra_noise': counters['pvedra_noise'],
        'pvedra_overlap': counters['pvedra_overlap'],
        'lugo_area': lugo_geom.area,
        'lugo_points': counters['lugo_points'],
        'lugo_noise': counters['lugo_noise'],
        'lugo_overlap': counters['lugo_overlap'],
        'ourense_area': ourense_geom.area,
        'ourense_points': counters['ourense_points'],
        'ourense_noise': counters['ourense_noise'],
        'ourense_overlap': counters['ourense_overlap'],
    }
    output['galicia_clean'] = output['galicia_points'] - \
        output['galicia_noise'] - output['galicia_overlap']
    output['galicia_west_clean'] = output['galicia_west_points'] - \
        output['galicia_west_noise'] - output['galicia_west_overlap']
    output['galicia_east_clean'] = output['galicia_east_points'] - \
        output['galicia_east_noise'] - output['galicia_east_overlap']
    output['coruna_clean'] = output['coruna_points'] - \
        output['coruna_noise'] - output['coruna_overlap']
    output['pvedra_clean'] = output['pvedra_points'] - \
        output['pvedra_noise'] - output['pvedra_overlap']
    output['lugo_clean'] = output['lugo_points'] - \
        output['lugo_noise'] - output['lugo_overlap']
    output['ourense_clean'] = output['ourense_points'] - \
        output['ourense_noise'] - output['ourense_overlap']
    return output


def init_counters():
    return {
        'galicia_points': 0,
        'galicia_noise': 0,
        'galicia_overlap': 0,
        'coruna_points': 0,
        'coruna_noise': 0,
        'coruna_overlap': 0,
        'pvedra_points': 0,
        'pvedra_noise': 0,
        'pvedra_overlap': 0,
        'lugo_points': 0,
        'lugo_noise': 0,
        'lugo_overlap': 0,
        'ourense_points': 0,
        'ourense_noise': 0,
        'ourense_overlap': 0
    }


def update_counters(
    counters, X, y, coruna_geom, pvedra_geom, lugo_geom, ourense_geom
):
    # Function to update counters for each subregion
    def update_subregion_counters(
        counters, X, y, geom, name, Xsup=None
    ):
        mask = points_inside_geometry(X, geom, Xsup=Xsup)
        Xmask = X[mask]
        ymask= y[mask]
        counters[f'{name}_points'] += len(Xmask)
        counters[f'{name}_noise'] += np.count_nonzero(
            ymask == classes.pnoa2['noise']
        )
        counters[f'{name}_overlap'] += np.count_nonzero(
            ymask == classes.pnoa2['overlap']
        )
    # Update Galicia counts
    counters['galicia_points'] += len(X)
    counters['galicia_noise'] += np.count_nonzero(y == classes.pnoa2['noise'])
    counters['galicia_overlap'] += np.count_nonzero(y == classes.pnoa2['overlap'])
    # Update counts for each province
    Xsup = []
    update_subregion_counters(counters, X, y, coruna_geom, 'coruna', Xsup=Xsup)
    Xsup = Xsup[0]
    update_subregion_counters(counters, X, y, pvedra_geom, 'pvedra', Xsup=Xsup)
    update_subregion_counters(counters, X, y, lugo_geom, 'lugo', Xsup=Xsup)
    update_subregion_counters(counters, X, y, ourense_geom, 'ourense', Xsup=Xsup)

def print_csv_format(analysis):
    print('\n')
    print(
        '{region:24s},{sensor:24s},{points:>16s},{area:>16s},{noise:>12s},'
        '{overlap:>12s},{clean:>12s}'.format(
            region='Region',
            sensor='Sensor',
            points='Points(10^6)',
            area='Area(km^2)',
            noise='Noise(%)',
            overlap='Overlap(%)',
            clean='Clean(%)'
        )
    )
    for region_id, region_name, sensor in zip(
        [
            'coruna', 'lugo', 'pvedra', 'ourense',
            'galicia_west', 'galicia_east', 'galicia'
        ],
        [
            'A Coruña', 'Lugo', 'Pontevedra', 'Ourense',
            'Galicia west', 'Galicia east', 'Galicia'
        ],
        [
            'LEICA ALS60', 'LEICA ALS80', 'LEICA ALS60', 'LEICA ALS80',
            'LEICA ALS60', 'LEICA ALS80', 'LEICA ALS60/80'
        ]
    ):
        points = analysis[f'{region_id}_points']
        noise = analysis[f'{region_id}_noise']/points if points != 0 else 0
        overlap = analysis[f'{region_id}_overlap']/points if points != 0 else 0
        clean = analysis[f'{region_id}_clean']/points if points != 0 else 0
        print(
            '{region:24s},{sensor:24s},{points:16.2f},{area:16.2f},{noise:12.2f},'
            '{overlap:12.2f},{clean:12.2f}'.format(
                region=region_name,
                sensor=sensor,
                points=points/1e6,
                area=analysis[f'{region_id}_area']/1e6,
                noise=100*noise,
                overlap=100*overlap,
                clean=100*clean
            )
        )

def print_latex_format(analysis):
    print('\nLatex table body:\n')
    for region_id, region_name, sensor in zip(
        [
            'coruna', 'lugo', 'pvedra', 'ourense',
            'galicia_west', 'galicia_east', 'galicia'
        ],
        [
            'A Coruña', 'Lugo', 'Pontevedra', 'Ourense',
            'Galicia west', 'Galicia east', 'Galicia'
        ],
        [
            'LEICA ALS60', 'LEICA ALS80', 'LEICA ALS60', 'LEICA ALS80',
            'LEICA ALS60', 'LEICA ALS80', 'LEICA ALS60/80'
        ]
    ):
        points = analysis[f'{region_id}_points']
        noise = analysis[f'{region_id}_noise']/points if points != 0 else 0
        overlap = analysis[f'{region_id}_overlap']/points if points != 0 else 0
        clean = analysis[f'{region_id}_clean']/points if points != 0 else 0
        print(
            '{region}&{sensor}&{points:.2f}&{area:.2f}&{noise:.2f}&'
            '{overlap:.2f}&{clean:.2f}\\\\'.format(
                region=region_name,
                sensor=sensor,
                points=points/1e6,
                area=analysis[f'{region_id}_area']/1e6,
                noise=100*noise,
                overlap=100*overlap,
                clean=100*clean
            )
        )
    print('\n-----------------------------------------------------------\n\n')

# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    start = time.perf_counter()
    pclouds_dir = parse_args()
    analysis = analyze_pclouds(pclouds_dir)
    print_latex_format(analysis)
    print_csv_format(analysis)
    end = time.perf_counter()
    print(f'\n\nAnalyze dataset script run in {end-start:.3f} seconds')

