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
            counters, X, y, coruna_geom, pvedra_geom, lugo_geom, ourense_geom,
            fname
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
        'scenes': counters['scenes']
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
        'ourense_overlap': 0,
        'scenes': []
    }


def update_counters(
    counters, X, y, coruna_geom, pvedra_geom, lugo_geom, ourense_geom, fname
):
    # Function to update counters for each subregion
    def update_subregion_counters(
        counters, X, y, geom, name, Xsup=None
    ):
        mask = points_inside_geometry(X, geom, Xsup=Xsup)
        Xmask = X[mask]
        ymask = y[mask]
        counters[f'{name}_points'] += len(Xmask)
        counters[f'{name}_noise'] += np.count_nonzero(
            ymask == classes.pnoa2['noise']
        )
        counters[f'{name}_overlap'] += np.count_nonzero(
            ymask == classes.pnoa2['overlap']
        )
        return len(Xmask)
    # Update Galicia counts
    galicia_points = len(X)
    galicia_unclassified = np.count_nonzero(y == classes.pnoa2['unclassified'])
    galicia_ground = np.count_nonzero(y == classes.pnoa2['ground'])
    galicia_lowveg = np.count_nonzero(y == classes.pnoa2['lowveg'])
    galicia_midveg = np.count_nonzero(y == classes.pnoa2['midveg'])
    galicia_highveg = np.count_nonzero(y == classes.pnoa2['highveg'])
    galicia_building = np.count_nonzero(y == classes.pnoa2['building'])
    galicia_noise = np.count_nonzero(y == classes.pnoa2['noise'])
    galicia_water = np.count_nonzero(y == classes.pnoa2['water'])
    galicia_overlap = np.count_nonzero(y == classes.pnoa2['overlap'])
    galicia_bridge = np.count_nonzero(y == classes.pnoa2['bridge'])
    counters['galicia_points'] += galicia_points
    counters['galicia_noise'] += galicia_noise
    counters['galicia_overlap'] += galicia_overlap
    # Update counts for each province
    Xsup = []
    coruna_points = update_subregion_counters(
        counters, X, y, coruna_geom, 'coruna', Xsup=Xsup
    )
    Xsup = Xsup[0]
    pvedra_points = update_subregion_counters(
        counters, X, y, pvedra_geom, 'pvedra', Xsup=Xsup
    )
    lugo_points = update_subregion_counters(
        counters, X, y, lugo_geom, 'lugo', Xsup=Xsup
    )
    ourense_points = update_subregion_counters(
        counters, X, y, ourense_geom, 'ourense', Xsup=Xsup
    )
    # Update scene counts (and other data)
    a, b = np.min(X, axis=0), np.max(X, axis=0)
    counters['scenes'].append({
        'name': fname,
        'points': galicia_points,
        'coruna_points': coruna_points,
        'pvedra_points': pvedra_points,
        'lugo_points': lugo_points,
        'ourense_points': ourense_points,
        'unclassified': galicia_unclassified,
        'ground': galicia_ground,
        'lowveg': galicia_lowveg,
        'midveg': galicia_midveg,
        'highveg': galicia_highveg,
        'building': galicia_building,
        'noise': galicia_noise,
        'water': galicia_water,
        'overlap': galicia_overlap,
        'bridge': galicia_bridge,
        'pxmin': a[0],
        'pymin': a[1],
        'pxmax': b[0],
        'pymax': b[1]
    })


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
            '{region} & {sensor} & {points:.2f} & {area:.2f} & {noise:.2f} & '
            '{overlap:.2f} & {clean:.2f} \\\\'.format(
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


def export_sql_inserts(analysis, pclouds_dir):
    # Prepare variables
    outpath = os.path.join(pclouds_dir, 'ddbb_inserts.sql')
    scenes = analysis['scenes']
    # Open file for write
    with open(outpath, 'w') as outf:
        export_sql_datasets(scenes, outf)
        export_sql_dataset_metas(scenes, outf)
        export_sql_dataset_regions(scenes, outf)
        export_sql_class_distributions(scenes, outf)
    print(f'SQL insert script exported to "{outpath}"')


def export_sql_datasets(scenes, outf):
    num_scenes_minus_one = len(scenes)-1
    num_classif_types_minus_one = len(classes.CLASSIF_TYPES)-1
    # Write INSERT senentece for datasets table
    outf.write(
        '-- TABLE: datasets\n'
        'INSERT INTO datasets '
        '(name, num_points, pxmin, pymin, pxmax, pymax) VALUES\n'
    )
    # Add VALUES to INSERT sentence for datasets table for each scene
    for i, scene in enumerate(scenes):
        for j, classif_type in enumerate(classes.CLASSIF_TYPES):
            outf.write(
                '\t('
                f"'{scene['name'][:-4]}_{classif_type}',"
                f'{scene["points"]},'
                f'{scene["pxmin"]},'
                f'{scene["pymin"]},'
                f'{scene["pxmax"]},'
                f'{scene["pymax"]}'
                ')'
            )
            if i < num_scenes_minus_one or j < num_classif_types_minus_one:
                outf.write(',\n')
    outf.write('\n\tON CONFLICT DO NOTHING;\n')


def export_sql_dataset_metas(scenes, outf):
    # Write INSERT sentence for dataset_metas table
    outf.write(
        '\n-- TABLE: dataset_metas\n'
        'INSERT INTO dataset_metas (dataset_id, meta_id) VALUES\n'
    )
    # Add VALUES to INSERT sentence for dataset_metas table
    for i, scene in enumerate(scenes):
        for j, classif_type in enumerate(classes.CLASSIF_TYPES):
            if i != 0 or j != 0:
                outf.write(',\n')
            outf.write(
                '\t('
                '\n\t\t(SELECT id FROM datasets WHERE name like '
                f"'{scene['name'][:-4]}_{classif_type}'),"
                '\n\t\t(SELECT id FROM metadatasets WHERE name like '
                "'PNOA-II GALICIA')"
                ')'
            )
    outf.write('\n\tON CONFLICT DO NOTHING;\n')


def export_sql_dataset_regions(scenes, outf):
    # Write INSERT sentence for dataset_regions table
    outf.write(
        '\n-- TABLE: dataset_regions\n'
        'INSERT INTO dataset_regions '
        '(dataset_id, region_id, num_points) VALUES\n\t'
    )
    # Add VALUES to INSERT sentence for dataset_regions table
    write_comma = False
    for i, scene in enumerate(scenes):
        for classif_type in classes.CLASSIF_TYPES:
            write_comma = _export_sql_dataset_regions(
                scene, outf, f'{scene["name"][:-4]}_{classif_type}',
                'points', 'galicia', write_comma
            )
            write_comma = _export_sql_dataset_regions(
                scene, outf, f'{scene["name"][:-4]}_{classif_type}',
                'coruna_points', 'a coruña', write_comma
            )
            write_comma = _export_sql_dataset_regions(
                scene, outf, f'{scene["name"][:-4]}_{classif_type}',
                'pvedra_points', 'pontevedra', write_comma
            )
            write_comma = _export_sql_dataset_regions(
                scene, outf, f'{scene["name"][:-4]}_{classif_type}',
                'lugo_points', 'lugo', write_comma
            )
            write_comma = _export_sql_dataset_regions(
                scene, outf, f'{scene["name"][:-4]}_{classif_type}',
                'ourense_points', 'ourense', write_comma
            )
    outf.write('\n\tON CONFLICT DO NOTHING;\n')


def _export_sql_dataset_regions(scene, outf, name, key, strlike, write_comma):
    if scene[key] > 0:
        if write_comma:
            outf.write(',(\n')
        else:
            outf.write('(\n')
        outf.write(
            '\t\t(SELECT id FROM datasets '
            f"WHERE name like '{name}'),\n"
            '\t\t(SELECT id FROM geographic_regions '
            f"WHERE LOWER(name) like '%{strlike}'),\n"
            f'\t\t{scene[key]}\n\t)'
        )
        write_comma = True
    return write_comma

def export_sql_class_distributions(scenes, outf):
    # Write INSERT sentence for class_distributions table
    outf.write(
        '\n-- TABLE: class_distributions\n'
        'INSERT INTO class_distributions (dataset_id, class_id, recount) '
        'VALUES\n\t'
    )
    # Add VALUES to INSERT sentence for class_distributions table
    write_comma = False
    export_funs = [
        _export_class_distribution_original,
        _export_class_distribution_vegetation,
        _export_class_distribution_lmh_vegetation,
        _export_class_distribution_building,
        _export_class_distribution_build_veg
    ]
    for i, scene in enumerate(scenes):
        for export_fun in export_funs:
            write_comma = export_fun(scene, outf, write_comma)
    outf.write('\n\tON CONFLICT DO NOTHING;\n')


def _export_class_distribution_original(scene, outf, write_comma):
    dataset_name = f"{scene['name'][:-4]}_ORIGINAL"
    class_keys = [key for key in classes.pnoa2.keys()]
    for class_key in class_keys:
        if scene[class_key] > 0:
            if write_comma:
                outf.write(',(\n')
            else:
                outf.write('(\n')
            outf.write(
                '\t\t(SELECT id FROM datasets '
                f"WHERE name like '{dataset_name}'),\n"
                '\t\t(SELECT id FROM classes '
                f"WHERE LOWER(name) like '{classes.CLASS_NAMES[class_key].lower()}'),\n"
                f'\t\t{scene[class_key]}\n\t)'
            )
            write_comma = True
    return write_comma


def _export_class_distribution_vegetation(scene, outf, write_comma):
    dataset_name = f"{scene['name'][:-4]}_VEGETATION"
    class_keys = ['vegetation', 'other', 'ignore']
    assoc_class_keys = [
        ['lowveg', 'midveg', 'highveg'],
        ['ground', 'building', 'water', 'bridge'],
        ['unclassified', 'noise', 'overlap']
    ]
    for i, class_key in enumerate(class_keys):
        count = 0
        for assoc_class_key in assoc_class_keys[i]:
            count += scene[assoc_class_key]
        if count > 0:
            if write_comma:
                outf.write(',(\n')
            else:
                outf.write('(\n')
            outf.write(
                '\t\t(SELECT id FROM datasets '
                f"WHERE name like '{dataset_name}'),\n"
                '\t\t(SELECT id FROM classes '
                f"WHERE LOWER(name) like '{classes.CLASS_NAMES[class_key].lower()}'),\n"
                f'\t\t{count}\n\t)'
            )
            write_comma = True
    return write_comma


def _export_class_distribution_lmh_vegetation(scene, outf, write_comma):
    dataset_name = f"{scene['name'][:-4]}_LMH_VEGETATION"
    class_keys = ['lowveg', 'midveg', 'highveg', 'other', 'ignore']
    assoc_class_keys = [
        ['lowveg'],
        ['midveg'],
        ['highveg'],
        ['ground', 'building', 'water', 'bridge'],
        ['unclassified', 'noise', 'overlap']
    ]
    for i, class_key in enumerate(class_keys):
        count = 0
        for assoc_class_key in assoc_class_keys[i]:
            count += scene[assoc_class_key]
        if count > 0:
            if write_comma:
                outf.write(',(\n')
            else:
                outf.write('(\n')
            outf.write(
                '\t\t(SELECT id FROM datasets '
                f"WHERE name like '{dataset_name}'),\n"
                '\t\t(SELECT id FROM classes '
                f"WHERE LOWER(name) like '{classes.CLASS_NAMES[class_key].lower()}'),\n"
                f'\t\t{count}\n\t)'
            )
            write_comma = True
    return write_comma


def _export_class_distribution_building(scene, outf, write_comma):
    dataset_name = f"{scene['name'][:-4]}_BUILDING"
    class_keys = ['building', 'other', 'ignore']
    assoc_class_keys = [
        ['building'],
        ['ground', 'lowveg', 'midveg', 'highveg', 'water', 'bridge'],
        ['unclassified', 'noise', 'overlap']
    ]
    for i, class_key in enumerate(class_keys):
        count = 0
        for assoc_class_key in assoc_class_keys[i]:
            count += scene[assoc_class_key]
        if count > 0:
            if write_comma:
                outf.write(',(\n')
            else:
                outf.write('(\n')
            outf.write(
                '\t\t(SELECT id FROM datasets '
                f"WHERE name like '{dataset_name}'),\n"
                '\t\t(SELECT id FROM classes '
                f"WHERE LOWER(name) like '{classes.CLASS_NAMES[class_key].lower()}'),\n"
                f'\t\t{count}\n\t)'
            )
            write_comma = True
    return write_comma


def _export_class_distribution_build_veg(scene, outf, write_comma):
    dataset_name = f"{scene['name'][:-4]}_BUILD_VEG"
    class_keys = ['building', 'vegetation', 'other', 'ignore']
    assoc_class_keys = [
        ['building'],
        ['lowveg', 'midveg', 'highveg'],
        ['ground', 'water', 'bridge'],
        ['unclassified', 'noise', 'overlap']
    ]
    for i, class_key in enumerate(class_keys):
        count = 0
        for assoc_class_key in assoc_class_keys[i]:
            count += scene[assoc_class_key]
        if count > 0:
            if write_comma:
                outf.write(',(\n')
            else:
                outf.write('(\n')
            outf.write(
                '\t\t(SELECT id FROM datasets '
                f"WHERE name like '{dataset_name}'),\n"
                '\t\t(SELECT id FROM classes '
                f"WHERE LOWER(name) like '{classes.CLASS_NAMES[class_key].lower()}'),\n"
                f'\t\t{count}\n\t)'
            )
            write_comma = True
    return write_comma


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    start = time.perf_counter()
    pclouds_dir = parse_args()
    analysis = analyze_pclouds(pclouds_dir)
    print_latex_format(analysis)
    print_csv_format(analysis)
    export_sql_inserts(analysis, pclouds_dir)
    end = time.perf_counter()
    print(f'\n\nAnalyze dataset script run in {end-start:.3f} seconds')

