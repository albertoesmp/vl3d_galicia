# ----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Utils to handle VL3D-Galicia project related assets
# ----------------------------------------------------------------------------

# ---   IMPORTS   --- #
# ------------------- #
from shapely.geometry import shape, Polygon, MultiPolygon
import pyproj
import json
import os


# --------------------------------------- #
# ---   P U B L I C   M E T H O D S   --- #
# --------------------------------------- #
def load_shape_geometry(region='galicia'):
    """
    :param region: Either galicia, coruna, pvedra, or ourense.
    :return: The shape representing the geometry of the region
    """
    region_low = region.lower()
    if region_low == 'galicia':
        return load_shape_from_geojson(locate_asset('galicia_boundary.geojson'))
    elif region_low == 'coruna':
        return load_shape_from_geojson(locate_asset('acoru_boundary.geojson'))
    elif region_low == 'lugo':
        return load_shape_from_geojson(locate_asset('lugo_boundary.geojson'))
    elif region_low == 'pvedra':
        return load_shape_from_geojson(locate_asset('pvedra_boundary.geojson'))
    elif region_low == 'ourense':
        return load_shape_from_geojson(locate_asset('ourense_boundary.geojson'))
    raise ValueError(
        'vl3dgal/assets.py load_shape_geometry FAILED to load for:\n'
        f'region="{region}"'
    )



# ------------------------------------- #
# ---   I N N E R   M E T H O D S   --- #
# ------------------------------------- #
def locate_asset(relpath):
    return os.path.join(os.path.dirname(__file__), 'assets', relpath)


def load_shape_from_geojson(fpath):
    # Read data
    geom = None
    with open(fpath, 'r') as gjsonf:
        gjson = json.load(gjsonf)
        geom = shape(gjson['features'][0]['geometry'])
    if geom is None:
        raise ValueError(
            'vl3dgal/assets.py load_shape_from_geojson FAILED to load:\n'
            f'"{fpath}"'
        )
    # Transform to point clouds CRS
    transformer = pyproj.Transformer.from_crs(
        'EPSG:4326',
        '+proj=utm +zone=29 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs',
        always_xy=True
    )
    if type(geom) == Polygon:
        geom = Polygon(
            transformer.transform(x, y) for x, y in geom.exterior.coords
        )
    elif type(geom) == MultiPolygon:
        polys = []
        for poly in geom.geoms:
            polys.append(Polygon(
                transformer.transform(x, y) for x, y in poly.exterior.coords
            ))
        geom = MultiPolygon(polys)
    else:
        raise ValueError(
            'vl3dgal/assets.py load_shape_from_geojson FAILED CRS transform.'
        )
    # Return
    return geom
