#!/bin/python3

# ---   IMPORTS   --- #
# ------------------- #
import numpy as np


# ---   GEOMETRIC GENERATORS   --- #
# -------------------------------- #
def torus(
    major_radius=2.0, minor_radius=1.0, num_points=1024, translate=np.zeros(3)
):
    steps = int(np.ceil(np.sqrt(num_points)))
    theta = np.linspace(-np.pi, np.pi, steps)
    theta, phi = [x.flatten() for x in np.meshgrid(theta, theta)]
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.array([
        (major_radius+minor_radius*cos_theta) * cos_phi,
        (major_radius+minor_radius*cos_theta) * sin_phi,
        minor_radius * sin_theta
    ]).T + translate


def onesheeted_hyperboloid(
    skirt_radius=1.0, height=1.0, num_points=1024, translate=np.zeros(3)
):
    steps = int(np.ceil(np.sqrt(num_points)))
    u = np.linspace(-height/2.0, height/2.0, steps)
    v = np.linspace(-np.pi, np.pi, steps)
    u, v = [x.flatten() for x in np.meshgrid(u, v)]
    sqrt = np.sqrt(np.square(u)+1.0)
    return np.array([
        skirt_radius * sqrt * np.cos(v),
        sqrt * np.sin(v),
        u
    ]).T + translate


def horizontal_plane(
    width=1.0, length=1.0, height=1.0, num_points=1024, translate=np.zeros(3)
):
    steps = int(np.ceil(np.sqrt(num_points)))
    u = np.linspace(-width/2.0, width/2.0, steps)
    v = np.linspace(-length/2.0, length/2.0, steps)
    u, v = [x.flatten() for x in np.meshgrid(u, v)]
    return np.array([
        u,
        v,
        np.ones_like(u)*height
    ]).T + translate


def vertical_plane(
    width=1.0, length=1.0, height=1.0, num_points=1024, translate=np.zeros(3)
):
    steps = int(np.ceil(np.sqrt(num_points)))
    u = np.linspace(-width/2.0, width/2.0, steps)
    v = np.linspace(-height/2.0, height/2.0, steps)
    u, v = [x.flatten() for x in np.meshgrid(u, v)]
    return np.array([
        u,
        np.ones_like(u)*length,
        v
    ]).T + translate


def sphere(
    radius=1.0, num_points=1024, translate=np.zeros(3)
):
    steps = int(np.ceil(np.sqrt(num_points)))
    u = np.linspace(0, np.pi, steps)
    v = np.linspace(-np.pi, np.pi, steps)
    u, v = [x.flatten() for x in np.meshgrid(u, v)]
    sin_u = np.sin(u)
    cos_u = np.cos(u)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    return np.array([
        radius*sin_u*cos_v,
        radius*sin_u*sin_v,
        radius*cos_u
    ]).T + translate


def hypocycloid(
    small_radius=1.0, cusps=11, height=0.0,
    num_points=1024, translate=np.zeros(3)
):
    theta = np.linspace(-np.pi, np.pi, num_points)
    large_radius = cusps*small_radius
    radius_diff = large_radius-small_radius
    return np.array([
        radius_diff * np.cos(theta) + small_radius * np.cos(
            radius_diff/small_radius * theta),
        radius_diff * np.sin(theta) - small_radius * np.sin(
            radius_diff/small_radius * theta),
        np.ones_like(theta)*height

    ]).T + translate


# ---   GEOMETRIC SPECIFICATIONS   --- #
# ------------------------------------ #
OUTPUT_PATH = 'Octree_X.xyz'
GEOMETRIC_SPECIFICATIONS = [
    {
        'generator': torus,
        'generator_args': {
            'major_radius': 5.0,
            'minor_radius': 2.0,
            'num_points': 4356,
            'translate': np.array([-33.0, 0.0, 1.0])
        }
    },
    {
        'generator': torus,
        'generator_args': {
            'major_radius': 2.0,
            'minor_radius': 1.0,
            'num_points': 3364,
            'translate': np.array([-11.0, 0.0, 0.5])
        }
    },
    {
        'generator': torus,
        'generator_args': {
            'major_radius': 5.0,
            'minor_radius': 2.0,
            'num_points': 4356,
            'translate': np.array([33.0, 0.0, 1.0])
        }
    },
    {
        'generator': torus,
        'generator_args': {
            'major_radius': 2.0,
            'minor_radius': 1.0,
            'num_points': 3364,
            'translate': np.array([11.0, 0.0, 0.5])
        }
    },
    {
        'generator': onesheeted_hyperboloid,
        'generator_args': {
            'skirt_radius': 1.0,
            'height': 7.0,
            'num_points': 12321,
            'translate': np.array([0.0, 0.0, 0.0])
        }
    },
    {
        'generator': horizontal_plane,
        'generator_args': {
            'width': 100.0,
            'length': 100.0,
            'height': -3.5,
            'num_points': 110889,
            'translate': np.array([0.0, 0.0, 0.0])
        }
    },
    {
        'generator': vertical_plane,
        'generator_args': {
            'width': 75.0,
            'length': 25.0,
            'height': 33.33,
            'num_points': 9801,
            'translate': np.array([0.0, 0.0, 13.165])
        }
    },
    {
        'generator': vertical_plane,
        'generator_args': {
            'width': 50.00,
            'length': -25.0,
            'height': 20.00,
            'num_points': 10201,
            'translate': np.array([0.0, 0.0, 6.5])
        }
    },
    {
        'generator': sphere,
        'generator_args': {
            'radius': 11.11,
            'num_points': 10201,
            'translate': np.array([0.0, 0.0, 111.11])
        }
    },
    {
        'generator': hypocycloid,
        'generator_args': {
            'small_radius': 1.1,
            'cusps': 33,
            'height': 66.67,
            'num_points': 1089,
            'translate': np.array([0.0, 0.0, 0.0])
        }
    }
]


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    # Generate point clouds
    X = []
    for geom_spec in GEOMETRIC_SPECIFICATIONS:
        X.append(geom_spec['generator'](**geom_spec['generator_args']))
    X = np.vstack(X)
    # Export point cloud
    np.savetxt(OUTPUT_PATH, X, fmt='%.6f', delimiter=',')

