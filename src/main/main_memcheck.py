#!/bin/python3
"""
:author: Alberto M. Esmoris Pena
:brief: Stand-alone script to check C++ memory handling of pyvl3dpp library.

Note that this script must be called straight forward and not called from
through vl3d.py. The rationale behind this is that loading the framework
implies loading joblib, tensorflow, and many other libraries that generate
entries in memory profilers and sanitizers (e.g., valgrind, ASAN, etc.).
The purpose of this script is to provide a clean environment to analyze the
memory management of pyvl3dpp.

WARNING! This script requires to manually set the PYTHONPATH to include the
cpp/build directory because it is a raw script that does not load the framework.
Consequently, it does not load the vl3dpp loader neither.
"""

# ---   IMPORTS   --- #
# ------------------- #
import pyvl3dpp as vl3dpp
import numpy as np
import sys


# ---   MAIN   --- #
# ---------------- #
if __name__ == '__main__':
    # Generate test data
    m = 8192  # Number of points
    num_classes = 5
    X = np.random.uniform(-0.5, 0.5, (m, 3)).astype(np.float32)
    F = np.random.normal(0, 1, (m, 2)).astype(np.float32)
    y = np.random.randint(0, num_classes, m).astype(np.int32)
    # Configure pre-processing
    to_unit_sphere = False
    num_points_per_depth = [256, 64, 16]
    num_downsampling_neighbors = [1, 16, 16]
    num_upsampling_neighbors = [1, 16, 16]
    num_pwise_neighbors = [8, 8, 8]
    fast_flag_per_depth = [False, False, False]
    neighborhood_spec = {
        'type': 'sphere',
        'radius': 0.33,
        'separation_factor': 0.8
    }
    radii = np.array([neighborhood_spec['radius']])
    support_strategy = 'fps'
    support_strategy_num_points = 64
    support_strategy_fast = False
    training_class_distribution = np.array([], dtype=np.int32)
    center_on_pcloud = True
    oversamplingArgs = [
        16,  # Min points
        num_points_per_depth[0],  # Target points,
        'knn',  # Oversampling strategy,
        8,  # k for knn strategies,
        1.0,  # Radius for spherical strategies,
        1  # Number of threads for oversampling in particular
    ]
    nthreads = -1  # Number of threads for pre-processing in general
    # Call C++ hierarchical fps pre-processing
    preout = vl3dpp.rf_dl_hfps_preproc_Xff_Ff_Iu32u32_ys32(
        X,
        F,
        y,
        num_classes,
        to_unit_sphere,
        np.array(num_points_per_depth),
        np.array(num_downsampling_neighbors),
        np.array(num_upsampling_neighbors),
        np.array(num_pwise_neighbors),
        fast_flag_per_depth,
        [  # Support args
            neighborhood_spec['type'].lower(),
            neighborhood_spec.get('K', 16),
            radii,
            neighborhood_spec['separation_factor'],
            support_strategy.lower(),
            support_strategy_num_points,
            support_strategy_fast,
            training_class_distribution,
            center_on_pcloud,
            True,  # Support extra nodes
        ],
        oversamplingArgs,
        nthreads
    )
    # Unpack pre-processing output
    xout, Fout, yout, Iout, Xout, NDout, NUout, Nout = preout
    Xout, Fout = [np.array(Xouti) for Xouti in Xout], np.array(Fout)
    if Fout.shape[-1] == 0:
        Fout = None
    if yout.shape[-1] == 0:
        yout = None
    if Iout[0].shape[1] == 1:
        for i in range(len(Iout)):
            Iout[i] = Iout[i].flatten()
    NDout = [np.array(NDouti) for NDouti in NDout]
    NUout[1:] = [np.array(NUouti) for NUouti in NUout[1:]]
    Nout = [np.array(Nouti) for Nouti in Nout]
    # Call C++ label reduction
    redout = vl3dpp.rf_reduce_label_mode_s32u32u32(
        Iout,
        [ND0outi for ND0outi in NDout[0]],
        y,
        num_classes,
        nthreads
    )
    # Simulate reduced probabilities
    z_reduced = np.random.normal(
        0,
        1,
        (support_strategy_num_points, num_points_per_depth[0], num_classes)
    )
    z_reduced = (
        (z_reduced.transpose(2, 0, 1) - np.min(z_reduced, axis=2)) /
        (np.max(z_reduced, axis=2) - np.min(z_reduced, axis=2))
    ).transpose(1, 2, 0).astype(np.float32)
    # Call C++ hierarchical fps post-processing
    postout = vl3dpp.rf_dl_fps_postproc_mean_fu32u32(
        "mean_reduce",
        X.shape[0],
        z_reduced,
        [NU0outi for NU0outi in NUout[0]],
        Iout,
        *[1e-6],  #*cpp_f_extra_args,
        nthreads
    )