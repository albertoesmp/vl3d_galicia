# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.inout.point_cloud_io import PointCloudIO
from src.utils.ptransf.receptive_field_hierarchical_sg import \
    ReceptiveFieldHierarchicalSG
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
from scipy.spatial import KDTree as KDT
import numpy as np
import os
import logging

# ---   CLASS   --- #
# ----------------- #
class VL3DPPHierarchicalSparseGridTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Deep learning pre-processor test that checks the C++ implementation of the
    hierarchical sparse grid (SG) pre-processors for deep learning models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ DL hierarchical sparse grid (HSG) test')
        self.eps = 0.5e-4  # Decimal tolerance threshold for the general case

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run C++ deep learning hierarchical sparse grid test.

        :return: True if the C++ hierarchical sparse grid works as expected for
            the test cases, False otherwise.
        :rtype: bool
        """
        vl3dpp.logging_disable()  # Disable C++ logging
        LOGGING.LOGGER.setLevel(logging.CRITICAL)  # Disable logger during test
        passed = True
        # Load data
        pcloud = PointCloudIO.read(os.path.join(
            'test_data', 'stanford-bunny-clustering-test.laz'
        ))
        X = pcloud.get_coordinates_matrix()
        pcloud0_ref = PointCloudIO.read(os.path.join(
            'test_data', 'sparse_cutted_bunny_centroids0.laz'
        ))
        centroids0_ref = pcloud0_ref.get_coordinates_matrix()
        pcloud1_ref = PointCloudIO.read(os.path.join(
            'test_data', 'sparse_cutted_bunny_centroids1.laz'
        ))
        centroids1_ref = pcloud1_ref.get_coordinates_matrix()
        pcloud2_ref = PointCloudIO.read(os.path.join(
            'test_data', 'sparse_cutted_bunny_centroids2.laz'
        ))
        centroids2_ref = pcloud2_ref.get_coordinates_matrix()
        # Build hierarchical sparse grid
        rfhsg = ReceptiveFieldHierarchicalSG(
            cell_size=0.002,
            submanifold_window=[1, 1, 1],
            downsampling_window=[2, 2],
            downsampling_stride=[2, 2],
            upsampling_window=[2, 2],
            upsampling_stride=[2, 2]
        ).fit(X)
        # Validate hierarchical sparse grid (depth 0)
        centroids = rfhsg.compute_active_centroids(0)
        kdt = KDT(centroids0_ref)
        d = kdt.query(centroids, k=1)[0]
        if np.any(d > self.eps):
            passed = False
        # Validate hierarchical sparse grid (depth 1)
        centroids = rfhsg.compute_active_centroids(1)
        kdt = KDT(centroids1_ref)
        d = kdt.query(centroids, k=1)[0]
        if np.any(d > self.eps):
            passed = False
        # Validate hierarchical sparse grid (depth 2)
        centroids = rfhsg.compute_active_centroids(2)
        kdt = KDT(centroids2_ref)
        d = kdt.query(centroids, k=1)[0]
        if np.any(d > self.eps):
            passed = False
        # Return status (True if all tests passed, False otherwise)
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        vl3dpp.logging_enable()  # Restore C++ logging
        return passed
