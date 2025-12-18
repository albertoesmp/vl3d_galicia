# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.inout.point_cloud_io import PointCloudIO
from src.model.deeplearn.dlrun.hierarchical_sg_pre_processorpp import \
    HierarchicalSGPreProcessorPP
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
from scipy.spatial import KDTree as KDT
import numpy as np
import pickle
import zipfile
import os
import logging

# ---   CLASS   --- #
# ----------------- #
class VL3DPPDLHierarchicalSGPreProcTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Deep learning pre-processor test that checks the C++ implementation of the
    hierarchical sparse grid (SG) pre-processors for deep learning models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ DL hierarchical SG pre-processor test')
        self.eps = 1e-5  # Decimal tolerance threshold for the general case

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run C++ deep learning hierarchical SG pre-processors test.

        :return: True if the C++ pre-processors work as expected for the test
            cases, False otherwise.
        :rtype: bool
        """
        vl3dpp.logging_disable()  # Disable C++ logging
        LOGGING.LOGGER.setLevel(logging.CRITICAL)  # Disable logger during test
        # Load test data
        pcloud = PointCloudIO.read(os.path.join(
            'test_data', 'stanford-bunny-clustering-test.laz'
        ))
        X = pcloud.get_coordinates_matrix()
        F = pcloud.get_features_matrix(['linear005', 'planar005'])
        y = pcloud.get_classes_vector()
        ny = len(np.unique(y))  # Number of classes
        # Load reference data for validation
        hsgpre_ref = None
        with zipfile.ZipFile(
            os.path.join('test_data', 'hsg_ref.zip'),
            'r'
        ) as ref_zip:
            hsgpre_ref = pickle.loads(ref_zip.read('hsg_ref.pickle'))
        passed = True
        # Validate HSG with labels
        hsgpre = HierarchicalSGPreProcessorPP(
            cell_size=0.002,
            submanifold_window=[1, 1, 1, 1, 1],
            downsampling_window=[2, 2, 2, 2],
            downsampling_stride=[2, 2, 2, 2],
            upsampling_window=[2, 2, 2, 2],
            upsampling_stride=[2, 2, 2, 2],
            num_classes=ny,
            support_strategy_num_points=128,
            support_strategy='grid',
            support_strategy_fast=False,
            center_on_pcloud=True,
            neighborhood={
                "type": "sphere",
                "radius": 0.05,
                "separation_factor": 0.8
            },
            nthreads=-1
        )
        inputs = {'X': [X, F], 'y': y}
        passed = passed and self.validateHSGPreProc(hsgpre, hsgpre_ref, inputs)
        # Validate HSG without labels
        inputs = {'X': [X, F]}
        passed = passed and self.validateHSGPreProc(hsgpre, hsgpre_ref, inputs)
        # Return status (True if all tests passed, False otherwise)
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        vl3dpp.logging_enable()  # Restore C++ logging
        return passed

    # ---   UTILS   --- #
    # ----------------- #
    def validateHSGPreProc(self, hsgpre, hsgpre_ref, inputs):
        """
        Check that the hierarchical sparse grid pre-processor works as expected.

        :param hsgpre: The object representing the Python-side pre-processor.
        :param hsgpre_ref: The object representing the reference values.
        :param inputs: The inputs for the pre-processor.
        :type inputs: dict
        :return: True if the hierarchical sparse grid pre-processor worked as
            expected, False otherwise.
        :rtype: bool
        """
        # Pre-process input
        if 'y' in inputs:
            pyout, yout = hsgpre(inputs)
        else:
            pyout = hsgpre(inputs)
            yout = None
        # Validate min points
        A = np.vstack([
            hsgi.get_min_point()
            for hsgi in hsgpre.last_call_receptive_fields
        ])
        Aref = np.array(hsgpre_ref['A'])
        if not np.allclose(A, Aref, atol=self.eps, rtol=0):
            return False
        # Validate num axis-wise partitions, features, and labels
        for i, hsgi in enumerate(hsgpre.last_call_receptive_fields):
            # Validate axis-wise partitions
            ni = hsgi.get_num_partitions()
            ni_ref = hsgpre_ref['n'][i]
            if not np.allclose(ni, ni_ref, atol=self.eps, rtol=0):
                return False
            # Validate features
            order = hsgi.get_submanifold_maps()[0][1] - 1
            argsort = np.argsort(order)
            Fi = pyout[0][i][1:][order][argsort]
            Fi_ref = hsgpre_ref['F'][i]
            if not np.allclose(
                    Fi,
                    Fi_ref,
                    atol=self.eps,
                    rtol=0
            ):
                return False
            # Validate labels
            if yout is not None and len(yout) > 0:
                yi = yout[i][order][argsort]
                yi_ref = hsgpre_ref['y'][i]
                if np.any(yi != yi_ref):
                    return False
        # Validate centroids
        for i, hsgi in enumerate(hsgpre.last_call_receptive_fields):
            max_depth = hsgi.get_max_depth()
            Xi_ref = hsgpre_ref['X'][i]
            for t in range(max_depth):
                Xit = hsgi.compute_active_centroids(t)
                Xit_ref = Xi_ref[t]
                kdt = KDT(Xit_ref)
                d = kdt.query(Xit, k=1)[0]
                if np.any(d > self.eps):
                    return False
        # Validate submanifold maps
        for i, hsgi in enumerate(hsgpre.last_call_receptive_fields):
            max_depth = hsgi.get_max_depth()
            hi = hsgi.get_submanifold_maps()
            hi_ref = hsgpre_ref['h'][i]
            for t in range(max_depth):
                hit_k, hit_v = hi[t]
                hit_ref_kv = hi_ref[t]
                for p, k in enumerate(hit_k):
                    hit_ref_v = hit_ref_kv[k]
                    if hit_v[p] != hit_ref_v:
                        return False
        # Validate downsampling and upsampling maps
        for i, hsgi in enumerate(hsgpre.last_call_receptive_fields):
            max_depth = hsgi.get_max_depth()
            hDi = hsgi.get_downsampling_vectors()
            hUi = hsgi.get_upsampling_vectors()
            hDi_ref = hsgpre_ref['hD'][i]
            hUi_ref = hsgpre_ref['hU'][i]
            for t in range(max_depth-1):
                hDit = hDi[t]
                hUit = hUi[t]
                hDit_ref = hDi_ref[t]
                hUit_ref = hUi_ref[t]
                if np.any(hDit != hDit_ref):
                    return False
                if np.any(hUit != hUit_ref):
                    return False
        # All checks passed
        return True