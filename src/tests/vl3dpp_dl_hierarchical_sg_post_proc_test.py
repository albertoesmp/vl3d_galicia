# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.inout.point_cloud_io import PointCloudIO
from src.model.deeplearn.dlrun.hierarchical_sg_pre_processorpp import \
    HierarchicalSGPreProcessorPP
from src.model.deeplearn.dlrun.hierarchical_sg_post_processorpp import \
    HierarchicalSGPostProcessorPP
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
import numpy as np
import os
import logging


# ---   CLASS   --- #
# ----------------- #
class VL3DPPDLHierarchicalSGPostProcTest(VL3DTest):
    """
    :author: Alberto M. Esmrois Pena

    Deep learning post-processor test that checks the C++ implementation of the
    hierarchical sparse-grid post-processing logic for deep learning models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ deep learning HSG post-processor test')
        self.eps = 1e-5

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run C++ deep learning hierarchical post-processors test.

        :return: True if the C++ hierarchical sparse grid post-processor works
            as expected, False otherwise.
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
        y = pcloud.get_classes_vector() - 1  # -1 to start at 0 instead of 1
        ny = len(np.unique(y))  # Number of classes
        passed = True
        # Test HSG post-processor (1)
        inputspre = {'X': [X, F], 'y': y}
        hsgpre = HierarchicalSGPreProcessorPP(
            cell_size = 0.002,
            submanifold_window = [1, 1, 1, 1, 1],
            downsampling_window = [2, 2, 2, 2],
            downsampling_stride = [2, 2, 2, 2],
            upsampling_window = [2, 2, 2, 2],
            upsampling_stride = [2, 2, 2, 2],
            num_classes = ny,
            support_strategy_num_points = 128,
            support_strategy = 'grid',
            support_strategy_fast = False,
            center_on_pcloud = True,
            neighborhood = {
                "type": "sphere",
                "radius": 0.05,
                "separation_factor": 0.8
            },
            nthreads = -1
        )
        passed = passed and VL3DPPDLHierarchicalSGPostProcTest\
            .testSparseDLPostProc(
                hsgpre, inputspre, X, y, ny
            )
        # Return status (True if all tests passed, False otherwise)
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        vl3dpp.logging_enable()  # Restore C++ logging
        return passed

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def z_from_y(y, ny):
        """
        Generate probabilities from labels.

        :param y: The labels from which the probabilities must be derived.
        :param ny: The number of different classes.
        :return: The matrix of probabilities for each receptive field.
        :rtype: list of :class:`np.ndarray`
        """
        z = []
        for yi in y:
            zi = [
                np.zeros((yi.shape[0], 1), dtype=np.float32)
                for c in range(ny)
            ]
            for c in range(ny):
                mask = yi == c
                zi[c][mask] = 1
            z.append(np.hstack(zi))
        return z

    @staticmethod
    def testSparseDLPostProc(hsgpre, inputspre, X, y, ny):
        """
        Check whether the :class:`.HierarchicalSGPostProcessorPP` yields the
        expected output or not.

        :param hsgpre: The :class:`.HierarchicalSGPreProcessorPP` associated
            to the :class:`.HierarchicalSGPostProcessorPP`.
        :param inputspre: The input dictionary for the pre-processor.
        :param X: The structure space representing the original input point
            cloud.
        :param y: The point-wise labels for the original input point cloud.
        :param ny: The number of classes.
        :return: True if the test is passed, False otherwise.
        :rtype: bool
        """
        # Prepare pre-processing
        pyout, yout = hsgpre(inputspre)
        zout = VL3DPPDLHierarchicalSGPostProcTest.z_from_y(yout, ny)
        # Check post-processing with straight forward X
        inputspost = {'X': X, 'z': zout}
        hsgpost = HierarchicalSGPostProcessorPP(hsgpre)
        zhat = hsgpost(inputspost)
        yhat = np.argmax(zhat, axis=1)
        if np.any(yhat != y):
            return False
        # Check post-processing with X in a single element list
        inputspost = {'X': [X], 'z': zout}
        hsgpost = HierarchicalSGPostProcessorPP(hsgpre)
        zhat = hsgpost(inputspost)
        yhat = np.argmax(zhat, axis=1)
        if np.any(yhat != y):
            return False
        # Check post-processing with X and F in the same list
        inputspost = {'X': [X, np.ones((X.shape[0], 1))], 'z': zout}
        hsgpost = HierarchicalSGPostProcessorPP(hsgpre)
        zhat = hsgpost(inputspost)
        yhat = np.argmax(zhat, axis=1)
        if np.any(yhat != y):
            return False
        # Return True if all checks were passed
        return True
