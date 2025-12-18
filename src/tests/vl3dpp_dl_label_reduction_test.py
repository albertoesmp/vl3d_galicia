# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.dlrun.furthest_point_subsampling_pre_processor \
    import FurthestPointSubsamplingPreProcessor
from src.model.deeplearn.dlrun.furthest_point_subsampling_pre_processorpp \
    import FurthestPointSubsamplingPreProcessorPP
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processor \
    import HierarchicalFPSPreProcessor
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processorpp \
    import HierarchicalFPSPreProcessorPP
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
import numpy as np
import logging


# ---   CLASS   --- #
# ----------------- #
class VL3DPPDLLabelReductionTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Deep learning label reduction test that checks the C++ implementation for
    label reduction for deep learning models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ DL label reduction tests')

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run C++ deep learning label reduction test.

        :return: True if the C++ label reduction works as expected for the test
            cases, False otherwise.
        :rtype: bool
        """
        vl3dpp.logging_disable()  # Disable C++ logging
        LOGGING.LOGGER.setLevel(logging.CRITICAL)  # Disable logger during test
        # Generate test data
        input_cases = 64
        fps_num_points = 64
        num_downsampling_neighbors = [1, 8, 8]
        num_pwise_neighbors = [16, 8, 8]
        num_upsampling_neighbors = [1, 8, 8]
        num_points_per_depth = [64, 32, 16]
        m = 16384  # Number of test points
        nx = 3  # Dimensionality of test structure space
        nc = 5  # Dimensionality of test classes
        X = np.random.uniform(-0.005, 0.005, (m, nx))  # Test structure space
        y = np.random.randint(0, nc, m)  # Test point-wise labels
        inputs_Xy = {'X': X, 'y': y}
        neighborhood_spec_sphere = {
            "type": "sphere",
            "radius": 0.001,
            "separation_factor": 1.0
        }
        passed = True
        # Test FPS pre-processor (1)
        dlprepp = FurthestPointSubsamplingPreProcessorPP(
            num_classes=nc,
            num_points=fps_num_points,
            num_encoding_neighbors=8,
            fast=False,
            neighborhood=neighborhood_spec_sphere,
            receptive_field_oversampling=None,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        passed = passed and self.testLabelReduction(dlprepp, inputs_Xy)
        # Test Hierarchical FPS pre-processor (2)
        dlprepp = HierarchicalFPSPreProcessorPP(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_sphere,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        passed = passed and self.testLabelReduction(dlprepp, inputs_Xy)
        # On all checks passed
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        vl3dpp.logging_enable()  # Restore C++ logging
        return passed

    # ---   UTILS   --- #
    # ----------------- #
    def testLabelReduction(self, dlprepp, inputs):
        """
        Checks that the C++ deep learning label reduction works as the python
        reference label reduction.

        :param dlprepp: The object representing the C++ receptive field
            pre-processor.
        :param inputs: The inputs for the label reduction.
        :type inputs: dict
        """
        # Extract input
        X = inputs['X']
        y = inputs['y']
        # Fit receptive field
        dlprepp(inputs)
        # Reduce labels
        yredpy = dlprepp.reduce_labels_python(X, y)
        yredpp = dlprepp.reduce_labels(X, y)
        # Validate
        if np.any(yredpy != yredpp):
            return False
        # On success
        return True
