# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processor import \
    HierarchicalFPSPreProcessor
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processorpp import \
    HierarchicalFPSPreProcessorPP
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
import numpy as np
import scipy
from scipy.spatial import KDTree as KDT
import logging


# ---   CLASS   --- #
# ----------------- #
class VL3DPPDLHierarchicalFPSPreProcTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Deep learning pre-processor test that checks the C++ implementation of the
    hierarchical FPS pre-processors for deep learning models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ DL hierarchical FPS pre-processor test')
        self.eps = 0.5e-4  # For the general case
        self.unit_eps = 0.5e-2  # Only for unit sphere-transformed structure

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run C++ deep learning hierarchical FPS pre-processors test.

        :return: True if the C++ pre-processors work as expected for the test
            cases, False otherwise.
        :rtype: bool
        """
        vl3dpp.logging_disable()  # Disable C++ logging
        LOGGING.LOGGER.setLevel(logging.CRITICAL)  # Disable logger during test
        # Generate test data
        input_cases = 64
        fps_num_points = 64
        m = 16384  # Number of test points
        nx = 3  # Dimensionality of test structure space
        nf = 2  # Dimensionality of the test feature space
        nc = 5  # Dimensionality of test classes
        X = np.random.uniform(-0.005, 0.005, (m, nx))  # Test structure space
        F = np.random.normal(0, 1, (m, nf))  # Test feature space
        y = np.random.randint(0, nc, m)  # Test point-wise labels
        inputs_X = {'X': X}
        inputs_XF = {'X': [X, F]}
        inputs_Xy = {'X': X, 'y': y}
        inputs_XFy = {'X': [X, F], 'y': y}
        num_downsampling_neighbors = [1, 8, 8]
        num_pwise_neighbors = [16, 8, 8]
        num_upsampling_neighbors = [1, 8, 8]
        num_points_per_depth = [64, 32, 16]
        neighborhood_spec_sphere = {
            "type": "sphere",
            "radius": 0.001,
            "separation_factor": 1.0
        }
        neighborhood_spec_cylinder = {
            "type": "cylinder",
            "radius": 0.001,
            "separation_factor": 1.0
        }
        neighborhood_spec_rect3d = {
            "type": "rectangular3d",
            "radius": 0.001,
            "separation_factor": 1.0
        }
        neighborhood_spec_rect2d = {
            "type": "rectangular2d",
            "radius": 0.001,
            "separation_factor": 1.0
        }
        neighborhood_spec_bounded_cylinder = {
            "type": "bounded_cylinder",
            "radius": 0.001,
            "separation_factor": 1.0
        }
        neighborhood_spec_knn3d = {
            "type": "knn",
            "K": fps_num_points,
            "radius": 0.001,
            "separation_factor": 1.0
        }
        neighborhood_spec_knn2d = {
            "type": "knn2d",
            "K": fps_num_points,
            "radius": 0.001,
            "separation_factor": 1.0
        }
        neighborhood_spec_bounded_knn3d = {
            "type": "bounded_knn",
            "K": fps_num_points,
            "radius": 0.001,
            "separation_factor": 1.0
        }
        neighborhood_spec_bounded_knn2d = {
            "type": "bounded_knn2d",
            "K": fps_num_points,
            "radius": 0.001,
            "separation_factor": 1.0
        }
        knn_oversampling_spec = {
            'min_points': 8,
            'strategy': 'knn',
            'k': 8,
            'radius': 1.0,
            'nthreads': 1
        }
        passed = True
        # Hierarchical FPS pre-processor test (1)
        dlpre = HierarchicalFPSPreProcessor(
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
            nthreads=1
        )
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
            nthreads=1
        )
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (2)
        dlpre = HierarchicalFPSPreProcessor(
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
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (3)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_sphere,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=-1
        )
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
            support_strategy='fps',
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (4)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_sphere,
            receptive_field_oversampling=knn_oversampling_spec,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        dlprepp = HierarchicalFPSPreProcessorPP(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_sphere,
            receptive_field_oversampling=knn_oversampling_spec,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (5)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_sphere,
            receptive_field_oversampling=knn_oversampling_spec,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=-1
        )
        dlprepp = HierarchicalFPSPreProcessorPP(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_sphere,
            receptive_field_oversampling=knn_oversampling_spec,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (6)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_cylinder,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        dlprepp = HierarchicalFPSPreProcessorPP(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_cylinder,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (7)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_rect2d,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        dlprepp = HierarchicalFPSPreProcessorPP(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_rect2d,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (8)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_rect3d,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        dlprepp = HierarchicalFPSPreProcessorPP(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_rect3d,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (9)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_rect3d,
            receptive_field_oversampling=knn_oversampling_spec,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        dlprepp = HierarchicalFPSPreProcessorPP(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_rect3d,
            receptive_field_oversampling=knn_oversampling_spec,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (10)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[True, False, False],
            neighborhood=neighborhood_spec_sphere,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        dlprepp = HierarchicalFPSPreProcessorPP(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[True, False, False],
            neighborhood=neighborhood_spec_sphere,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_X)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XF)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_Xy)
        passed = passed and self.testDLPreProc(dlpre, dlprepp, inputs_XFy)
        # Hierarchical FPS pre-processor test (11)
        dlpre = HierarchicalFPSPreProcessor(
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
            to_unit_sphere=True,
            nthreads=-1
        )
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
            to_unit_sphere=True,
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(
            dlpre, dlprepp, inputs_X, eps=self.unit_eps
        )
        passed = passed and self.testDLPreProc(
            dlpre, dlprepp, inputs_XF, eps=self.unit_eps
        )
        passed = passed and self.testDLPreProc(
            dlpre, dlprepp, inputs_Xy, eps=self.unit_eps
        )
        passed = passed and self.testDLPreProc(
            dlpre, dlprepp, inputs_XFy, eps=self.unit_eps
        )
        # Hierarchical FPS pre-processor test (12)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_cylinder,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            to_unit_sphere=True,
            nthreads=-1
        )
        dlprepp = HierarchicalFPSPreProcessorPP(
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors,
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec_cylinder,
            receptive_field_oversampling=None,
            num_classes=nc,
            support_strategy_num_points=input_cases,
            support_strategy='grid',
            to_unit_sphere=True,
            nthreads=-1
        )
        passed = passed and self.testDLPreProc(
            dlpre, dlprepp, inputs_X, eps=self.unit_eps
        )
        passed = passed and self.testDLPreProc(
            dlpre, dlprepp, inputs_XF, eps=self.unit_eps
        )
        passed = passed and self.testDLPreProc(
            dlpre, dlprepp, inputs_Xy, eps=self.unit_eps
        )
        passed = passed and self.testDLPreProc(
            dlpre, dlprepp, inputs_XFy, eps=self.unit_eps
        )
        # Return status (True if all tests passed, False otherwise)
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        vl3dpp.logging_enable()  # Restore C++ logging
        return passed

    # ---   UTILS   --- #
    # ----------------- #
    def testDLPreProc(self, dlpre, dlprepp, inputs, eps=None):
        """
        Check that the C++ deep learning pre-processor works as the python
        reference pre-processor.

        :param dlpre: The object representing the Python-side pre-processor.
        :param dlprepp: The object representing the C++-side pre-processor.
        :param inputs: The inputs for the pre-processor.
        :type inputs: dict
        :param eps: The decimal tolerance threshold. If not given, it will be
            taken from self.eps. It must be given as self.unit_eps when doing
            tests on receptive fields transformed to the unit sphere.
        :type eps: float or None
        :return: True if the C++ pre-processor matches the results of the
            Python pre-processor, False otherwise.
        :rtype: bool
        """
        # Take eps from self if not given
        if eps is None:
            eps = self.eps
        # Determine if checks must be done in 2D
        check2D = (
            dlpre.neighborhood_spec['type'] == 'cylinder' or
            dlpre.neighborhood_spec['type'] == 'rectangular2d'
        )
        # Extract inputs for later checks
        Xin, Fin, yin = inputs['X'], None, inputs.get('y', None)
        if isinstance(Xin, list):
            Xin, Fin = Xin[0], Xin[1]
        # Python pre-processor
        outpre = dlpre(inputs)
        if isinstance(outpre, tuple):
            Xpy = outpre[0]
            if isinstance(Xpy, list):
                Fpy = Xpy[1]
                Xpy = Xpy[0]
            else:
                Fpy = None
            ypy = outpre[1]
        elif isinstance(outpre, list):
            Xpy, Fpy, ypy = outpre[0], outpre[1], None
        elif isinstance(outpre, np.ndarray):
            Xpy, Fpy, ypy = outpre, None, None
        else:
            Xpy, Fpy, ypy = outpre[0][0], outpre[0][1], outpre[1]
        # C++ pre-processor
        outprepp = dlprepp(inputs)  # C++ pre-processor
        if isinstance(outprepp, tuple):
            Xpp = outprepp[0]
            if isinstance(Xpp, list):
                Fpp = Xpp[1]
                Xpp = Xpp[0]
            else:
                Fpp = None
            ypp = outprepp[1]
        elif isinstance(outprepp, list):
            Xpp, Fpp, ypp = outprepp[0], outprepp[1], None
        elif isinstance(outprepp, np.ndarray):
            Xpp, Fpp, ypp = outprepp, None, None
        else:
            Xpp, Fpp, ypp = outprepp[0][0], outprepp[0][1], outprepp[1]
        # Validate output dimensionality
        if np.any(np.array(Xpy.shape) != np.array(Xpp.shape)):
            return False
        # Validate neighborhood centers
        num_rfs = Xpy.shape[0]
        assoc = []
        for i in range(num_rfs):
            pyrfi = dlpre.last_call_receptive_fields[i]
            xpy = pyrfi.x
            valid = False
            for k in range(num_rfs):
                pprfi = dlprepp.last_call_receptive_fields[k]
                xpp = pprfi.x
                if check2D:
                    mean_abs_diff = np.mean(np.abs(xpy[:2]-xpp[:2]))
                else:
                    mean_abs_diff = np.mean(np.abs(xpy-xpp))
                if mean_abs_diff <= self.eps:
                    valid = True
                    assoc.append([i, k])
                    break
            if not valid:
                return False
        # Validate structure space
        for associ in assoc:
            if check2D:
                kdt = KDT(Xpy[associ[0]][:, :2])
                d = kdt.query(Xpp[associ[1]][:, :2], 1)[0]
                if np.mean(d) > eps*10:
                    return False
                # Validate substructure spaces too
                pyrfi = dlpre.last_call_receptive_fields[associ[0]]
                pprfi = dlprepp.last_call_receptive_fields[associ[1]]
                for depth in range(pprfi.max_depth):
                    Ypyd, Yppd = pyrfi.Ys[depth], pprfi.Ys[depth]
                    kdt = KDT(Ypyd[:, :2])
                    d = kdt.query(Yppd[:, :2], 1)[0]
                    if np.mean(d) > eps * 10**(depth + 1):
                        return False
            else:
                # Validate the main structure space of the receptive field
                kdt = KDT(Xpy[associ[0]])
                d = kdt.query(Xpp[associ[1]], 1)[0]
                if np.mean(d) > eps*10:
                    return False
                # Validate substructure spaces too
                pyrfi = dlpre.last_call_receptive_fields[associ[0]]
                pprfi = dlprepp.last_call_receptive_fields[associ[1]]
                for depth in range(pprfi.max_depth):
                    Ypyd, Yppd = pyrfi.Ys[depth], pprfi.Ys[depth]
                    kdt = KDT(Ypyd)
                    d = kdt.query(Yppd, 1)[0]
                    if np.mean(d) > eps * 10**(depth+1):
                        return False
        # Validate feature space
        if Fpy is not None:  # Validate Python-side feature spaces
            for i in range(num_rfs):
                pyrfi = dlpre.last_call_receptive_fields[i]
                Ii = dlpre.last_call_neighborhoods[i]
                Fref = np.array([
                    np.mean(Fin[Ii][Ni], axis=0) for Ni in pyrfi.NDs[0]
                ])
                if not np.allclose(Fpy[i], Fref, rtol=0, atol=self.eps):
                    return False
        if Fpp is not None:  # Validate C++-side feature spaces
            for i in range(num_rfs):
                pprfi = dlprepp.last_call_receptive_fields[i]
                Ii = dlprepp.last_call_neighborhoods[i]
                Fref = np.array([
                    np.mean(Fin[Ii][Ni], axis=0) for Ni in pprfi.NDs[0]
                ])
                if not np.allclose(Fpp[i], Fref, rtol=0, atol=self.eps):
                    return False
        # Validate labels
        if ypy is not None:  # Validate Python-side labels
            for i in range(num_rfs):
                pyrfi = dlpre.last_call_receptive_fields[i]
                Ii = dlpre.last_call_neighborhoods[i]
                yref = np.array([
                    scipy.stats.mode(yin[Ii][Ni])[0] for Ni in pyrfi.NDs[0]
                ])
                if np.any(ypy[i] != yref):
                    return False
        if ypp is not None:  # Validate C++-side labels
            for i in range(num_rfs):
                pprfi = dlprepp.last_call_receptive_fields[i]
                Ii = dlprepp.last_call_neighborhoods[i]
                yref = np.array([
                    scipy.stats.mode(yin[Ii][Ni])[0] for Ni in pprfi.NDs[0]
                ])
                if np.any(ypp[i] != yref):
                    return False
        # All checks passed
        return True
