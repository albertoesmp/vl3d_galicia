# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.utils.neighborhood.support_neighborhoods import SupportNeighborhoods
from src.utils.neighborhood.support_neighborhoodspp import \
    SupportNeighborhoodsPP
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
import numpy as np
from scipy.spatial import KDTree as KDT
import os
import logging


# ---   CLASS   --- #
# ----------------- #
class VL3DPPSupportNeighborhoodsTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Deep learning pre-processor tests that checks the C++ implementation of the
    support neighborhood-generation algorithms.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ support neighborhoods test')
        self.eps = 1e-5

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run C++ support neighborhoods test.

        :return: True if the C++ support neighborhoods are generated as
            expected for the test cases, False otherwise.
        :rtype: bool
        """
        vl3dpp.logging_disable()  # Disable C++ logging
        LOGGING.LOGGER.setLevel(logging.CRITICAL)  # Disable logger during test
        # Load test data
        X = np.loadtxt(os.path.join(
            os.getcwd(), 'cpp', 'test_data', 'support_test_X.xyz'
        ))
        y = np.loadtxt(os.path.join(
            os.getcwd(), 'cpp', 'test_data', 'support_test_y.xyz'
        ))
        m = X.shape[0]  # Number of test 3D points
        nx = X.shape[1]  # Dimensionality of the structure space (3D)
        nc = len(np.unique(y))  # Number of classes
        # Neighborhood specifications
        neigh_spec_spherical = {
            'type': 'sphere',
            'radius': 1.5,
            'separation_factor': 0.7
        }
        neigh_spec_cylindrical = {
            'type': 'cylinder',
            'radius': 1.0,
            'separation_factor': 0.7
        }
        neigh_spec_rect2d = {
            'type': 'rectangular2d',
            'radius': [1.0, 1.0],
            'separation_factor': 0.7
        }
        neigh_spec_rect3d = {
            'type': 'rectangular3d',
            'radius': [1.5, 1.5, 1.5],
            'separation_factor': 0.7
        }
        # Support specifications
        sup_spec_distr = {
            'support_strategy': 'training_class_distribution',
            'support_strategy_num_points': 0,
            'support_strategy_fast': False,
            'training_class_distribution': np.random.randint(8, 16, nc).tolist(),
            'center_on_pcloud': False,
            'nthreads': -1
        }
        sup_spec_grid1 = {
            'support_strategy': 'grid',
            'support_strategy_num_points': 0,
            'support_strategy_fast': False,
            'training_class_distribution': None,
            'center_on_pcloud': False,
            'nthreads': -1
        }
        sup_spec_grid2 = {
            'support_strategy': 'grid',
            'support_strategy_num_points': 0,
            'support_strategy_fast': False,
            'training_class_distribution': None,
            'center_on_pcloud': True,
            'nthreads': -1
        }
        sup_spec_fps1 = {
            'support_strategy': 'fps',
            'support_strategy_num_points': 64,
            'support_strategy_fast': False,
            'training_class_distribution': None,
            'center_on_pcloud': True,
            'nthreads': -1
        }
        sup_spec_fps2 = {
            'support_strategy': 'fps',
            'support_strategy_num_points': 256,
            'support_strategy_fast': True,
            'training_class_distribution': None,
            'center_on_pcloud': True,
            'nthreads': -1
        }
        # Test grid support neighborhoods
        passed = self.testSupport(neigh_spec_spherical, sup_spec_grid1, X, y)
        passed = passed and self.testSupport(
            neigh_spec_spherical, sup_spec_grid2, X, y
        )
        passed = passed and self.testSupport(
            neigh_spec_cylindrical, sup_spec_grid1, X, y
        )
        passed = passed and self.testSupport(
            neigh_spec_cylindrical, sup_spec_grid2, X, y
        )
        passed = passed and self.testSupport(
            neigh_spec_cylindrical, sup_spec_grid2, X, y
        )
        passed = passed and self.testSupport(
            neigh_spec_rect3d, sup_spec_grid1, X, y
        )
        passed = passed and self.testSupport(
            neigh_spec_rect3d, sup_spec_grid2, X, y
        )
        passed = passed and self.testSupport(
            neigh_spec_rect2d, sup_spec_grid1, X, y
        )
        passed = passed and self.testSupport(
            neigh_spec_rect2d, sup_spec_grid2, X, y
        )
        # Test furthest point subsampling neighborhoods
        passed = passed and self.testSupportByFPS(
            neigh_spec_spherical, sup_spec_fps1, X, y
        )
        passed = passed and self.testSupportByFPS(
            neigh_spec_spherical, sup_spec_fps2, X, y
        )
        passed = passed and self.testSupportByFPS(
            neigh_spec_cylindrical, sup_spec_fps1, X, y
        )
        passed = passed and self.testSupportByFPS(
            neigh_spec_cylindrical, sup_spec_fps2, X, y
        )
        passed = passed and self.testSupportByFPS(
            neigh_spec_rect3d, sup_spec_fps1, X, y
        )
        passed = passed and self.testSupportByFPS(
            neigh_spec_rect3d, sup_spec_fps2, X, y
        )
        passed = passed and self.testSupportByFPS(
            neigh_spec_rect2d, sup_spec_fps1, X, y
        )
        passed = passed and self.testSupportByFPS(
            neigh_spec_rect2d, sup_spec_fps2, X, y
        )
        # Test training class distribution-based support neighborhoods
        passed = passed and self.testSupportByDistr(
            neigh_spec_spherical, sup_spec_distr, X, y
        )
        passed = passed and self.testSupportByDistr(
            neigh_spec_cylindrical, sup_spec_distr, X, y
        )
        passed = passed and self.testSupportByDistr(
            neigh_spec_rect3d, sup_spec_distr, X, y
        )
        passed = passed and self.testSupportByDistr(
            neigh_spec_rect2d, sup_spec_distr, X, y
        )
        # Return status
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        vl3dpp.logging_enable()  # Restore C++ logging
        return passed

    # ---   UTILS   --- #
    # ----------------- #
    def testSupport(self, neigh_spec, sup_spec, X, y):
        """
        Check that the C++ support neighborhoods work as the python reference
        support neighborhoods in the general case.

        :param neigh_spec: The neighborhood specification.
        :type neigh_spec: dict
        :param sup_spec: The support specification.
        :type sup_spec: dict
        :return: True if the C++ support neighborhoods match the Python-side
            support neighborhoods.
        :rtype: bool
        """
        # Determine if checks must be done in 2D
        check2D = (
            neigh_spec['type'] == 'cylinder' or
            neigh_spec['type'] == 'rectangular2d'
        )
        # Compute support neighborhoods
        sup_Xpy, Ipy = SupportNeighborhoods(neigh_spec, **sup_spec).compute(
            X, y
        )
        sup_Xpp, Ipp = SupportNeighborhoodsPP(neigh_spec, **sup_spec).compute(
            X, y
        )
        # Validate structure spaces
        if any([
            sup_Xpy.shape[i] != sup_Xpp.shape[i]
            for i in range(len(sup_Xpy.shape))
        ]):
            return False
        if check2D:
            kdt = KDT(sup_Xpy[:, :2])
            d = kdt.query(sup_Xpp[:, :2], k=1)[0]
        else:
            kdt = KDT(sup_Xpy)
            d = kdt.query(sup_Xpp, k=1)[0]
        if np.any(d > self.eps):
            return False
        # Validate neighborhoods
        py_sum = sum([len(Ipyi) for Ipyi in Ipy])
        pp_sum = sum([len(Ippi) for Ippi in Ipp])
        if abs(1-pp_sum/py_sum) > self.eps: # Small diff. due to C++ and Python
            return False                    # KDTrees with diff. boundary policy
        # On all checks passed
        return True

    def testSupportByDistr(self, neigh_spec, sup_spec, X, y):
        """
        Check that the C++ support neighborhoods work as the python reference
        support neighborhoods for the training class distribution strategy.

        :param neigh_spec: The neighborhood specification.
        :type neigh_spec: dict
        :param sup_spec: The support specification.
        :type sup_spec: dict
        :return: True if the C++ support neighborhoods match the Python-side
            support neighborhoods.
        :rtype: bool
        """
        # Determine if checks must be done in 2D
        check2D = (
            neigh_spec['type'] == 'cylinder' or
            neigh_spec['type'] == 'rectangular2d'
        )
        # Compute support neighborhoods
        sup_Xpy, Ipy = SupportNeighborhoods(neigh_spec, **sup_spec).compute(
            X, y
        )
        sup_Xpp, Ipp = SupportNeighborhoodsPP(neigh_spec, **sup_spec).compute(
            X, y
        )
        # Get classes by indices
        kdt = KDT(X[:, :2]) if check2D else KDT(X)
        I = kdt.query(sup_Xpy[:, :2] if check2D else sup_Xpy, 1)[1]
        ypy = y[I]
        I = kdt.query(sup_Xpp[:, :2] if check2D else sup_Xpp, 1)[1]
        ypp = y[I]
        # Count samples by class
        count_py = np.array([
            np.count_nonzero(ypy == c)
            for c in range(len(sup_spec['training_class_distribution']))
        ])
        count_pp = np.array([
            np.count_nonzero(ypp == c)
            for c in range(len(sup_spec['training_class_distribution']))
        ])
        # Check both counts are as expected
        ref = np.array(sup_spec['training_class_distribution'])
        if np.any(count_py != ref):
            return False
        if np.any(count_pp != ref):
            return False
        # On all checks passed
        return True

    def testSupportByFPS(self, neigh_spec, sup_spec, X, y):
        """
        Check that the C++ support neighborhoods work as the python reference
        support neighborhoods for the furthest point subsampling (FPS)
        strategy.

        :param neigh_spec: The neighborhood specification.
        :type neigh_spec: dict
        :param sup_spec: The support specification.
        :type sup_spec: dict
        :return: True if the C++ support neighborhoods match the Python-side
            support neighborhoods.
        :rtype: bool
        """
        # Determine if checks must be done in 2D
        check2D = (
            neigh_spec['type'] == 'cylinder' or
            neigh_spec['type'] == 'rectangular2d'
        )
        # Compute support neighborhoods
        sup_Xpy, Ipy = SupportNeighborhoods(neigh_spec, **sup_spec).compute(
            X, y
        )
        sup_Xpp, Ipp = SupportNeighborhoodsPP(neigh_spec, **sup_spec).compute(
            X, y
        )
        # Check number of points
        expected_num_points = sup_spec['support_strategy_num_points']
        if len(np.unique(sup_Xpy, axis=0)) != expected_num_points:
            return False
        if len(np.unique(sup_Xpp, axis=0)) != expected_num_points:
            return False
        # Validate structure spaces
        if any([
            sup_Xpy.shape[i] != sup_Xpp.shape[i]
            for i in range(len(sup_Xpy.shape))
        ]):
            return False
        kdt = KDT(sup_Xpy[:, :2]) if check2D else KDT(sup_Xpy)
        d = kdt.query(
            sup_Xpp[:, :2] if check2D else sup_Xpp,
        k=1)[0]
        if np.any(d > self.eps):
            return False
        # On all checks passed
        return True
