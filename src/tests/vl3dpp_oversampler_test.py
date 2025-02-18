# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
import pyvl3dpp as vl3dpp
import numpy as np
from scipy.spatial import KDTree as KDT


# ---   CLASS   --- #
# ----------------- #
class VL3DPPOversamplerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Oversampling test that checks the C++ oversampling implementations.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ oversampler test')
        self.eps = 1e-12  # Finite precission decimal representation of the zero
        self.max_discrepancy_ratio = 0.1

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run C++ oversampling tests.

        :return: True if the C++ oversampling works as expected for the test
            cases, False otherwise.
        :rtype: bool
        """
        # Generate test data
        m = 50  # Number of points
        nx = 3  # Structure space dimensionality
        target_points = 401  # Target number of points
        X = np.random.uniform(-0.5, 0.5, (m, nx))  # Test structure space
        passed = True
        # Test nearest oversampling
        nearest_spec = {
            'min_points': 8,
            'strategy': 'nearest',
            'k': 16,
            'radius': 1.0,
            'nthreads': 1
        }
        passed = passed and self.testOversample(
            X, target_points, nearest_spec, self.max_discrepancy_ratio
        )
        # Test KNN oversampling
        knn_spec = {
            'min_points': 8,
            'strategy': 'knn',
            'k': 16,
            'radius': 1.0,
            'nthreads': 1
        }
        passed = passed and self.testOversample(
            X, target_points, knn_spec, 0
        )
        # Test Gaussian KNN oversampling
        gaussknn_spec = {
            'min_points': 8,
            'strategy': 'gaussian_knn',
            'k': 16,
            'radius': 1.0,
            'nthreads': 1
        }
        passed = passed and self.testOversample(
            X, target_points, gaussknn_spec, 0
        )
        # Test spherical oversampling
        spherical_spec = {
            'min_points': 8,
            'strategy': 'spherical',
            'k': 16,
            'radius': 0.5,
            'nthreads': 1
        }
        passed = passed and self.testOversample(
            X, target_points, spherical_spec, 0
        )
        # Test spherical radiation oversampling
        radiation_spec = {
            'min_points': 8,
            'strategy': 'spherical_radiation',
            'k': 16,
            'radius': 0.5,
            'nthreads': 1
        }
        passed = passed and self.testOversample(
            X, target_points, radiation_spec, 0
        )
        # On success
        return passed

    # ---   UTILS   --- #
    # ----------------- #
    def testOversample(
        self, X, target_points, nearest_spec, max_discrepancy_ratio=0.0
    ):
        """
        Check that the C++ oversampling works as the Python reference
        oversampling.

        :param X: The test structure space that must be oversampled.
        :type X: :class:`np.ndarray`
        :param target_points: How many points must be obtained through
            oversampling (provided it is needed).
        :type target_points: int
        :param nearest_spec: The key-word arguments governing the oversampling.
        :type nearest_spec: dict
        :param max_discrepancy_ratio: The ratio in :math:`[0, 1]` of points that
            can be different between the Python and C++ implementations.
        :type max_discrepancy_ratio: float
        :return: True if the C++ oversampling works as expected, False
            otherwise.
        :rtype: bool
        """
        # Python-side oversampling
        _nearest_spec = dict(nearest_spec)
        if _nearest_spec['strategy'] == 'spherical':  # Use naive spherical
            _nearest_spec['strategy'] = 'spherical_naive'  # in Python
        if _nearest_spec['strategy'] == 'spherical_radiation':  # Same as above
            _nearest_spec['strategy'] = 'spherical_radiation_naive'
        Ypy = ReceptiveFieldFPS.oversample(X, target_points, **_nearest_spec)
        # C++ oversampling
        Ypp = vl3dpp.alg_oversampler_du32(
            nearest_spec['min_points'],
            target_points,
            nearest_spec['strategy'],
            nearest_spec['k'],
            nearest_spec['radius'],
            X
        )
        # Check there are oversampled points
        if len(np.unique(Ypy, axis=0)) <= X.shape[0]:
            return False
        if len(np.unique(Ypp, axis=0)) <= X.shape[0]:
            return False
        # Check structure space
        kdt = KDT(Ypy)
        d = kdt.query(Ypp, 1)[0]
        if np.count_nonzero(d > self.eps) > len(d)*max_discrepancy_ratio:
            return False
        # On success
        return True
