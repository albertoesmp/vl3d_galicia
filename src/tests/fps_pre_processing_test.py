# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.dlrun.furthest_point_subsampling_pre_processor import \
    FurthestPointSubsamplingPreProcessor
import src.main.main_logger as LOGGING
import scipy
from scipy.spatial import KDTree as KDT
import numpy as np
import logging


# ---   CLASS   --- #
# ----------------- #
class FPSPreProcessingTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    FPS pre-processing test that checks the FPS-based preprocessing of a point
    cloud is correct, especially regarding feature and label encoding.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__("FPS pre-processing test")
        self.eps = 1e-5

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run FPS pre-processing test.

        :return: True if the FPS pre-processor works as expected for the test
            cases, False otherwise.
        :rtype bool:
        """
        LOGGING.LOGGER.setLevel(logging.CRITICAL)  # Disable logger during test
        # Generate test data
        m = 5000  # Number of points
        nf = 2  # Number of features
        nc = 4  # Number of classes
        X = np.random.uniform(-0.9, 0.9, (m, 3))
        F = np.random.normal(0, 1, (m, nf))
        y = np.random.randint(0, nc, m)
        # Build FPS pre-processor
        neighborhood = {
            "type": "sphere",
            "radius": 0.33,
            "separation_factor": 0.6
        }
        fps_pre_proc = FurthestPointSubsamplingPreProcessor(
            num_points=100,
            num_encoding_neighbors=7,
            fast=False,
            neighborhood=neighborhood,
            receptive_field_oversampling=None,
            support_strategy_num_points=32,
            support_strategy='fps',
            num_classes=nc,
            nthreads=-1
        )
        # Validate FPS pre-processor
        if not self.testFPSPreProc(X, F, y, neighborhood, fps_pre_proc):
            return False
        # Build FPS pre-processor working in the unit sphere
        neighborhood = {
            "type": "sphere",
            "radius": 0.33,
            "separation_factor": 0.6
        }
        fps_pre_proc = FurthestPointSubsamplingPreProcessor(
            num_points=100,
            num_encoding_neighbors=7,
            fast=False,
            neighborhood=neighborhood,
            receptive_field_oversampling=None,
            support_strategy_num_points=32,
            support_strategy='fps',
            to_unit_sphere=True,
            num_classes=nc,
            nthreads=-1
        )
        # Validate FPS pre-processor
        if not self.testFPSPreProc(
            X, F, y, neighborhood, fps_pre_proc, unit_sphere=True
        ):
            return False
        # On success
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        return True

    # ---   UTILS   --- #
    # ----------------- #
    def testFPSPreProc(
        self, X, F, y, neighborhood, fps_pre_proc, unit_sphere=False
    ):
        """
        Check whether the given FPS pre-processor is correct.

        :param X: The test structure space.
        :type X: :class:`np.ndarray`
        :type F: The test feature space.
        :type F: :class:`np.ndarray`
        :param y: The test labels.
        :type y: :class:`np.ndarray`
        :param neighborhood: The neighborhood specification.
        :type neighborhood: dict
        :param fps_pre_proc: The FPS pre-processor specification.
        :type fps_pre_proc: dict
        :param unit_sphere: Flag governing whether the structure space of the
            receptive fields must be checked to fit a unit sphere (True) or
            not (False).
        :type unit_sphere: bool
        """
        # Call FPS pre-processor
        out = fps_pre_proc({'X': [X, F], 'y': y})
        Xout, Fout, yout = out[0][0], out[0][1], out[1]
        # Validate FPS pre-processor
        kdt = KDT(X)
        for i, rfi in enumerate(fps_pre_proc.last_call_receptive_fields):
            Xouti, Fouti, youti = Xout[i], Fout[i], yout[i]
            # Ii = fps_pre_proc.last_call_neighborhoods[i]  # Legacy
            # Improved alternative ---
            Ii = np.array(kdt.query_ball_point(
                rfi.x,
                neighborhood['radius']
            ))
            Ii = Ii[np.argsort(Ii)]
            # --- Improved alternative
            Fref = np.array([np.mean(F[Ii][Ni], axis=0) for Ni in rfi.N])
            if not np.allclose(Fouti, Fref, rtol=0, atol=self.eps):
                return False
            yref = np.array([scipy.stats.mode(y[Ii][Ni])[0] for Ni in rfi.N])
            if np.any(youti != yref):
                return False
            # Check that the rfield structure is scaled to the unit sphere
            if unit_sphere:
                if abs(
                    np.sqrt(np.max(np.sum(np.square(Xouti), axis=1)))-1
                ) > self.eps:
                    return False
        # On success
        return True
