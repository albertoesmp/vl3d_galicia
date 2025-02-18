# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.dlrun.hierarchical_sg_pre_processorpp import \
    HierarchicalSGPreProcessorPP
from src.model.deeplearn.sequencer.dl_sparse_shadow_sequencer import \
    DLSparseShadowSequencer
from src.inout.point_cloud_io import PointCloudIO
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
import numpy as np
import logging
import os

class DLSparseShadowSequencerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Test to check that the :class:`.DLSparseShadowSequencer` works as expected.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('DL sparse shadow sequencer test')
        self.eps = 1e-5 # Decimal tolerance threshold (ref. decimals use %.5f)

    # ---  TEST INTERFACE   --- #
    # ------------------------- #
    def run(self):
        """
        Run DL sparse shadow sequencer test.

        :return: True if the DL sparse shadow sequencer works as expected for
            the test cases, False otherwise.
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
        num_batches = 32
        X_ref = [[] for k in range(17)]
        y_ref = []
        for k in range(num_batches):
            for bkl in range(len(X_ref)):
                X_bkl = X_ref[bkl]
                X_bkl.append(np.loadtxt(
                    os.path.join(
                        'test_data', 'shadow_batches', f'X_{bkl}_{k}.xyz'
                    )
                ))
            y_ref.append(np.loadtxt(
                os.path.join(
                    'test_data', 'shadow_batches', f'y_{k}.xyz'
                )
            ))
        # Build hierarchical sparse grid pre-processor
        hsgpre = HierarchicalSGPreProcessorPP(**{
            "support_strategy_num_points": num_batches,
            "support_strategy": "fps",
            "support_strategy_fast": 4,
            "center_on_pcloud": True,
            "training_class_distribution": None,
            "neighborhood": {
                "type": "sphere",
                "radius": 0.05,
                "separation_factor": 0.8
            },
            "cell_size": 0.005,
            "submanifold_window": [1, 1, 1],
            "downsampling_window": [2, 2],
            "downsampling_stride": [2, 2],
            "upsampling_window": [2, 2],
            "upsampling_stride": [2, 2],
            "nthreads": -1,
            "training_receptive_fields_distribution_report_path": None,
            "training_receptive_fields_distribution_plot_path": None,
            "training_receptive_fields_dir": None,
            "receptive_fields_distribution_report_path": None,
            "receptive_fields_distribution_plot_path": None,
            "receptive_fields_dir": None,
            "training_support_points_report_path": None
        })
        X_rf, y_rf = hsgpre({
            'X': [X, F], 'y':y, 'training_support_points': True
        })
        # Build DL sparse sequencer
        batch_size = 3
        dlsss = DLSparseShadowSequencer(
            X_rf, y_rf, batch_size, **{
                'random_shuffle_indices': False,
                'training': True
            }
        )
        # TODO Remove : Debug ---
        dlsss.init_random_indices()
        dlsss.apply_random_indices()
        # --- TODO Remove : Debug
        k = 0
        for batch_X, batch_y in dlsss:
            for bk in range(batch_X[0].shape[0]):
                for bkl in range(len(batch_X)):
                    if not np.allclose(
                        X_ref[bkl][k], batch_X[bkl][bk], rtol=0, atol=self.eps
                    ):
                        return False
                    if not np.allclose(
                        y_ref[k], batch_y[bk], rtol=0, atol=self.eps
                    ):
                        return False
                k += 1
        # Return True on success
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        vl3dpp.logging_enable()  # Restore C++ logging
        return True
