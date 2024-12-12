# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.dlrun.furthest_point_subsampling_post_processor import\
    FurthestPointSubsamplingPostProcessor
from src.model.deeplearn.dlrun.furthest_point_subsampling_pre_processor import\
    FurthestPointSubsamplingPreProcessor
from src.model.deeplearn.dlrun.furthest_point_subsampling_post_processorpp\
    import FurthestPointSubsamplingPostProcessorPP
from src.model.deeplearn.dlrun.hierarchical_fps_post_processor import \
    HierarchicalFPSPostProcessor
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processor import \
    HierarchicalFPSPreProcessor
from src.model.deeplearn.dlrun.hierarchical_fps_post_processorpp import \
    HierarchicalFPSPostProcessorPP
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processorpp import \
    HierarchicalFPSPreProcessorPP
from src.utils.preds.prediction_reducer_factory import PredictionReducerFactory
import src.main.main_logger as LOGGING
import numpy as np
import logging


# ---   CLASS   --- #
# ----------------- #
class VL3DPPDLPostProcTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Deep learning post processor test that checks the C++ implementation of the
    post processors for deep learning models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Vl3D++ deep learning post-processors test')
        self.eps = 1e-5

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run C++ deep learning post-processors test.

        :return: True if the C++ post-processors work as expected for the test
            cases, False otherwise.
        :rtype: bool
        """
        LOGGING.LOGGER.setLevel(logging.CRITICAL)  # Disable logger during test
        # Generate test data
        input_cases = 64
        fps_num_points = 16
        m = 4096  # Number of test points
        nx = 3  # Dimensionality of test structure space
        nc = 5  # Dimensionality of test classes
        X = np.random.normal(0, 1, (m, nx))  # Test structure space
        Z = np.random.normal(0, 1, (input_cases, fps_num_points, nc))  # Probs.
        Z = (
            (Z.transpose(2, 0, 1)-np.min(Z, axis=2)) /
            (np.max(Z, axis=2)-np.min(Z, axis=2))
        ).transpose(1, 2, 0)
        Z = Z.astype(np.float32)
        z = np.random.uniform(0, 1, (input_cases, fps_num_points, 1))  # 1xProb.
        z = z.astype(np.float32)
        inputs = {'X': [X, np.ones([X.shape[0], 1])], 'z': Z}
        inputs_single = {'X': [X, np.ones([X.shape[0], 1])], 'z': z}
        neighborhood_spec = {
            "type": "sphere",
            "radius": 0.67,
            "separation_factor": 0.6
        }
        # Test FPS post-processor (1)
        dlpre = FurthestPointSubsamplingPreProcessor(
            num_points=fps_num_points,
            num_encoding_neighbors=8,
            fast=False,
            neighborhood=neighborhood_spec,
            receptive_field_oversampling=None,
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=-1
        )
        Xpre = dlpre({'X': X})
        passed = self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs
        )
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs,
            reducer=PredictionReducerFactory.make_from_dict({
                'reduce_strategy': {
                    'type': 'MeanPredReduceStrategy'
                }
            })
        )
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs,
            reducer=PredictionReducerFactory.make_from_dict({
                'reduce_strategy': {
                    'type': 'SumPredReduceStrategy'
                }
            })
        )
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs,
            reducer=PredictionReducerFactory.make_from_dict({
                'reduce_strategy': {
                    'type': 'MaxPredReduceStrategy'
                }
            })
        )
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs,
            reducer=PredictionReducerFactory.make_from_dict({
                'reduce_strategy': {
                    'type': 'EntropicPredReduceStrategy'
                }
            })
        )
        # Test FPS post-processor (2)
        dlpre = FurthestPointSubsamplingPreProcessor(
            num_points=fps_num_points,
            num_encoding_neighbors=8,
            fast=False,
            neighborhood=neighborhood_spec,
            receptive_field_oversampling=None,
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=1
        )
        Xpre = dlpre({'X': X})
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs
        )
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs,
            reducer=PredictionReducerFactory.make_from_dict({
                'reduce_strategy': {
                    'type': 'EntropicPredReduceStrategy'
                }
            })
        )
        # Test FPS post-processor (3)
        dlpre = FurthestPointSubsamplingPreProcessor(
            num_points=fps_num_points,
            num_encoding_neighbors=1,
            fast=False,
            neighborhood=neighborhood_spec,
            receptive_field_oversampling=None,
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=-1
        )
        Xpre = dlpre({'X': X})
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs
        )
        # Test FPS post-processor (4)
        dlpre = FurthestPointSubsamplingPreProcessor(
            num_points=fps_num_points,
            num_encoding_neighbors=8,
            fast=True,
            neighborhood=neighborhood_spec,
            receptive_field_oversampling=None,
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=-1
        )
        Xpre = dlpre({'X': X})
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs
        )
        # Test FPS post-processor (5)
        dlpre = FurthestPointSubsamplingPreProcessor(
            num_points=fps_num_points,
            num_encoding_neighbors=8,
            fast=False,
            neighborhood=neighborhood_spec,
            receptive_field_oversampling={
                "min_points": 2,
                "strategy": "nearest",
                "k": 3,
                "radius": 2.5
            },
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=-1
        )
        Xpre = dlpre({'X': X})
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs
        )
        # Test FPS post-processor (6)
        dlpre = FurthestPointSubsamplingPreProcessor(
            num_points=fps_num_points,
            num_encoding_neighbors=16,
            fast=False,
            neighborhood=neighborhood_spec,
            receptive_field_oversampling=None,
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=-1
        )
        Xpre = dlpre({'X': X})
        passed = passed and self.testDLPostProc(
            FurthestPointSubsamplingPostProcessor(dlpre),
            FurthestPointSubsamplingPostProcessorPP(dlpre),
            inputs_single
        )
        # Test Hierarchical FPS post-processor (1)
        dlpre = HierarchicalFPSPreProcessor(
            num_downsampling_neighbors=[1, 8, 8],
            num_pwise_neighbors=[8, 8, 8],
            num_upsampling_neighbors=[1, 8, 8],
            num_points_per_depth=[
                fps_num_points, fps_num_points//2, fps_num_points//4
            ],
            fast_flag_per_depth=[False, False, False],
            neighborhood=neighborhood_spec,
            receptive_field_oversampling=None,
            support_strategy_num_points=input_cases,
            support_strategy='fps',
            nthreads=-1
        )
        Xpre = dlpre({'X': X})
        passed = passed and self.testDLPostProc(
            HierarchicalFPSPostProcessor(dlpre),
            HierarchicalFPSPostProcessorPP(dlpre),
            inputs
        )
        passed = passed and self.testDLPostProc(
            HierarchicalFPSPostProcessor(dlpre),
            HierarchicalFPSPostProcessorPP(dlpre),
            inputs_single
        )
        passed = passed and self.testDLPostProc(
            HierarchicalFPSPostProcessor(dlpre),
            HierarchicalFPSPostProcessorPP(dlpre),
            inputs,
            reducer=PredictionReducerFactory.make_from_dict({
                'reduce_strategy': {
                    'type': 'MaxPredReduceStrategy'
                }
            })
        )
        passed = passed and self.testDLPostProc(
            HierarchicalFPSPostProcessor(dlpre),
            HierarchicalFPSPostProcessorPP(dlpre),
            inputs_single,
            reducer = PredictionReducerFactory.make_from_dict({
                'reduce_strategy': {
                    'type': 'MaxPredReduceStrategy'
                }
            })
        )
        # Return status (True if all tests passed, False otherwise)
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        return passed

    # ---   UTILS   --- #
    # ----------------- #
    def testDLPostProc(self, dlpost, dlpostpp, inputs, reducer=None):
        """
        Check that the C++ deep learning post-processor works as the python
        reference post-processor.

        :param dlpost: The object representing the Python-side post-processor.
        :param dlpostpp: The object representing the C++-side post-processor
            that corresponds with the Python-side post-processor (dlpost).
        :param inputs: The inputs for the post-processor.
        :type inputs: dict
        :param reducer: The prediction reducer for the post-processor, if any.
        :type reducer: :class:`.PredictionReducer`
        :return: True if the C++ post-processor matches the results of the
            Python post-processor, False otherwise.
        :rtype: bool
        """
        # Compute post-processors
        z = dlpost(inputs, reducer=reducer)  # Python post-processor
        zpp = dlpostpp(inputs, reducer=reducer)  # C++ post-processor
        if not np.allclose(z, zpp, atol=self.eps, rtol=0):
            return False
        # All checks passed
        return True
