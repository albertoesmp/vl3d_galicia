# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.hourglass_layer import HourglassLayer
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class HourglassLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris PEna

    Hourglass layer test that checks the operations of a Hourglass layer yield
    the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Hourglass layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run Hourglass layer test.

        :return: True if :class:`.HourglassLayer` works as expected for the
            test cases, False otherwise.
        :rtype: bool
        """
        # Generate test data
        batch_size = 4
        num_points = 128
        num_features = 8
        mid_features = 5
        out_features = 10
        F = np.random.normal(0, 1, (batch_size, num_points, num_features))
        # Instantiate hourglass
        hl = HourglassLayer(
            mid_features,
            out_features,
            activation='ReLU',
            regularize=True
        )
        hl.build(F.shape)
        # Compute hourglass layer
        with tf.device("cpu:0"):
            hl_out = hl(F)
        # Validate
        valid = True
        valid = valid and self.validate_hourglass(
            F,
            mid_features,
            out_features,
            hl,
            hl_out
        )
        return valid

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def validate_hourglass(self, F, mid_features, out_features, hl, hl_out):
        """
        Check whether the :class:`.HourglassLayer` yielded the expected
        output (True) or not (False). It also checks that the regularization
        computation is correct.

        :param f: The inputs to the layer.
        :param mid_features: The number of features in the internal
            representation.
        :param out_features: The number of output features.
        :param hl: The layer
        :type hl: :class:`.HourglassLayer`
        :param hl_out: The output of the layer.
        :return: True if the output is okay, False otherwise.
        """
        num_elems_in_batch = F.shape[0]
        F_batch = F
        hl_out = np.array(hl_out)
        hl_reg = hl._do_hourglass_regularization()
        W1, W2 = list(map(np.array, [hl.W1, hl.W2]))
        for batch in range(num_elems_in_batch):
            F = F_batch[batch]
            Y = np.array(
                hl.sigma2(hl.sigma(F@W1)@W2)
            )
            # Output validation
            if not np.allclose(hl_out[batch], Y, atol=self.eps, rtol=0):
                return False
            # Regularization validation
            reg = (W1.T@W1) / np.sqrt(np.max(np.linalg.eigvals(W1.T@W1)))
            reg -= np.eye(mid_features)
            reg = np.sqrt(np.sum(np.square(reg)))
            if np.abs(hl_reg-reg) > self.eps:
                return False
        return True

