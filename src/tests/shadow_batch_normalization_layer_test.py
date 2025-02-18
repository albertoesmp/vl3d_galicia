# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.shadow_batch_normalization_layer import \
    ShadowBatchNormalizationLayer
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class ShadowBatchNormalizationLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Shadow batch normalization layer test that checks the operations of a
    :class:`.ShadowBatchNormalizationLayer` yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Shadow batch normalization layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run shadow batch normalization layer test.

        :return: True if :class:`.ShadowBatchNormalizationLayer` works as
            expected for the test cases, False otherwise.
        :rtype: bool
        """
        # Generate test batch
        X1 = np.random.normal(0, 1, (16, 256, 8)).astype(np.float32)
        X2 = np.random.normal(0, 1, (16, 32)).astype(np.float32)
        rag_X1 = [
            np.random.normal(0, 1, (128, 8)),
            np.random.normal(0, 1, (320, 8)),
            np.random.normal(0, 1, (256, 8)),
            np.random.normal(0, 1, (192, 8))
        ]
        rag_X1_nopad = [np.array(rag_X1i) for rag_X1i in rag_X1]
        rag_X2 = [
            np.random.normal(0, 1, 32),
            np.random.normal(0, 1, 32),
            np.random.normal(0, 1, 32),
            np.random.normal(0, 1, 32)
        ]
        rag_X2_nopad = [np.array(rag_X2i) for rag_X2i in rag_X2]
        # Add padding to input
        max_rows1 = np.max([rag_X1i.shape[0] for rag_X1i in rag_X1])
        max_rows2 = np.max([rag_X2i.shape[0] for rag_X2i in rag_X2])
        start1, start2 = [], []
        for i, rag_Xi in enumerate(rag_X1):
            rows = rag_Xi.shape[0]
            padding = max_rows1 - rows
            start1.append(padding)
            pad = [[padding, 0], [0, 0]]
            rag_X1[i] = np.pad(rag_Xi, pad, "constant", constant_values=0)
        for i, rag_Xi in enumerate(rag_X2):
            rows = rag_Xi.shape[0]
            padding = max_rows2 - rows
            start2.append(padding)
            pad = [padding, 0]
            rag_X2[i] = np.pad(rag_Xi, pad, "constant", constant_values=0)
        # Batch normalization layer spec
        bn_spec = {
            'axis': -1,
            'momentum': 0.99,
            'epsilon': 0.001,
            'center': True,
            'scale': True,
            'beta_initializer': 'zeros',
            'gamma_initializer': 'ones',
            'moving_mean_initializer': 'ones',
            'moving_variance_initializer': 'ones',
            'beta_regularizer': None,
            'gamma_regularizer': None,
            'beta_constraint': None,
            'gamma_constraint': None,
            'synchronized': False
        }
        # Reference batch normalization layer
        bnl1 = tf.keras.layers.BatchNormalization(**bn_spec)
        for X1k in X1:
            bnl1(X1k, training=True)
        bnl2 = tf.keras.layers.BatchNormalization(**bn_spec)
        for X2k in X2:
            bnl2(X2k, training=True)
        # Shadow batch normalization layer to be tested
        rbnl1 = ShadowBatchNormalizationLayer(**bn_spec)
        rbnl1([tf.constant(X1), tf.constant([0 for k in range(X1.shape[0])])])
        rbnl2 = ShadowBatchNormalizationLayer(**bn_spec)
        rbnl2([tf.constant(X2), tf.constant([0 for k in range(X2.shape[0])])])
        # Validate
        valid = True
        with tf.device("cpu:0"):
            # Validate with test batch (1)
            y_ref = np.array([bnl1(X1k, training=True) for X1k in X1])
            y = rbnl1([
                tf.constant(X1),
                tf.constant([0 for k in range(X1.shape[0])])
            ]).numpy()
            if np.any(np.abs(y - y_ref) > self.eps):
                valid = False
            # Validate with test batch (2)
            y_ref = np.array([bnl2(X2k, training=True) for X2k in X2])
            y = rbnl2([
                tf.constant(X2),
                tf.constant([0 for k in range(X2.shape[0])])
            ]).numpy()
            if np.any(np.abs(y - y_ref) > self.eps):
                valid = False
            # Validate with irregular batch (1)
            rag_y_ref = [bnl1(rag_Xi, training=True) for rag_Xi in rag_X1_nopad]
            rag_y = rbnl1([tf.constant(rag_X1), tf.constant(start1)])
            for k in range(len(rag_X1)):
                if np.any(
                    np.abs(
                        rag_y[k][start1[k]:].numpy() - rag_y_ref[k].numpy()
                    ) > self.eps
                ):
                    valid = False
            # Validate with irregular batch (2)
            rag_y_ref = [bnl2(rag_Xi, training=True) for rag_Xi in rag_X2_nopad]
            rag_y = rbnl2([tf.constant(rag_X2), tf.constant(start2)])
            for k in range(len(rag_X2)):
                if np.any(
                    np.abs(
                        rag_y[k][start2[k]:].numpy() - rag_y_ref[k].numpy()
                    ) > self.eps
                ):
                    valid = False
        # Return
        return valid
