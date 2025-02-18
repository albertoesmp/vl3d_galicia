# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.shadow_dense_layer import ShadowDenseLayer
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class ShadowDenseLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Shadow dense layer test that checks the operations of a
    :class:`.ShadowDenseLayer` yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Shadow dense layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run shadow dense layer test.

        :return: True if :class:`.ShadowDenseLayer` works as expected for the
            test cases, False otherwise.
        :rtype: bool
        """
        # Generate test batch
        X = np.random.normal(0, 1, (16, 256, 8)).astype(np.float32)
        rag_X = [
            np.random.normal(0, 1, (128, 8)),
            np.random.normal(0, 1, (320, 8)),
            np.random.normal(0, 1, (256, 8)),
            np.random.normal(0, 1, (192, 8))
        ]
        rag_X_nopad = [np.array(rag_Xi) for rag_Xi in rag_X]
        # Add padding to input
        max_rows = np.max([rag_Xi.shape[0] for rag_Xi in rag_X])
        start = []
        for i, rag_Xi in enumerate(rag_X):
            rows = rag_Xi.shape[0]
            padding = max_rows-rows
            start.append(padding)
            pad = [[padding, 0], [0, 0]]
            rag_X[i] = np.pad(rag_Xi, pad, "constant", constant_values=0)
        # Dense layer spec
        units = 64
        dense_spec = {
            'activation': 'ReLU',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None
        }
        # Reference dense layer
        dl = tf.keras.layers.Dense(units, **dense_spec)
        dl(X)
        # Shadow dense layer to be tested
        rdl = ShadowDenseLayer(units, **dense_spec)
        rdl([tf.constant(X), tf.constant([0 for k in range(X.shape[0])])])
        rdl.bias = dl.bias
        rdl.kernel = dl.kernel
        # Validate with test batch
        valid = True
        with tf.device("cpu:0"):
            y_ref = dl(X).numpy()
            y = rdl([
                tf.constant(X),
                tf.constant([0 for k in range(X.shape[0])])
            ]).numpy()
            if np.any(np.abs(y-y_ref) > self.eps):
                valid = False
            # Validate with irregular batch
            rag_y_ref = [dl(rag_Xi) for rag_Xi in rag_X_nopad]
            rag_y = rdl([tf.constant(rag_X), tf.constant(start)])
            for k in range(len(rag_X)):
                if np.any(
                    np.abs(
                        rag_y[k][start[k]:].numpy()-rag_y_ref[k].numpy()
                    ) > self.eps
                ):
                    valid = False
        # Return
        return valid

