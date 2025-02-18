# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.shadow_conv1d_layer import ShadowConv1DLayer
import src.main.main_logger as LOGGING
import numpy as np
import tensorflow as tf
import logging


# ---   CLASS   --- #
# ----------------- #
class ShadowConv1DLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Shadow Conv1D layer test that checks the operations of a
    :class:`.ShadowConv1DLayer` yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Shadow Conv1D layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run shadow 1-dimensional convolution layer test.

        :return: True if :class:`.ShadowConv1DLayer` works as expected for the
            test cases, False otherwise.
        :rtype: bool
        """
        LOGGING.LOGGER.setLevel(logging.CRITICAL)  # Disable logger during test
        # Generate test batch
        X = np.random.normal(0, 1, (16, 256, 8))
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
        # Test with first Conv1D layer spec
        valid = True
        conv1D_filters1, conv1D_kernel_size1 = 7, 1
        conv1D_spec1 = {
            'strides': 1,
            'padding': 'valid',
            'data_format': 'channels_last',
            'dilation_rate': 1,
            'groups': 1,
            'activation': None,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None
        }
        valid = valid and ShadowConv1DLayerTest.validate(
            conv1D_filters1,
            conv1D_kernel_size1,
            conv1D_spec1,
            X,
            rag_X,
            rag_X_nopad,
            start,
            self.eps
        )
        # Test with second Conv1D layer spec
        conv1D_filters2, conv1D_kernel_size2 = 7, 3
        conv1D_spec2 = {
            'strides': 1,
            'padding': 'valid',
            'data_format': 'channels_last',
            'dilation_rate': 1,
            'groups': 1,
            'activation': None,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'ones',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None
        }
        valid = valid and ShadowConv1DLayerTest.validate(
            conv1D_filters2,
            conv1D_kernel_size2,
            conv1D_spec2,
            X,
            rag_X,
            rag_X_nopad,
            start,
            self.eps
        )
        # Test with third Conv1D layer spec
        conv1D_filters3, conv1D_kernel_size3 = 7, 3
        conv1D_spec3 = {
            'strides': 1,
            'padding': 'valid',
            'data_format': 'channels_last',
            'dilation_rate': 1,
            'groups': 1,
            'activation': "ReLU",
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'glorot_normal',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None
        }
        valid = valid and ShadowConv1DLayerTest.validate(
            conv1D_filters3,
            conv1D_kernel_size3,
            conv1D_spec3,
            X,
            rag_X,
            rag_X_nopad,
            start,
            self.eps
        )
        # Return
        LOGGING.LOGGER.setLevel(logging.DEBUG)  # Restore logging
        return valid

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def validate(filters, kernel_size, spec, X, rag_X, rag_X_nopad, start, eps):
        # Reference conv1D layer
        cv = tf.keras.layers.Conv1D(filters, kernel_size, **spec)
        cv(X)
        # Shadow conv1D layer to be tested
        rcv = ShadowConv1DLayer(filters, kernel_size, **spec)
        rcv([tf.constant(X), tf.constant([0 for k in range(X.shape[0])])])
        rcv.conv1D.bias = cv.bias
        rcv.conv1D.kernel = cv.kernel
        # Validate with test batch
        with tf.device("cpu:0"):
            y_ref = cv(X).numpy()
            y = rcv([
                tf.constant(X),
                tf.constant([0 for k in range(X.shape[0])])
            ]).numpy()
            if np.any(np.abs(y-y_ref) > eps):
                return False
            # Validate with irregular batch
            rag_y_ref = [cv(rag_Xi[np.newaxis, :]) for rag_Xi in rag_X_nopad]
            rag_y = rcv([tf.constant(rag_X), tf.constant(start)])
            for k in range(len(rag_X)):
                if np.any(
                    np.abs(
                        rag_y[k][start[k]:].numpy() - rag_y_ref[k].numpy()
                ) > eps
                ):
                    return False
        # All checks passed
        return True
