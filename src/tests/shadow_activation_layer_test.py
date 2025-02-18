# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.shadow_activation_layer import \
    ShadowActivationLayer
import numpy as np
import tensorflow as tf

# ---   CLASS   --- #
# ----------------- #
class ShadowActivationLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Shadow activation layer test that checks the operations of a
    :class:`.ShadowActivationLayer` yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Shadow activation layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run shadow activation layer test.

        :return: True if :class:`.ShadowActivationLayer` works as expected for
            the test cases, False otherwise.
        :rtype: bool
        """
        # Generate test batches
        X1 = np.random.normal(0, 1, (16, 256, 8))
        X2 = np.random.normal(0, 1, (16, 8))
        X3 = np.random.normal(0, 1, (16, 32, 64, 8))
        rag_X1 = [
            np.random.normal(0, 1, (128, 8)),
            np.random.normal(0, 1, (320, 8)),
            np.random.normal(0, 1, (256, 8)),
            np.random.normal(0, 1, (192, 8))
        ]
        rag_X1_nopad = [np.array(rag_X1i) for rag_X1i in rag_X1]
        rag_X2 = [
            np.random.normal(0, 1, 128),
            np.random.normal(0, 1, 320),
            np.random.normal(0, 1, 256),
            np.random.normal(0, 1, 192)
        ]
        rag_X2_nopad = [np.array(rag_X2i) for rag_X2i in rag_X2]
        # Add padding to input
        max_rows1 = np.max([rag_X1i.shape[0] for rag_X1i in rag_X1])
        max_rows2 = np.max([rag_X2i.shape[0] for rag_X2i in rag_X2])
        start1, start2 = [], []
        start = [start1, start2]
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
        # Activation layer
        al = tf.keras.layers.Activation(tf.keras.activations.relu)
        # Shadow activation layer
        ral = ShadowActivationLayer(tf.keras.activations.relu)
        # Validate with test batches
        valid = True
        with tf.device("cpu:0"):
            # Validate with regular batches
            for Xi in [X1, X2, X3]:
                y_ref = al(Xi).numpy()
                y = ral([
                    tf.constant(Xi),
                    tf.constant([0 for k in range(Xi.shape[0])])
                ]).numpy()
                if np.any(np.abs(y-y_ref) > self.eps):
                    valid = False
            # Validate with irregular batches
            for i, (rag_Xi, rag_Xi_nopad) in enumerate(zip(
                [rag_X1, rag_X2], [rag_X1_nopad, rag_X2_nopad]
            )):
                rag_y_ref = [al(rag_Xij) for rag_Xij in rag_Xi_nopad]
                rag_y = ral([tf.constant(rag_Xi), tf.constant(start[i])])
                for k in range(len(rag_Xi)):
                    if np.any(
                        np.abs(
                            rag_y[k][start[i][k]:].numpy()-rag_y_ref[k].numpy()
                        ) > self.eps
                    ):
                        valid = False
        # Return
        return valid


