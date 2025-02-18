# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.light_kpconv_layer import LightKPConvLayer
from scipy.spatial import KDTree as KDT
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class LightKPConvLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Light kernel point convolution (LightKPConv) layer test that checks the
    operations of a light KPConv layer yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Light KPConv layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run light KPConv layer test.

        :return: True if :class:`.LightKPConvLayer` works as expected for the
            test cases, False otherwise.
        :rtype bool:
        """
        # Generate test data
        points_per_axis = 10
        num_points = points_per_axis ** 3
        num_near_neighs = 8
        num_features = 6
        dim_out = 7
        t = np.linspace(0, 2, points_per_axis)
        X1 = np.array([
            [x, y, z]
            for x in t for y in t for z in t
        ])
        F1 = np.random.normal(0, 1, (num_points, num_features))
        N1 = KDT(X1).query(X1, k=num_near_neighs)[1]
        X2 = X1 / np.array([1.0, 1.0, 2.0])
        F2 = np.random.normal(0, 1, (num_points, num_features))
        N2 = KDT(X2).query(X2, k=num_near_neighs)[1]
        inputs = [
            np.array([X1, X2], dtype='float32'),
            np.array([F1, F2], dtype='float32'),
            np.array([N1, N2], dtype='int')
        ]
        # Instantiate light KPConv layer
        lkpcl = LightKPConvLayer(
            sigma=0.5,
            kernel_radius=1.5,
            num_kernel_points=13,
            deformable=False,
            Dout=dim_out,
            A_initializer='GlorotNormal'
        )
        lkpcl.build([inputs[i].shape for i in range(len(inputs))])
        # Compute light KPConv layer
        with tf.device("cpu:0"):
            lkpcl_out = lkpcl.call(inputs)
        # Validate
        valid = True
        valid = valid and self.validate_no_activation(
            inputs, num_near_neighs, dim_out, lkpcl, lkpcl_out
        )
        return valid

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def validate_no_activation(self, inputs, nneighs, Dout, lkpcl, lkpcl_out):
        """
        Check whether the :class:`.LightKPConvLayer` yielded
        the expected output (True) or not (False).

        :param inputs: The inputs to the layer.
        :param nneighs: The number of neighbors per group.
        :param Dout: The output dimensionality.
        :param lkpcl: The layer.
        :type lkpcl: :class:`.LightKPConvLayer`
        :param lkpcl_out: The output of the layer.
        :return: True if the output is okay, False otherwise.
        """
        X_batch, F_batch, N_batch = inputs
        num_elems_in_batch = X_batch.shape[0]
        lkpcl_out = np.array(lkpcl_out)
        Q, W, A = list(map(np.array, [lkpcl.Q, lkpcl.W, lkpcl.A]))
        sigma = lkpcl.sigma
        mq = Q.shape[0]  # Number of points defining the kernel
        for batch in range(num_elems_in_batch):
            X, F, N = X_batch[batch], F_batch[batch], N_batch[batch]
            Fout = np.zeros((F.shape[0], Dout), dtype='float32')
            for i, xi in enumerate(X):  # For each point in the receptive field
                for j in N[i]:  # For each point in its neighborhood
                    weight_scales = np.zeros(W.shape[0])
                    for k in range(mq):
                        dist_weight = max(
                            0,
                            1 - np.linalg.norm(X[j]-X[i]-Q[k])/sigma
                        )
                        weight_scales += dist_weight * A[k, :]
                    weights = weight_scales * W.T
                    Fout[i] += weights @ F[j].reshape((-1, 1)).flatten()
            if not np.allclose(lkpcl_out[batch], Fout, atol=self.eps, rtol=0):
                return False
        return True
