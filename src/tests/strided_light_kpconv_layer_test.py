# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.strided_light_kpconv_layer import \
    StridedLightKPConvLayer
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
from scipy.spatial import KDTree as KDT
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class StridedLightKPConvLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Strided light kernel point convolution (Strided Light KPConv) layer test
    that checks the operation of a strided light KPConv yields the expected
    results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Strided Light KPConv layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run Strided Light KPConv layer test.

        :return: True if :class:`.StridedLightKPConvLayer` works as expected
            for the test cases, False otherwise.
        :rtype bool:
        """
        # Generate test data
        points_per_axis = 10
        num_points = points_per_axis ** 3
        num_near_neighs = 8
        num_features = 6
        dim_out = 7
        t = np.linspace(0, 2, points_per_axis)
        o = np.ones(3)
        X1a = np.array([
            [x, y, z]
            for x in t for y in t for z in t
        ])
        F1a = np.random.normal(0, 1, (num_points, num_features))
        X2a = X1a / np.array([1.0, 1.0, 2.0])
        F2a = np.random.normal(0, 1, (num_points, num_features))
        rf1 = ReceptiveFieldFPS(
            num_points=points_per_axis,
            num_encoding_neighbors=num_near_neighs,
            fast=False
        )
        rf1.fit(X1a, o)
        rf2 = ReceptiveFieldFPS(
            num_points=points_per_axis,
            num_encoding_neighbors=num_near_neighs,
            fast=False
        )
        rf2.fit(X2a, o)
        X1b = rf1.centroids_from_points(X1a)
        X2b = rf2.centroids_from_points(X2a)
        ND1 = rf1.N  # Downsampling indexing matrix
        ND2 = rf2.N
        inputs = [
            np.array([X1a, X2a], dtype='float32'),
            np.array([X1b, X2b], dtype='float32'),
            np.array([F1a, F2a], dtype='float32'),
            np.array([ND1, ND2], dtype='int')
        ]
        # Instantiate strided light KPConv layer
        slkpcl = StridedLightKPConvLayer(
            sigma=0.5,
            kernel_radius=1.5,
            num_kernel_points=13,
            deformable=False,
            Dout=dim_out
        )
        slkpcl.build([inputs[i].shape for i in range(len(inputs))])
        # Compute strided light KPConv layer
        with tf.device("cpu:0"):
            slkpcl_out = slkpcl.call(inputs)
        # Validate
        valid = True
        valid = valid and self.validate_no_activation(
            inputs, num_near_neighs, dim_out, slkpcl, slkpcl_out
        )
        return valid


    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def validate_no_activation(self, inputs, nneighs, Dout, slkpcl, slkpcl_out):
        """
        Check whether the :class:`.StridedLightKPConvLayer` yielded
        the expected output (True) or not (False).

        :param inputs: The inputs to the layer.
        :param nneighs: The number of neighbors per group.
        :param Dout: The output dimensionality.
        :param slkpcl: The layer.
        :type slkpcl: :class:`.StridedLightKPConvLayer`
        :param slkpcl_out: The output of the layer.
        :return: True if the output is okay, False otherwise
        :rtype: bool
        """
        Xa_batch, Xb_batch, Fa_batch, ND_batch = inputs
        num_elems_in_batch = Xa_batch.shape[0]
        skpcl_out = np.array(slkpcl_out)
        Q, W, A = list(map(np.array, [slkpcl.Q, slkpcl.W, slkpcl.A]))
        sigma = slkpcl.sigma
        mq = Q.shape[0]  # Number of points defining the kernel
        for batch in range(num_elems_in_batch):
            Xa, Xb = Xa_batch[batch], Xb_batch[batch]
            Fa, ND = Fa_batch[batch], ND_batch[batch]
            Fout = np.zeros((Xb.shape[0], Dout), dtype='float32')
            for i, xi in enumerate(Xb):  # For each point in the output R.F.
                for j in ND[i]:  # For each point in its neighborhood
                    weight_scales = np.zeros(W.shape[0])
                    for k in range(mq):
                        dist_weight = max(
                            0,
                            1 - np.linalg.norm(Xa[j]-Xb[i]-Q[k])/sigma
                        )
                        weight_scales += dist_weight * A[k, :]
                    weights = weight_scales * W.T
                    Fout[i] += weights @ Fa[j].reshape((-1, 1)).flatten()
            if not np.allclose(slkpcl_out[batch], Fout, atol=self.eps, rtol=0):
                return False
        return True
