# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.submanifold_spconv3d_layer import \
    SubmanifoldSpConv3DLayer
from src.utils.ptransf.receptive_field_hierarchical_sg import \
    ReceptiveFieldHierarchicalSG
from src.model.deeplearn.layer.sparse_indexing_map_layer import \
    SparseIndexingMapLayer
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class SubmanifoldSpConv3DLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Submanifold 3D sparse convolution layer test that checks the operations of
    a :class:`.SubmanifoldSpConv3DLayer` yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Submanifold SpConv3D layer test')
        self.eps = 0.5e-4

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run submanifold 3D sparse convolution layer test.

        :return: True if :class:`.SubmanifoldSpConv3DLayer` works as expected
            for the test cases, False otherwise.
        :rtype: bool
        """
        # Generate test data
        X = [
            np.vstack([
                np.random.normal(0, 0.1, (512, 3)),
                np.random.normal(-1, 0.1, (512, 3)),
                np.random.normal(1, 0.1, (512, 3))
            ]),
            np.vstack([
                np.random.normal(0, 1, (512, 3)),
                np.random.normal(-0.5, 0.1, (512, 3)),
                np.random.normal(0.5, 0.1, (512, 3))
            ])
        ]
        # Preprocess test data
        rf = [
            ReceptiveFieldHierarchicalSG(
                cell_size=0.1,
                submanifold_window=[1, 1],
                downsampling_window=[2],
                downsampling_stride=[2],
                upsampling_window=[2],
                upsampling_stride=[2]
            ).fit(Xi)
            for Xi in X
        ]
        w = rf[0].get_submanifold_windows()[0]  # Submanifold conv. window size
        h = [rfi.get_submanifold_maps()[0] for rfi in rf]  # Submanifold map h
        hk = [hi[0] for hi in h]
        hv = [hi[1] for hi in h]
        nf = 4  # Input feature space dimensionality
        F = [
            np.vstack([
                np.zeros((1, nf)),
                np.random.normal(0, 1, (hi[0].shape[0], nf))/np.pi
            ]).astype(np.float32)
            for hi in h
        ]
        # Add padding to input
        max_rows = np.max([Fi.shape[0] for Fi in F])
        start_rows = []
        k_offset = 0
        for i, Fi in enumerate(F):
            rows = Fi.shape[0]
            padding = max_rows-rows
            start_rows.append(padding)
            pad_mat = [[padding, 0], [0, 0]]
            pad_vec = [padding, 0]
            F[i] = np.pad(Fi, pad_mat, "constant", constant_values=0)
            hk[i] = np.pad(
                hk[i] + k_offset,
                pad_vec,
                "constant",
                constant_values=-1
            )
            k_offset = max(k_offset, 1+np.max(hk[i]))
            hv[i] = np.pad(hv[i], pad_vec, "constant", constant_values=0)
        # Instantiate SubmanifoldSpConv3DLayer
        ng = 5  # Output feature space dimensionality
        f = 7  # Number of convolutional filters
        ssc3D = SubmanifoldSpConv3DLayer(w, f, nf, ng)
        # Compute SubmanifoldSpConv3DLayer
        with tf.device("cpu:0"):
            N = [rfi.get_num_partitions()[:, 0] for rfi in rf]
            ssc3D.siml = SparseIndexingMapLayer()
            ssc3D.siml([tf.constant(hk), tf.constant(hv), tf.constant(F)])
            G = ssc3D([
                tf.constant(F),
                tf.constant(hk),
                tf.constant(N),
                tf.constant(start_rows)
            ])
        # Validate
        valid = True
        W = ssc3D.W.numpy()
        K = len(hk)
        wp = 2 * w + 1
        wpsq = wp*wp
        wp_to_nx = wpsq*wp
        for k in range(K):
            start_row = start_rows[k]
            Fk = F[k][start_row:]
            hkk = hk[k][start_row:]
            hvk = hv[k][start_row:]
            h_k = dict(zip(hkk, hvk))
            Nk = N[k]
            ny, nz = Nk[1:]
            nynz = ny*nz
            Gk = np.zeros((Fk.shape[0], ng), dtype=np.float32)
            shift = w * (1 + nz + nynz)
            for i in hkk:
                q = h_k[i]
                for p in range(wp_to_nx):
                    jp = i - shift
                    jp += p % wp
                    jp += (p//wp % wp)*nz
                    jp += (p//wpsq % wp)*nynz
                    jp = h_k.get(jp, 0)
                    Fkjp = Fk[jp].reshape((1, -1))
                    for Wl in W:
                        Gk[q] += (Fkjp @ Wl).flatten()
            valid = valid and np.all(
                np.abs(G[k].numpy()[start_row:]-Gk) <= self.eps
            )
        return valid