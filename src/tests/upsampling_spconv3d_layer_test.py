# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.upsampling_spconv3d_layer import \
    UpsamplingSpConv3DLayer
from src.model.deeplearn.layer.sparse_indexing_map_layer import \
    SparseIndexingMapLayer
from src.utils.ptransf.receptive_field_hierarchical_sg import \
    ReceptiveFieldHierarchicalSG
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class UpsamplingSpConv3DLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Upsampling 3D sparse convolution layer test that checks the operations of
    a :class:`.UpsamplingSpConv3DLayer` yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Upsampling SpConv3D layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run upsampling 3D sparse convolution layer test.

        :return: True if :class:`.UpsamplingSpConv3DLayer` works as expected
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
        wU = rf[0].get_upsampling_windows()[0]  # Upsamp. conv. window size
        h = [rfi.get_submanifold_maps()[1] for rfi in rf]  # Submanifold map h
        hk = [hi[0] for hi in h]
        hv = [hi[1] for hi in h]
        hU = [rfi.get_upsampling_vectors()[0] for rfi in rf]  # Downs. map hU
        nf = 4  # Input feature space dimensionality
        F = [
            np.vstack([
                np.zeros((1, nf)),
                np.random.normal(0, 1, (hi[1].shape[0], nf))/np.pi
            ]).astype(np.float32)
            for hi in h
        ]
        # Add padding to input
        max_rows = np.max([hUi.shape[0] for hUi in hU])
        max_src_rows = np.max([Fi.shape[0] for Fi in F])
        start_rows, start_src_rows = [], []
        k_offset = 0
        for i, Fi in enumerate(F):
            rows = hU[i].shape[0]
            padding = max_rows - rows
            start_rows.append(padding)
            pad_vec = [padding, 0]
            # TODO Rethink : Handle k2_offset for hU
            hU[i] = np.pad(
                hU[i] + k_offset,
                pad_vec,
                "constant",
                constant_values=-1
            )
            src_rows = Fi.shape[0]
            src_padding = max_src_rows - src_rows
            start_src_rows.append(src_padding)
            src_pad_vec = [src_padding, 0]
            src_pad_mat = [[src_padding, 0], [0, 0]]
            F[i] = np.pad(Fi, src_pad_mat, "constant", constant_values=0)
            hk[i] = np.pad(
                hk[i] + k_offset,
                src_pad_vec,
                "constant",
                constant_values=-1
            )
            k_offset = max(k_offset, 1+np.max(hk[i]))
            hv[i] = np.pad(hv[i], src_pad_vec, "constant", constant_values=0)
        # Instantiate DownsamplingSpConv3DLayer
        ng = 5  # Output feature space dimensionality
        f = 7  # Number of convolutional filters
        usc3D = UpsamplingSpConv3DLayer(wU, f, nf, ng)
        # Compute UpsamplingSpConv3DLayer
        with tf.device("cpu:0"):
            N = [rfi.get_num_partitions()[:, 0] for rfi in rf]
            usc3D.siml = SparseIndexingMapLayer()
            usc3D.siml([tf.constant(hk), tf.constant(hv), tf.constant(F)])
            G = usc3D([
                tf.constant(F),
                tf.constant(hU),
                tf.constant(N),
                tf.constant(start_rows),
                tf.constant(start_src_rows)
            ])
        # Validate
        valid = True
        W = usc3D.W.numpy()
        K = len(hk)
        wUsq = wU*wU
        wU_to_nx = wUsq*wU
        for k in range(K):
            start_row = start_rows[k]
            start_src_row = start_src_rows[k]
            Fk = F[k][start_src_row:]
            hkk = hk[k][start_src_row:]
            hvk = hv[k][start_src_row:]
            h_k = dict(zip(hkk, hvk))
            hUk = hU[k][start_row:]
            Nk = N[k]
            ny, nz = Nk[1:]
            nynz = ny*nz
            Gk = np.zeros((hUk.shape[0]+1, ng), dtype=np.float32)
            for q, i in enumerate(hUk):
                for p in range(wU_to_nx):
                    jp = i
                    jp += p % wU
                    jp += (p//wU % wU)*nz
                    jp += (p//wUsq % wU)*nynz
                    jp = h_k.get(jp, 0)
                    Fkjp = Fk[jp].reshape((1, -1))
                    for Wl in W:
                        Gk[q+1] += (Fkjp @ Wl).flatten()
            valid = valid and np.all(
                np.abs(G[k].numpy()[start_row:]-Gk) <= self.eps
            )
        return valid
