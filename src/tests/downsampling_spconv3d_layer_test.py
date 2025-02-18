# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.downsampling_spconv3d_layer import \
    DownsamplingSpConv3DLayer
from src.model.deeplearn.layer.sparse_indexing_map_layer import \
    SparseIndexingMapLayer
from src.utils.ptransf.receptive_field_hierarchical_sg import \
    ReceptiveFieldHierarchicalSG
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class DownsamplingSpConv3DLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Downsampling 3D sparse convolution layer test that checks the operations of
    a :class:`.DownsamplingSpConv3DLayer` yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Downsampling SpConv3D layer test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run downsampling 3D sparse convolution layer test.

        :return: True if :class:`.DownsamplingSpConv3DLayer` works as expected
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
        wD = rf[0].get_downsampling_windows()[0]  # Downsamp. conv. window size
        h = [rfi.get_submanifold_maps()[0] for rfi in rf]  # Submanifold map h
        hk = [hi[0] for hi in h]
        hv = [hi[1] for hi in h]
        hD = [rfi.get_downsampling_vectors()[0] for rfi in rf]  # Downs. map hD
        nf = 4  # Input feature space dimensionality
        F = [
            np.vstack([
                np.zeros((1, nf)),
                np.random.normal(0, 1, (hi[0].shape[0], nf))/np.pi
            ]).astype(np.float32)
            for hi in h
        ]
        # Add padding to input
        max_rows = np.max([hDi.shape[0] for hDi in hD])
        max_src_rows = np.max([Fi.shape[0] for Fi in F])
        start_rows, start_src_rows = [], []
        k_offset = 0
        for i, Fi in enumerate(F):
            rows = hD[i].shape[0]
            padding = max_rows - rows
            start_rows.append(padding)
            pad_vec = [padding, 0]
            hD[i] = np.pad(
                hD[i] + k_offset,
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
        dsc3D = DownsamplingSpConv3DLayer(wD, f, nf, ng)
        # Compute DownsamplingSpConv3DLayer
        with tf.device("cpu:0"):
            N = [rfi.get_num_partitions()[:, 0] for rfi in rf]
            dsc3D.siml = SparseIndexingMapLayer()
            dsc3D.siml([tf.constant(hk), tf.constant(hv), tf.constant(F)])
            G = dsc3D([
                tf.constant(F),
                tf.constant(hD),
                tf.constant(N),
                tf.constant(start_rows),
                tf.constant(start_src_rows)
            ])
        # Validate
        valid = True
        W = dsc3D.W.numpy()
        K = len(hk)
        wDsq = wD*wD
        wD_to_nx = wDsq*wD
        for k in range(K):
            start_row = start_rows[k]
            start_src_row = start_src_rows[k]
            Fk = F[k][start_src_row:]
            hkk = hk[k][start_src_row:]
            hvk = hv[k][start_src_row:]
            h_k = dict(zip(hkk, hvk))
            hDk = hD[k][start_row:]
            Nk = N[k]
            ny, nz = Nk[1:]
            nynz = ny*nz
            Gk = np.zeros((hDk.shape[0]+1, ng), dtype=np.float32)
            for q, i in enumerate(hDk):
                for p in range(wD_to_nx):
                    jp = i
                    jp += p % wD
                    jp += (p//wD % wD)*nz
                    jp += (p//wDsq % wD)*nynz
                    jp = h_k.get(jp, 0)
                    Fkjp = Fk[jp].reshape((1, -1))
                    for Wl in W:
                        Gk[q+1] += (Fkjp @ Wl).flatten()
            valid = valid and np.all(
                np.abs(G[k].numpy()[start_row:]-Gk) <= self.eps
            )
        return valid
