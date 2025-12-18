# ---   IMPORTS   --- #
# ------------------- #
from contextlib import nullcontext

from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.layer.spconv3d_encoding_layer import \
    SpConv3DEncodingLayer
from src.utils.ptransf.receptive_field_hierarchical_sg import \
    ReceptiveFieldHierarchicalSG
from src.model.deeplearn.layer.sparse_indexing_map_layer import \
    SparseIndexingMapLayer
from src.model.deeplearn.layer.submanifold_spconv3d_layer import \
    SubmanifoldSpConv3DLayer
from src.model.deeplearn.layer.downsampling_spconv3d_layer import \
    DownsamplingSpConv3DLayer
from src.model.deeplearn.layer.shadow_conv1d_layer import ShadowConv1DLayer
from src.model.deeplearn.layer.shadow_activation_layer import \
    ShadowActivationLayer
from src.model.deeplearn.layer.shadow_batch_normalization_layer import \
    ShadowBatchNormalizationLayer
import numpy as np
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class SpConv3DEncodingLayerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris PEna

    Sparse convolutional 3D encoding layer test that checks the operations of a
    :class:`.SpConv3DEncodingLayer` yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('SpConv3D encoding layer test')
        self.eps = 0.5e-4

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run sparse convolutional 3D encoding layer test.

        :return: True if :class:`.SpConv3DEncodingLayer` works as expected for
            the test cases, False otherwise.
        :rtype: bool
        """
        # Test hyperparameters
        hp = {
            'submanifold_window': 1,
            'submanifold_filters': 16,
            'Din': 4,
            'Dout': 32,
            'submanifold_initializer': 'glorot_normal',
            'submanifold_regularizer': None,
            'submanifold_constraint': None,
            'submanifold_bn_momentum': 0.9,
            'downsampling_window': 2,
            'downsampling_stride': 2,
            'downsampling_initializer': 'glorot_normal',
            'downsampling_regularizer': None,
            'downsampling_constraint': None,
            'downsampling_bn_momentum': 0.9,
            'spconvs_per_encoder': 2,
            'feature_dim_divisor': 2,
            'dim_transform_kernel_initializer': 'glorot_normal',
            'dim_transform_kernel_regularizer': None,
            'dim_transform_kernel_constraint': None,
            'dim_transform_activation': 'relu',
            'dim_transform_bn_momentum': 0.9
        }
        # Generate test structure space
        X = np.array([
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
        ])
        # Preprocess test data
        rf = [
            ReceptiveFieldHierarchicalSG(
                cell_size=0.1,
                submanifold_window=[hp['submanifold_window']]*2,
                downsampling_window=[hp['downsampling_window']],
                downsampling_stride=[hp['downsampling_stride']],
                upsampling_window=[hp['downsampling_window']],
                upsampling_stride=[hp['downsampling_stride']]
            ).fit(Xi)
            for Xi in X
        ]
        h = [rfi.get_submanifold_maps()[0] for rfi in rf]
        hk = [hi[0] for hi in h]
        hv = [hi[1] for hi in h]
        h2 = [rfi.get_submanifold_maps()[1] for rfi in rf]
        hk2 = [h2i[0] for h2i in h2]
        Hk = [hk, hk2]
        hD = [rfi.get_downsampling_vectors()[0] for rfi in rf]
        # Generate test feature space
        F = np.array([
            np.random.normal(0, 1, (hki.shape[0], hp['Din'])) for hki in hk
        ])
        # Compute start row for each element of the batch at each depth
        max_rows = [
            np.max([Hk[i][t].shape[0] for i in range(len(X))])
            for t in range(2)
        ]
        start = [[] for i in range(len(X))]
        for i in range(len(X)):
            starti = start[i]
            for t in range(2):
                max_rows_t = max_rows[t]
                rows = Hk[i][t].shape[0]
                padding = max_rows_t - rows
                starti.append(padding)
        start = np.array(start, dtype=np.int32)
        # Add padding to input
        k_offset = 0
        for i, Fi in enumerate(F):
            starti = start[i]
            F[i] = np.pad(
                Fi,
                [[starti[0], 0], [0, 0]],
                "constant",
                constant_values=0
            )
            hk[i] = np.pad(
                hk[i] + k_offset,
                [starti[0], 0],
                "constant",
                constant_values=-1
            )
            k_offset = max(k_offset, 1+np.max(hk[i]))
            hv[i] = np.pad(
                hv[i],
                [starti[0], 0],
                "constant",
                constant_values=0
            )
            hD[i] = np.pad(
                hD[i] + k_offset,
                [starti[1], 0],
                "constant",
                constant_values=0
            )
        # Build sparse indexing map layer
        siml = SparseIndexingMapLayer()
        siml([hk, hv, F])
        # Build layer-by-layer encoder
        lbl_enc = self.build_layer_by_layer_encoder(hp)
        # Build SpConv 3D encoding layer
        # TODO Rethink : Implement
        # Validate SpConv 3D encoding layer
        # TODO Rethink : Implement
        # Return true on success
        return True

    # ---   LAYER-BY-LAYER ENCODER   --- #
    # ---------------------------------- #
    def build_layer_by_layer_encoder(self, x, hk, n, start, siml, hp):
        """
        Build a sparse convolutional 3D encoder with many layers to validate
        the single-layer approach against it.

        :param x: Input for the model.
        :type x: :class:`tf.Tensor`
        :param hk: The keys of active cells.
        :type hk: :class:`np.ndarray` or :class:`tf.Tensor`
        :param n: The number of partitions along each axis.
        :type n: :class:`np.ndarray`
        :param start: The start row-indices for each element in the input
            batch.
        :type start: :class:`np.ndarray` or :class:`tf.Tensor`
        :param siml: The sparse indexing map layer to build the
            :class:`.SubmanifoldSpConv3DLayer` layers.
        :type siml: :class:`.SparseIndexingMapLayer`
        :param hp: The hyperparameters governing the encoder.
        :type hp: dict
        :return: Built sparse convolutional 3D encoder.
        :rtype: ????
        """
        # TODO Rethink : Finish docs
        output = {}
        # Build pre-wrap block
        DinTrans = hp['Din'] // hp['feature_dim_divisor']
        DoutTrans = hp['Dout'] // hp['feature_dim_divisor']
        x = ShadowConv1DLayer(
            DinTrans,
            1,
            kernel_initializer=hp['dim_transform_kernel_initializer'],
            kernel_regularizer=hp['dim_trasnform_kernel_regularizer'],
            kernel_constraint=hp['dim_Transform_kernel_constraint'],
            activation=hp['dim_transform_activation'],
            offset=1
        )([x, start])
        x = ShadowBatchNormalizationLayer(
            momentum=hp['dim_trasnform_bn_momentum'],
            offset=1
        )([x, start])
        x = ShadowActivationLayer(
            tf.keras.activations.relu,
            offset=1,
        )([x, start])
        # Build spconv block
        for d in range(hp['spconvs_per_encoder']):
            SubmanifoldSpConv3DLayer(
                hp['submanifold_window'],
                hp['submanifold_filters'],
                DinTrans,
                DoutTrans,
                W_initializer=hp['submanifold_initializer'],
                W_regularizer=hp['submanifold_regularizer'],
                W_constraint=hp['submanifold_constraint'],
                siml=siml
            )([x, hk, n, start])
            ShadowBatchNormalizationLayer(
                momentum=hp['submanifold_bn_momentum'],
                offset=1
            )([x, start])
            ShadowActivationLayer(
                tf.keras.activations.relu,
                offset=1
            )([x, start])
        # Build post-wrap block
        x = ShadowConv1DLayer(
            hp['Dout'],
            1,
            kernel_initializer=hp['dim_transform_kernel_initializer'],
            kernel_regularizer=hp['dim_transform_kernel_regularizer'],
            kernel_constraint=hp['dim_transform_kernel_constraint'],
            activation=hp['dim_transform_activation'],
            offset=1
        )([x, start])
        x = ShadowBatchNormalizationLayer(
            momentum=hp['dim_transform_bn_momentum'],
            offset=1
        )([x, start])
        x = ShadowActivationLayer(
            tf.keras.activations.relu,
            offset=1
        )([x, start])
        # Build downsamplin block
        # TODO Rethink : Implement
        # Return built block
        return x
        # TODO Rethink : Also track and return layers